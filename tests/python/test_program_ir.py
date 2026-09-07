import json
import random
import struct

import numpy as np
import pytest
from deap import gp
from gp_faco.primitives import FEATURE_NAMES, make_primitive_set
from gp_faco.program_ir import Program, choose_actions, evaluate, export_tree


def test_noncommutative_order_and_stable_ids():
    pset = make_primitive_set()
    tree = gp.PrimitiveTree.from_string("SUB(MUL(return_rate,restart),AQ(ls_work,mne_level))", pset)
    program = export_tree(tree)
    assert program.opcode == (0, 0, 4, 0, 0, 8, 3)
    assert program.operand == (2, 4, 0, 3, 5, 0, 0)
    features = np.zeros(12, dtype=np.float32)
    features[[2, 4, 3, 5]] = [0.5, 1.0, 0.75, 0.5]
    expected = 0.5 - 0.75 / np.sqrt(1.25)
    assert np.isclose(evaluate(program, features), expected, atol=1e-7)
    no_feedback = make_primitive_set(no_feedback=True)
    exported = export_tree(gp.PrimitiveTree.from_string("restart", no_feedback))
    assert exported.operand == (4,)
    assert not set(FEATURE_NAMES[1:4]) & set(no_feedback.arguments)


def test_each_intermediate_is_clipped_and_json_preserves_erc():
    pset = make_primitive_set()
    tree = gp.PrimitiveTree.from_string("SUB(MUL(MUL(2.0,2.0),MUL(2.0,2.0)),2.0)", pset)
    program = export_tree(tree)
    assert evaluate(program, np.zeros(12, dtype=np.float32)) == 6.0
    assert Program.from_dict(json.loads(program.to_json())).to_json() == program.to_json()
    tree = gp.PrimitiveTree.from_string("0.123456789", pset)
    program = export_tree(tree)
    assert program.constant_bits == (struct.unpack("<I", struct.pack("<f", 0.123456789))[0],)


@pytest.mark.parametrize(
    "values",
    [
        {"opcode": (), "operand": ()},
        {"opcode": (99,), "operand": (0,)},
        {"opcode": (0,), "operand": (12,)},
        {"opcode": (1,), "operand": (0,)},
        {"opcode": (3,), "operand": (0,)},
        {"opcode": (0, 0), "operand": (0, 1)},
        {"opcode": (0,) + (7,) * 6, "operand": (0,) * 7},
        {"opcode": (1,), "operand": (0,), "constant_bits": (0x7FC00000,)},
        {"opcode": (1,), "operand": (0,), "constant_bits": (0x40400000,)},
        {"opcode": (0,), "operand": (0,), "numeric_spec_id": 2},
    ],
)
def test_untrusted_ir_rejected(values):
    with pytest.raises(ValueError):
        Program(**values)


def test_random_trees_match_independent_deap_prefix_execution():
    random.seed(2309)
    rng = np.random.default_rng(513)
    features = rng.random((12, 7, 32), dtype=np.float32)
    for no_feedback in (False, True):
        pset = make_primitive_set(no_feedback)
        for _ in range(128):
            tree = gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=0, max_=5))
            program = export_tree(tree)
            oracle = gp.compile(tree, pset)
            arguments = [features[FEATURE_NAMES.index(name)] for name in pset.arguments]
            reference = np.broadcast_to(np.float32(oracle(*arguments)), (7, 32))
            np.testing.assert_array_equal(evaluate(program, features), reference)


def test_action_masks_and_global_epsilon_ties():
    scores = np.zeros((3, 32), dtype=np.float32)
    scores[0, :3] = [0, 0.75e-6, 1.5e-6]
    masks = np.array([7, 1 << 31, 0xFFFFFFFF], dtype=np.uint32)
    np.testing.assert_array_equal(choose_actions(scores, masks), [1, 31, 0])
    with pytest.raises(ValueError):
        choose_actions(scores, np.zeros(3, dtype=np.uint32))
