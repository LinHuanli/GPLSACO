import importlib
import os
import random

import numpy as np
import pytest
from deap import gp
from gp_faco.primitives import make_primitive_set
from gp_faco.program_ir import Program, choose_actions, evaluate, export_tree

if os.environ.get("GP_FACO_REQUIRE_NATIVE") == "1":
    extension = importlib.import_module("gp_faco_ext")
else:
    extension = pytest.importorskip("gp_faco_ext")

BACKENDS = ["score_cpu"]
if os.environ.get("GP_FACO_REQUIRE_CUDA") == "1":
    # 指定 CUDA 验收时不能以缺模块/缺设备的 skip 冒充通过。
    assert hasattr(extension, "score_cuda")
    BACKENDS.append("score_cuda")


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_scores_and_actions(backend):
    random.seed(4433)
    rng = np.random.default_rng(6677)
    features = rng.random((12, 7, 32), dtype=np.float32)
    masks = rng.integers(1, 0xFFFFFFFF, size=7, dtype=np.uint32)
    masks[:3] = [1, 1 << 31, 0xFFFFFFFF]
    pset = make_primitive_set()
    for _ in range(256):
        tree = gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=0, max_=5))
        program = export_tree(tree)
        output = getattr(extension, backend)(program.to_dict(), features, masks)
        reference = evaluate(program, features)
        np.testing.assert_allclose(output["scores"], reference, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(output["actions"], choose_actions(reference, masks))


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_input_validation_and_masked_ties(backend):
    score = getattr(extension, backend)
    features = np.zeros((12, 3, 32), dtype=np.float32)
    program = Program((0,), (4,)).to_dict()
    masks = np.array([0xFFFFFFFF, 1 << 31, 1 << 7], dtype=np.uint32)
    np.testing.assert_array_equal(score(program, features, masks)["actions"], [0, 31, 7])
    with pytest.raises((TypeError, ValueError)):
        score(program, features.astype(np.float64), masks)
    with pytest.raises((TypeError, ValueError)):
        score(program, features[:, :, ::-1], masks)
    with pytest.raises(ValueError):
        score(program, features, np.zeros(3, dtype=np.uint32))
    features[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        score(program, features, masks)
    features[0, 0, 0] = 0
    for malformed in (
        {**program, "ir_version": 4},
        {**program, "ir_version": True},
        {**program, "opcode": [False]},
        {**program, "operand": [4.0]},
        {**program, "opcode": [0] + [7] * 6, "operand": [0] * 7},
        {**program, "opcode": [0], "operand": [12]},
        {**program, "opcode": [1], "operand": [0], "constant_bits": [0x7FC00000]},
        {**program, "opcode": [0, 0, 3], "operand": [0, 1, 1]},
    ):
        with pytest.raises(ValueError):
            score(malformed, features, masks)
