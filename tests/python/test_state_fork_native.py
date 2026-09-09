"""状态文件跨Python对象与Engine重建后仍保留次数轨迹。"""

import hashlib
import json
import os

import numpy as np
import pytest
from gp_faco.baseline_policy import BaselinePolicy

native = pytest.importorskip("gp_faco_ext")
pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_RUN_GPU_TESTS") != "1", reason="需要显式空闲GPU"
)

PROGRAM = {
    "ir_version": 1,
    "numeric_spec_id": 1,
    "feature_spec_id": 2,
    "opcode": [0, 0, 2, 0, 0, 3, 0, 4, 2],
    "operand": [4, 8, 0, 0, 2, 0, 5, 0, 0],
    "constant_bits": [],
}


@pytest.fixture
def setup():
    settings = native.FixedFacoSettings()
    settings.ants = 8
    xy = np.random.default_rng(763).random((67, 2))
    keys = np.array([1, 1], dtype=np.uint64)
    seeds = np.array([17, 29], dtype=np.uint64)

    def make(colonies=2):
        engine = native.FacoBatchEngine(67, colonies, settings)
        engine.register_problem(1, xy)
        return engine

    return make, keys, seeds, BaselinePolicy(mne_level=2, max_mne_level=2, region=1).to_dict()


def outcome(result):
    return {
        "solutions": [(item["tour"], item["cost"]) for item in result["items"]],
        "control_states": result["control_states"],
    }


@pytest.mark.parametrize("kind", ["program", "baseline"])
def test_persistent_snapshot_rebuilds_native_run(setup, tmp_path, kind):
    make, keys, seeds, policy = setup
    argument = PROGRAM if kind == "program" else policy
    engine = make()
    reference = getattr(engine, f"evaluate_{kind}_evaluations")(keys, seeds, 64, argument)
    captured = getattr(engine, f"capture_{kind}_state")(keys, seeds, 64, 24, argument)
    metadata = captured["state"].describe()
    assert metadata["progress_evaluation_limit"] == 64 and metadata["completed_batches"] == 3
    assert len(metadata["buffers"]) == 40
    assert (
        sum(b["bytes"] for b in metadata["buffers"])
        == captured["evaluation"]["allocated_device_bytes"]
    )
    assert [m["return_rate"] for m in metadata["members"]] == [
        m["return_rate"] for m in captured["evaluation"]["control_states"]
    ]
    json.dumps(metadata, allow_nan=False)
    path = tmp_path / "snapshot.bin"
    path.write_bytes(captured["state"].to_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    del engine, captured
    restored = native.CountedState.from_bytes(path.read_bytes())
    assert restored.describe() == metadata
    engine = make()
    engine.evaluate_baseline_evaluations(keys, seeds, 32, policy)
    result = getattr(engine, f"continue_{kind}_state")(restored, 40, argument)
    assert outcome(result) == outcome(reference)
    assert result["completed_tour_evaluations_per_colony"] == 40
    assert result["total_tour_evaluations"] == 80
    assert result["budget_seconds"] is None and result["charged_seconds"] == 0
    assert result["overrun_seconds"] == 0 and result["discarded_batches"] == 0
    assert hashlib.sha256(restored.to_bytes()).hexdigest() == digest
    selected = restored.select_colony(1)
    one = getattr(make(1), f"continue_{kind}_state")(selected, 40, argument)
    assert outcome(one)["solutions"][0] == outcome(reference)["solutions"][1]


def test_paired_branch_order_and_serialized_state_are_independent(setup):
    make, keys, seeds, policy = setup
    engine = make()
    saved = engine.capture_program_state(keys, seeds, 64, 32, PROGRAM)["state"]
    original = saved.to_bytes()
    small = engine.continue_baseline_state(saved, 32, policy, fork_seed=53, region=0, mne=2)
    large = engine.continue_baseline_state(saved, 32, policy, fork_seed=53, region=0, mne=16)
    rebuilt = native.CountedState.from_bytes(original)
    repeated = make().continue_baseline_state(rebuilt, 32, policy, fork_seed=53, region=0, mne=2)
    assert outcome(small) == outcome(repeated)
    assert small["total_tour_evaluations"] == large["total_tour_evaluations"] == 64
    assert saved.to_bytes() == original


def test_snapshot_corruption_and_invalid_binding_arguments_are_rejected(setup):
    make, keys, seeds, policy = setup
    engine = make()
    saved = engine.capture_baseline_state(keys, seeds, 64, 32, policy)["state"]
    original = saved.to_bytes()
    corrupted = bytearray(original)
    corrupted[len(corrupted) // 2] ^= 1
    for value in (b"", original[:-1], bytes(corrupted), original + b"extra"):
        with pytest.raises(ValueError, match="快照"):
            native.CountedState.from_bytes(value)
    for value in (True, -1, 1.5, 1 << 64):
        with pytest.raises(ValueError):
            engine.continue_baseline_state(saved, 32, policy, fork_seed=value)
        with pytest.raises(ValueError):
            engine.capture_baseline_state(keys, seeds, 64, value, policy)
    with pytest.raises(ValueError):
        saved.select_colony(True)
    with pytest.raises(ValueError):
        engine.continue_baseline_state(saved, 0, policy, fork_seed=1)
    assert saved.to_bytes() == original
