"""曲线移交保留原始结果，分片齐备之前不选择训练预算。"""

import json

import pytest
from gp_faco.campaign_calibration import calibration_jobs, import_calibration, summarize_calibration
from gp_faco.checkpoint import atomic_json


def test_calibration_import_preserves_completed_rows_and_incomplete_tail(tmp_path):
    origin, campaign = tmp_path / "origin", tmp_path / "campaign"
    origin.mkdir()
    campaign.mkdir()
    manifest = {
        "stage": "curves",
        "backend": "exact",
        "panels": [{"dimension": 500, "ids": ["a"], "seeds": list(range(32))}],
        "controllers": {"policy-01": {"kind": "static"}},
        "iterations": 5000,
        "checkpoints": [0, 5000],
    }
    atomic_json(origin / "manifest.json", manifest)
    job = next(calibration_jobs(manifest))
    rows = [
        {
            "status": "completed",
            "dimension": 500,
            "controller": "policy-01",
            "instance_id": "a",
            "seed": seed,
            "iterations": it,
            "gap_percent": 1.0,
        }
        for seed in range(32)
        for it in (0, 5000)
    ]
    record = {
        "job_id": job["source_job_id"],
        "rows": rows,
        "native_result": {"total_tour_evaluations": 20480000, "tour": [1, 2]},
        "timing": {"solve_seconds": 3.0},
        "end_to_end_seconds": 4.0,
        "device": {"model": "NVIDIA RTX A5000"},
    }
    original = json.dumps(record).encode() + b'\n{"incomplete":'
    (origin / "results.jsonl").write_bytes(original)
    receipt = import_calibration(campaign, origin)
    assert receipt["completed_jobs"] == 1 and receipt["incomplete_trailing_bytes"] > 0
    assert (origin / "results.jsonl").read_bytes() == original
    path = campaign / "jobs" / job["id"] / "result.json"
    result = json.loads(path.read_text())
    assert result["rows"] == rows
    assert result["raw_record"]["offset"] == 0
    assert "native_result" not in result
    assert import_calibration(campaign, origin) == receipt
    summary = summarize_calibration(campaign, [result])
    assert summary["total_fe"] == 20480000 and summary["selection"]["iterations"] == 5000
    with pytest.raises(ValueError, match="不完整|尚未完整"):
        summarize_calibration(campaign, [{**result, "rows": rows[:-1]}])
