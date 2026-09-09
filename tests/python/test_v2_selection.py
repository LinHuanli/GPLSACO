"""选择门槛、配对单位与缓存预算隔离；避免通过删失败案例得到伪改进。"""

from dataclasses import replace

import pytest
from gp_faco.experiment_v2 import GPU_BASELINE, BaselineCache, choose_horizon, paired_quality
from gp_faco.training import TrainingRun
from test_v2_runtime import Farm, setup


def quality_rows():
    return [
        {
            "dimension": n,
            "controller": controller,
            "instance_id": f"n{n}-{i}",
            "seed": seed,
            "status": "completed",
            "gap_percent": 1.0,
        }
        for n in (500, 1000)
        for controller in ("fixed", "gp")
        for i in range(8)
        for seed in (17, 29)
    ]


def test_quality_uses_paired_instances_and_rejects_missing_or_failed_member():
    exact = quality_rows()
    better = [{**row, "gap_percent": 0.999} for row in exact]
    outcome = paired_quality(exact, better)
    assert outcome["passed"] and all(r["paired_instances"] == 8 for r in outcome["rows"])
    assert not paired_quality(exact, [{**r, "gap_percent": 1.02} for r in exact])["passed"]
    assert not paired_quality(exact, better[:-1])["passed"]
    better[3]["status"] = "failed"
    assert not paired_quality(exact, better)["passed"]


def test_horizon_requires_quality_and_rank_on_both_scales():
    panels = [
        {"dimension": n, "ids": [f"{n}-{i}" for i in range(3)], "seeds": [17, 29]}
        for n in (500, 1000)
    ]
    rows = [
        {
            "dimension": p["dimension"],
            "controller": c,
            "instance_id": name,
            "seed": seed,
            "iterations": it,
            "status": "completed",
            "gap_percent": cost,
        }
        for p in panels
        for name in p["ids"]
        for seed in p["seeds"]
        for c, final in (("a", 1.0), ("b", 2.0))
        for it, cost in ((0, 10.0), (100, final + 0.1), (5000, final))
    ]
    kwargs = {"controllers": ["a", "b"], "expected_panels": panels, "checkpoints": (0, 100, 5000)}
    assert choose_horizon(rows, **kwargs)["iterations"] == 100
    bad = [
        {**r, "gap_percent": 9.0} if r["dimension"] == 1000 and r["iterations"] == 100 else r
        for r in rows
    ]
    assert choose_horizon(bad, **kwargs)["iterations"] == 5000
    with pytest.raises(ValueError, match="完整"):
        choose_horizon(rows[:-1], **kwargs)
    failed = [dict(r) for r in rows]
    failed[0]["status"] = "failed"
    assert choose_horizon(failed, **kwargs)["reason"] == "failed_member"
    no_gain = [{**r, "gap_percent": 1.0} for r in rows]
    assert choose_horizon(no_gain, **kwargs)["iterations"] == 5000


def test_baseline_cache_keeps_seed_budget_and_backend_distinct(tmp_path):
    manifest = {
        "numeric_backend": "exact",
        "gpu_initialization": "gpu-1",
        "native_initialization": "cpu-1",
    }
    cache = BaselineCache(tmp_path / "baseline.sqlite", manifest)
    cache.put(
        GPU_BASELINE,
        500,
        "instance-1",
        2**64 - 1,
        500,
        cost=10,
        gap_percent=0.2,
        seconds=1,
        tour=[0, 1, 2],
    )
    cache.commit()
    assert cache.get(GPU_BASELINE, 500, "instance-1", 2**64 - 1, 500)["gap_percent"] == 0.2
    assert cache.get(GPU_BASELINE, 500, "instance-1", 2**64 - 1, 5000) is None
    assert cache.get(GPU_BASELINE, 500, "instance-1", 17, 500) is None
    cache.close()
    with pytest.raises(ValueError, match="不同"):
        BaselineCache(tmp_path / "baseline.sqlite", {**manifest, "numeric_backend": "fp32"})


def test_training_separates_validation_budget_and_uses_shared_monitoring(tmp_path):
    settings, protocol, data = setup()
    settings = replace(
        settings,
        budgets=((5, 64), (7, 64)),
        validation_budgets=((5, 192), (7, 192)),
        monitoring_every=2,
    )
    panels = [
        {"dimension": n, "ids": list(data.training[n][:2]), "seeds": [17, 29]} for n in (5, 7)
    ]
    data.shared_panels = [{"generation": g, "panels": panels} for g in range(3)]
    data.monitoring_panels = [
        {**p, "ids": list(data.validation[p["dimension"]][:2]), "evaluation_limit": 192}
        for p in panels
    ]

    class Cache:
        manifest = {"cache_id": "toy-baseline"}

        def panel(self, method, panel, iterations):
            return {(name, seed): 0.5 for name in panel["ids"] for seed in panel["seeds"]}

    data.baseline_cache = Cache()
    farm = Farm()
    tasks = []

    def factory(protocol):
        worker = farm(protocol)
        original = worker.submit

        def submit(task):
            tasks.append(task)
            return original(task)

        worker.submit = submit
        return worker

    report = TrainingRun(
        tmp_path / "shared", settings, protocol, data, worker_factory=factory
    ).run()
    assert report["status"] == "complete" and len(report["generation_metrics"]) == 3
    assert len(report["monitoring_results"]) == 1
    assert all(
        t.evaluation_limit_per_colony == 64 for t in tasks if ":generation" in t.occurrence_id
    )
    assert all(
        t.evaluation_limit_per_colony == 192
        for t in tasks
        if ":validation" in t.occurrence_id or ":monitor" in t.occurrence_id
    )
    assert all(
        row["paired_baselines"]["status"] == "completed"
        for row in report["generation_metrics"].values()
    )


def test_v2_keeps_full_declared_controller_families():
    import json
    from collections import Counter

    from gp_faco.baseline_policy import BaselinePolicy, policy_grid
    from gp_faco.worker import PROJECT

    value = json.loads((PROJECT / "configs/controller_families_v2.json").read_text())
    policies = policy_grid(value["tuning_policy_grid"])
    assert Counter(p.kind for p in policies) == {"static": 160, "rule": 616}
    assert len({p.identifier for p in policies}) == 776
    curves = [BaselinePolicy.from_dict(p) for p in value["calibration_policies"]]
    assert len(curves) == len({p.identifier for p in curves}) == 24
