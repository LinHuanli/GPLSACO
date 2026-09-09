#!/usr/bin/env python3
# ruff: noqa: E402
"""在空闲A5000检查条件控制、真实64情境和相同FE工作负载的额外开销。"""

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

import numpy as np
from deap import gp
from gp_faco.campaign import read, source
from gp_faco.checkpoint import atomic_json
from gp_faco.controller import Controller, score_controller
from gp_faco.diverse_evolution import DiverseEvolution
from gp_faco.evolution import Individual
from gp_faco.gpu_session import GpuSession
from gp_faco.program_ir import Program, evaluate, export_tree
from gp_faco.representation_pilot import (
    BUILD,
    DEFAULT_DIRECTORY,
    REPRESENTATIONS,
    SEEDS,
    initialize,
    load_contexts,
    solve_gpu_job,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    initialize(DEFAULT_DIRECTORY)
    session = GpuSession(args.gpu, "exact", extension_directory=BUILD)
    result = {"build": BUILD, "device": session.device}
    try:
        context_path = DEFAULT_DIRECTORY / "contexts.npz"
        if not context_path.exists():
            raw = solve_gpu_job(
                DEFAULT_DIRECTORY, {"kind": "baseline_gpu", "role": "contexts"}, session
            )
            atomic_json(DEFAULT_DIRECTORY / "contexts_collection.json", raw)
            rows = raw["contexts"]
            assert len(rows) == 64 and raw["total_fe"] == 2048000
            with context_path.open("wb") as handle:
                np.savez(
                    handle,
                    features=np.asarray([r["features"] for r in rows], dtype=np.float32).transpose(
                        1, 0, 2
                    ),
                    masks=np.asarray([r["legal_mask"] for r in rows], dtype=np.uint32),
                )
        contexts = load_contexts(DEFAULT_DIRECTORY)
        assert np.any(contexts.features[8:12] > 0)
        result["contexts"] = {
            "count": 64,
            "pre_decision_progress": sorted(np.unique(contexts.features[0]).tolist()),
            "legal_masks": sorted(np.unique(contexts.masks).tolist()),
            "separate_total_fe": 2048000,
        }
        print("64 real training contexts ready", flush=True)
        result["initialization"] = []
        result["cpu_aq_boundary_differences"] = []
        contexts.scorer = lambda c, f, m: session.native.score_controller_cuda(c.to_dict(), f, m)
        for representation in REPRESENTATIONS:
            for seed in SEEDS:
                e = DiverseEvolution(representation, seed, contexts)
                e.initialize()
                for p in e.population:
                    expected = score_controller(p.controller(), contexts.features, contexts.masks)
                    actual = session.native.score_controller_cuda(
                        p.controller().to_dict(), contexts.features, contexts.masks
                    )
                    # CPU独立做条件分支/合法性/并列决策，使用旧GPU算子的分数排除rsqrt舍入差异。
                    exact = score_controller(
                        p.controller(),
                        contexts.features,
                        contexts.masks,
                        evaluator=lambda program, f: session.native.score_cuda(
                            program.to_dict(), f, contexts.masks
                        )["scores"],
                    )
                    assert np.array_equal(exact["actions"], actual["actions"])
                    if not np.array_equal(expected["actions"], actual["actions"]):
                        assert any(8 in program.opcode for program in p.controller().trees)
                        result["cpu_aq_boundary_differences"].append(
                            {
                                "representation": representation,
                                "seed": seed,
                                "expressions": p.expressions(),
                                "contexts": np.flatnonzero(
                                    expected["actions"] != actual["actions"]
                                ).tolist(),
                            }
                        )
                    if representation == "joint_single":
                        assert np.allclose(
                            actual["scores"], expected["scores"], atol=3e-6, rtol=3e-6
                        )
                    if representation == "conditional_three":
                        for program in p.controller().trees:
                            values = session.native.score_cuda(
                                program.to_dict(), contexts.features, contexts.masks
                            )["scores"]
                            assert np.allclose(
                                values, evaluate(program, contexts.features), atol=3e-6, rtol=3e-6
                            )
                        for colony, stages in enumerate(actual["stages_by_colony"]):
                            for role, stage in enumerate(stages):
                                assert np.array_equal(
                                    stage["scores"], exact["stages"][role]["scores"][colony]
                                )
                                assert (
                                    stage["legal"]
                                    == exact["stages"][role]["legal"][colony].tolist()
                                )
                initial = e.metrics()
                e.assign([1.0] * 128)
                e.advance()
                result["initialization"].append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "initial": initial,
                        "tied_fitness_offspring": e.metrics(),
                    }
                )
                print(representation, seed, "128 CPU/CUDA controllers passed", flush=True)
        zero = Program((1,), (0,), (0,), feature_spec_id=2)
        mne = Program((0,), (5,), feature_spec_id=2)
        three = Controller("conditional_three", (zero, zero, mne), "three-fixed16")
        single = Controller("joint_single", (mne,), "single-fixed16")
        rng = np.random.default_rng(3301)
        random_masks = rng.integers(1, 0xFFFFFFFF, size=64, dtype=np.uint32)
        random_masks[:3] = [1 << 31, (1 << 2) | (1 << 11), 0xFFFF]
        for c in (three, single):
            expected = score_controller(c, contexts.features, random_masks)
            actual = session.native.score_controller_cuda(
                c.to_dict(), contexts.features, random_masks
            )
            assert np.array_equal(expected["actions"], actual["actions"])
        panel = read(DEFAULT_DIRECTORY / "panels.json")["training"][0]["panels"][0]
        pairs = [(name, panel["seeds"][0]) for name in panel["ids"]]
        with source() as data:
            problems = {name: data.load_instance(name) for name, _ in pairs}
            session.set_colonies(2)
            legacy, _ = session.solve(
                problems, pairs[:2], 12, mne.to_dict(), decision_iterations=(1, 4, 12)
            )
            conditional, _ = session.solve(
                problems, pairs[:2], 12, three.to_dict(), decision_iterations=(1, 4, 12)
            )
            population, _ = session.solve_population(
                problems, pairs[:2], 12, [single.to_dict(), three.to_dict()]
            )

            def tours(r):
                return [i["tour"] for i in r["items"]]

            assert (
                tours(legacy) == tours(conditional) == tours(population[0]) == tours(population[1])
            )
            assert legacy["total_tour_evaluations"] == conditional["total_tour_evaluations"] == 3072
            for row in conditional["decisions"]:
                assert "scores" not in row and [len(s["scores"]) for s in row["stages"]] == [
                    2,
                    4,
                    4,
                ]
                f = np.asarray(row["features"], dtype=np.float32)[:, None, :]
                action = score_controller(
                    three, f, np.asarray([row["legal_mask"]], dtype=np.uint32)
                )["actions"][0]
                assert row["action"] == int(action)
            result["engine_parity"] = {
                "old_single_new_single_three_and_population_tours_equal": True,
                "decision_samples": conditional["decisions"],
                "fe_per_controller": 3072,
            }
            if args.benchmark:
                # 相同零评分、相同动作、相同路线工作量；各表示总节点均为63。
                def zeros(leaves):
                    return (
                        "0.0"
                        if leaves == 1
                        else f"ADD({zeros(leaves // 2)}, {zeros(leaves - leaves // 2)})"
                    )

                pset = e.psets[0]
                big = export_tree(Individual(gp.PrimitiveTree.from_string(zeros(32), pset), 2))
                medium = export_tree(Individual(gp.PrimitiveTree.from_string(zeros(11), pset), 2))
                bundle = Controller(
                    "conditional_three", (medium, medium, medium), "benchmark-three"
                )
                session.set_colonies(16)
                # 小预热独立记账，不拼入正式训练FE。
                session.solve_population(problems, pairs, 2, [big.to_dict()] * 128)
                times = {}
                previous = None
                for name, controllers in (
                    ("legacy_single_63_nodes", [big.to_dict()] * 128),
                    ("conditional_three_total_63_nodes", [bundle.to_dict()] * 128),
                ):
                    raw, timing = session.solve_population(problems, pairs, 1000, controllers)
                    assert sum(r["total_tour_evaluations"] for r in raw) == 262144000
                    current = [tours(r) for r in raw]
                    if previous is not None:
                        assert current == previous
                    previous = current
                    times[name] = timing
                    print(name, timing, flush=True)
                result["benchmark"] = {
                    "times": times,
                    "population": 128,
                    "instances": 16,
                    "iterations": 1000,
                    "ants": 128,
                    "fe_each": 262144000,
                    "warmup_fe": 524288,
                    "identical_final_tours": True,
                    "extra_fraction": times["conditional_three_total_63_nodes"]["solve_seconds"]
                    / times["legacy_single_63_nodes"]["solve_seconds"]
                    - 1,
                }
        result["status"] = "passed"
        atomic_json(PROJECT / "results/v3/representation_cuda.json", result)
        print("representation CUDA checks passed", flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
