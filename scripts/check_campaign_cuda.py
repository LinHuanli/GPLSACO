#!/usr/bin/env python3
"""A5000 上核对分片、单程序与稀疏决策采样；可与另一张卡的记录直接比较。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.campaign import BUILD, read, source  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.evolution import Evolution, EvolutionSettings  # noqa: E402
from gp_faco.gpu_session import GpuSession  # noqa: E402
from gp_faco.program_ir import export_tree  # noqa: E402


def signature(native):
    return {
        "items": [{"tour": row["tour"], "cost": row["cost"]} for row in native["items"]],
        **{
            key: native[key]
            for key in (
                "control_states",
                "completed_construction_steps",
                "completed_ls_evaluations",
                "total_tour_evaluations",
            )
        },
    }


def check(gpu, build=BUILD):
    evolution = Evolution(EvolutionSettings(population=8, elites=2, feature_spec_id=2), 1103)
    evolution.initialize()
    programs = [export_tree(p).to_dict() for p in evolution.population]
    session = GpuSession(gpu, "exact", colonies=4, extension_directory=build)
    record = {"device": session.device, "build": build, "scales": {}, "status": "passed"}
    try:
        with source() as dataset:
            for n in (500, 1000):
                ids = dataset.record_ids("development", n)[80:82]
                problems = {name: dataset.load_instance(name) for name in ids}
                pairs = [(name, seed) for name in ids for seed in (17, 29)]
                population, _ = session.solve_population(problems, pairs, 20, programs)
                signatures = [signature(r) for r in population]
                for index, program in enumerate(programs):
                    serial, _ = session.solve(problems, pairs, 20, program)
                    if signature(serial) != signatures[index]:
                        raise AssertionError("种群与逐程序入口不一致")
                    for (name, _), item in zip(pairs, serial["items"], strict=True):
                        tour_cost(problems[name], item["tour"])
                partitioned = []
                for values in (programs[:3], programs[3:]):
                    result, _ = session.solve_population(problems, pairs, 20, values)
                    partitioned.extend(signature(r) for r in result)
                if partitioned != signatures:
                    raise AssertionError("分块改变了随机轨迹或工作量")
                trace, _ = session.solve(
                    problems, pairs, 20, programs[0], decision_iterations=[1, 10, 20]
                )
                if signature(trace) != signatures[0] or len(trace["decisions"]) != 12:
                    raise AssertionError("稀疏决策采样改变了求解或缺失样本")
                for row in trace["decisions"]:
                    features = np.asarray(row["features"], dtype=np.float32).reshape(12, 1, 32)
                    masks = np.asarray([row["legal_mask"]], dtype=np.uint32)
                    score = session.native.score_cuda(programs[0], features, masks)
                    if int(score["actions"][0]) != row["action"]:
                        raise AssertionError("实际输入不能复现选中的动作")
                    actual = [float(v) if np.isfinite(v) else None for v in score["scores"][0]]
                    if actual != row["scores"]:
                        raise AssertionError("记录的32动作评分与真实输入不符")
                record["scales"][str(n)] = signatures
        return record
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--build", default=BUILD)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(PROJECT):
        parser.error("检查输出必须在项目内")
    result = check(args.gpu, args.build)
    if args.compare:
        previous = read(args.compare)
        if previous["device"]["gpu_uuid"] == args.gpu:
            raise AssertionError("跨卡检查需要两张不同的 A5000")
        if previous["scales"] != result["scales"]:
            raise AssertionError("两张 A5000 的完整路线、反馈、工作量和 FE 不同")
        result["cross_gpu_equal"] = True
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "scales"}), flush=True)


if __name__ == "__main__":
    main()
