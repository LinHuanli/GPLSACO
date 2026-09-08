#!/usr/bin/env python3
"""从冻结先验和真实返回重建图、逐边检查Hard路线与FE账目；不读取标签。"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.candidate_prior import PriorSettings, parse_candidates  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import Instance, tour_cost  # noqa: E402
from gp_faco.graph_matching import match_graphs  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import content_hash, file_hash  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((args.inputs / "manifest.json").read_text())
    run = json.loads((args.results / "manifest.json").read_text())
    completed = json.loads((args.results / "completed.json").read_text())
    require(inputs["status"] == "complete", "准备尚未完成")
    require(
        run["input_manifest_sha256"] == file_hash(args.inputs / "manifest.json"), "输入身份改变"
    )
    require(run["native_binary_sha256"] == inputs["native_binary_sha256"], "二进制身份改变")
    require(run["colonies"] == run["ants"] == 32, "固定形状改变")
    require(run["wall_clock_limit"] is inputs["wall_clock_limit"] is None, "出现墙钟限额")
    require(len(inputs["files"]) == 32 and len(completed["calls"]) == 18, "任务未完整执行")
    for name, digest in inputs["sources"].items():
        require(file_hash(PROJECT / name) == digest, "准备或执行源码身份改变")
    entries, graph_rows = {}, []
    lkh_wall, lkh_cpu, prep_wall = 0.0, 0.0, 0.0
    for entry in inputs["files"]:
        path = args.inputs / entry["path"]
        require(file_hash(path) == entry["sha256"], "缓存图资料改变")
        value = json.loads(path.read_text())
        problem = Instance(
            value["instance_id"], tuple(map(tuple, value["coordinates"])), value["distance_spec"]
        )
        n, i = entry["dimension"], entry["index"]
        require(n == problem.dimension and (n, i) not in entries, "实例规模或身份重复")
        require(
            abs(tour_cost(problem, value["common"]["tour"]) - value["common"]["cost"]) < 1e-10,
            "共同初始tour成本不符",
        )
        priors = {}
        for kind in ("ALPHA", "POPMUSIC"):
            directory = args.inputs / f"prior-{n}-{i}-{kind}"
            manifest = json.loads((directory / "manifest.json").read_text())
            raw_path = directory / "native-candidates.json"
            prior = parse_candidates(
                problem, PriorSettings(**manifest["settings"]), json.loads(raw_path.read_text())
            )
            prior.update(
                manifest_sha256=content_hash(manifest), native_output_sha256=file_hash(raw_path)
            )
            require(prior == json.loads((directory / "prior.json").read_text()), "原始先验复核不符")
            for filename, field in (
                ("problem.tsp", "problem_file_sha256"),
                ("parameters.par", "parameter_file_sha256"),
            ):
                require(file_hash(directory / filename) == manifest[field], "原生先验输入改变")
            resources = json.loads((directory / "resources.json").read_text())
            require(
                resources["exit_code"] == 0
                and resources["log_sha256"] == file_hash(directory / "native.log"),
                "候选进程失败",
            )
            lkh_wall += resources["wall_seconds"]
            lkh_cpu += resources["cpu_user_seconds"] + resources["cpu_system_seconds"]
            priors[kind] = prior
        graphs = match_graphs(problem, tuple(value["common"]["tour"]), priors)
        require(graphs == value["graphs"], "冻结图与独立重建不一致")
        require(
            len(graphs["ALPHA"]["edges"]) == len(graphs["POPMUSIC"]["edges"]), "实际图边数不匹配"
        )
        for kind, graph in graphs.items():
            graph_rows.append(
                {
                    "dimension": n,
                    "index": i,
                    "kind": kind,
                    "graph_sha256": graph["sha256"],
                    "undirected_edges": len(graph["edges"]),
                    "discarded_extra_edges": graph["discarded_extra_edges"],
                    "degree_histogram": dict(sorted(Counter(graph["degrees"]).items())),
                    "real_slots": {
                        name: sum(node < n for row in graph[name] for node in row)
                        for name in ("primary", "backup", "ls")
                    },
                }
            )
        entries[n, i] = (problem, value)
        prep_wall += value["preparation_wall_seconds"]
    for n in (500, 1000):
        names = {entries[n, i][0].instance_id for i in range(16)}
        require(len(names) == 16, "同规模实例不独立")

    tours, edges, fe, replays, maximum_error = 0, 0, 0, 0, 0.0
    earlier, native_wall, construction, ls_checks, rejected = {}, 0.0, 0, 0, 0
    expected = []
    for n in (500, 1000):
        expected.append((n, "unrestricted", 0, "cached", "gp"))
        for kind in ("ALPHA", "POPMUSIC"):
            expected.extend(
                (n, kind, *row)
                for row in (
                    (0, "cached", "gp"),
                    (128, "cached", "gp"),
                    (64, "cached", "baseline"),
                    (128, "end_to_end", "gp"),
                )
            )
    for position, (entry, identity) in enumerate(zip(completed["calls"], expected, strict=True)):
        path = args.results / entry["path"]
        require(
            path.name == f"call-{position:02}.json" and file_hash(path) == entry["sha256"],
            "原始返回身份或顺序改变",
        )
        record = json.loads(path.read_text())
        n, kind, limit, mode, controller = identity
        require(
            tuple(record[k] for k in ("dimension", "kind", "limit", "mode", "controller"))
            == identity,
            "预登记调用顺序改变",
        )
        require(record["seeds"] == [17, 29] * 16, "共同求解seed改变")
        offset = 16 if kind == "POPMUSIC" else 0
        require(
            record["keys"] == [i + 1 + offset for i in range(16) for _ in (17, 29)],
            "colony与图key不匹配",
        )
        result = record["result"]
        controller_spec = (
            Program((0,), (4,), feature_spec_id=2).to_dict()
            if controller == "gp"
            else BaselinePolicy(
                mne_level=3,
                max_mne_level=3,
                region=3,
                restart_mode="bernoulli",
                restart_probability=1.0,
            ).to_dict()
        )
        require(record["controller_spec"] == controller_spec, "控制器规范身份改变")
        if controller == "baseline":
            require(result["baseline_policy"] == controller_spec, "原生基线身份改变")
        for stage in ("before", "after"):
            boundary = record[stage]
            device = next(csv.reader(boundary["device_csv"].splitlines()))
            require(
                device[0].strip() == run["gpu_uuid"] and boundary["pid"] == run["before"]["pid"],
                "设备或执行进程改变",
            )
            require(
                not any(
                    row and row[0].strip() == run["gpu_uuid"] and int(row[1]) != boundary["pid"]
                    for row in boundary["processes"]
                ),
                "GPU任务边界出现外来进程",
            )
        require(
            result["preparation_completed"] and result["preparation_mode"] == mode,
            "准备阶段未完整执行",
        )
        require(
            result["completed_batches"] == result["launched_batches"] == limit // 32
            and result["evaluation_limit_per_colony"]
            == result["completed_tour_evaluations_per_colony"]
            == limit
            and result["total_tour_evaluations"] == limit * 32,
            "FE账目不符",
        )
        require(
            result["budget_seconds"] is None
            and result["discarded_batches"]
            == result["overrun_seconds"]
            == result["charged_seconds"]
            == 0,
            "存在隐藏时间扣费/丢批",
        )
        require(
            result["budget_kind"] == "search_tour_evaluations" and len(result["items"]) == 32,
            "预算类型或返回形状改变",
        )
        if kind != "unrestricted":
            require(
                result["constraint_mode"] == "hard" and result["graph_spec_id"] == 1, "约束语义错误"
            )
            require(
                result["preparation_scope"] == "engine_only; external candidate graph cached",
                "外部准备费用被误称包含在Engine内",
            )
        for c, item in enumerate(result["items"]):
            problem, value = entries[n, c // 2]
            require(item["has_incumbent"], "没有完整incumbent")
            actual = tour_cost(problem, item["tour"])
            error = abs(actual - item["cost"])
            maximum_error = max(maximum_error, error)
            require(
                error < 1e-10 and actual <= value["common"]["cost"] + 1e-10,
                "返回成本错误或劣于共同初始化",
            )
            if not limit:
                require(
                    item["tour"] == value["common"]["tour"]
                    and item["cost"] == value["common"]["cost"],
                    "0 FE共同初始解改变",
                )
            if kind != "unrestricted":
                graph = value["graphs"][kind]
                allowed = set(map(tuple, graph["edges"]))
                require(result["graph_edges_per_colony"][c] == len(allowed), "CSR图身份混用")
                tour = item["tour"]
                require(
                    all(tuple(sorted((tour[j - 1], tour[j]))) in allowed for j in range(n)),
                    "Hard tour出现图外边",
                )
                edges += n
            tours += 1
        if controller == "gp" and limit:
            if mode == "cached":
                earlier[n, kind] = result
            else:
                previous = earlier[n, kind]
                for name in (
                    "control_states",
                    "completed_construction_steps",
                    "completed_ls_evaluations",
                    "completed_constraint_rejections",
                ):
                    require(previous[name] == result[name], "控制器复用后状态或工作量不一致")
                require(
                    [(v["tour"], v["cost"]) for v in previous["items"]]
                    == [(v["tour"], v["cost"]) for v in result["items"]],
                    "重复运行路线不一致",
                )
                replays += 1
        native_wall += record["native_wall_seconds"]
        construction += result["completed_construction_steps"]
        ls_checks += result["completed_ls_evaluations"]
        rejected += result.get("completed_constraint_rejections", 0)
        fe += result["total_tour_evaluations"]
    require((tours, fe, replays) == (576, 40960, 4), "完整验收覆盖不足")
    report = {
        "status": "passed",
        "scope": "engineering_development; Hard integration only",
        "instances": 32,
        "prior_calls": 64,
        "matched_graph_pairs": 32,
        "solve_calls": 18,
        "returned_tours_verified": tours,
        "hard_edges_verified": edges,
        "search_tour_evaluations": fe,
        "maximum_cost_error": maximum_error,
        "exact_replay_pairs": replays,
        "completed_construction_steps": construction,
        "completed_ls_evaluations": ls_checks,
        "completed_constraint_rejections": rejected,
        "native_call_wall_seconds": native_wall,
        "lkh_child_wall_seconds": lkh_wall,
        "lkh_child_cpu_seconds": lkh_cpu,
        "whole_preparation_wall_seconds_sum": prep_wall,
        "resource_note": "LKH child time is nested in whole preparation, not additive; "
        "CLI logs separately",
        "native_binary_sha256": run["native_binary_sha256"],
        "gpu_uuid": run["gpu_uuid"],
        "input_manifest_sha256": file_hash(args.inputs / "manifest.json"),
        "run_manifest_sha256": file_hash(args.results / "manifest.json"),
        "call_receipts": completed["calls"],
        "graphs": graph_rows,
        "limitation": "Complete ant/archive traces verified by separate C++ checks. "
        "Escape and E3 efficacy pending.",
    }
    atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("graphs", "call_receipts")}))


if __name__ == "__main__":
    main()
