#!/usr/bin/env python3
"""重读开发坐标、重建v2图并核验Hard/Escape真实返回；不启动GPU或读取标签。"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from check_graph_matching import verify as verify_graphs  # noqa: E402
from check_matched_graph_engine import load_inputs  # noqa: E402
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.graph_matching import match_graphs  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import file_hash  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "results", "database", "dataset-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    matched, original, records = load_inputs(args.inputs)
    run = json.loads((args.results / "manifest.json").read_text())
    completed = json.loads((args.results / "completed.json").read_text())
    require(run["status"] == completed["status"] == "returned_unverified", "真实任务没有完整返回")
    require(
        run["input_manifest_sha256"] == file_hash(args.inputs / "manifest.json"),
        "输入manifest身份改变",
    )
    require(
        run["database_sha256"] == original["database_sha256"] == file_hash(args.database),
        "开发数据库身份改变",
    )
    require(
        run["colonies"] == run["solver"]["ants"] == 32 and run["wall_clock_limit"] is None,
        "固定形状或次数协议改变",
    )
    require(run["solver"] == original["solver"], "共同求解器参数改变")
    require(run["matching_spec_id"] == 2 and run["escape_spec_id"] == 1, "图/例外规格改变")
    for name, digest in run["sources"].items():
        require(file_hash(PROJECT / name) == digest, "执行源码身份改变")
    problems, graphs, slot_counts = {}, {}, []
    with IndexedDataset(args.database, args.dataset_root) as source:
        for n in (500, 1000):
            expected_ids = sorted(source.record_ids("development", n))[:16]
            for i in range(16):
                entry = records[n, i]
                value, matched_value = entry["original"], entry["matched"]
                require(value["instance_id"] == expected_ids[i], "输入不属于冻结开发面板")
                problem = source.load_instance(expected_ids[i])
                require(
                    problem.coordinates == tuple(map(tuple, value["coordinates"])),
                    "缓存坐标与源文件不同",
                )
                require(
                    abs(tour_cost(problem, value["common"]["tour"]) - value["common"]["cost"])
                    < 1e-10,
                    "共同初解成本不正确",
                )
                priors = {}
                for name in matched_value["source_priors_sha256"]:
                    prior = json.loads((PROJECT / name).read_text())
                    priors[prior["settings"]["kind"]] = prior
                rebuilt = match_graphs(problem, tuple(value["common"]["tour"]), priors)
                require(rebuilt == matched_value["graphs"], "重建图与冻结v2图不同")
                checked = verify_graphs(problem, rebuilt, priors, value["graphs"])
                require(checked == matched_value["checked"], "实际槽位或E0复核不一致")
                problems[n, i], graphs[n, i] = problem, rebuilt
                slot_counts.append({"dimension": n, "index": i, **checked})

    expected = []
    for n in (500, 1000):
        expected.append((n, "unrestricted", "unrestricted", 0, "cached", "gp"))
        for constraint in ("hard", "escape"):
            for kind in ("ALPHA", "POPMUSIC"):
                expected.extend(
                    (n, kind, constraint, *tail)
                    for tail in (
                        (0, "cached", "gp"),
                        (128, "cached", "gp"),
                        (64, "cached", "baseline"),
                        (128, "end_to_end", "gp"),
                    )
                )
    require(
        len(completed["calls"]) == len(expected) == run["expected_calls"] == 34, "调用矩阵不完整"
    )
    require(len(run["registrations"]) == 160, "原生准备记录不完整")
    fe, tours, hard_edge_checks, maximum_error, replays = 0, 0, 0, 0.0, 0
    earlier, memory, counters, outside, native_seconds = {}, {}, Counter(), Counter(), 0.0
    for index, (reference, identity) in enumerate(zip(completed["calls"], expected, strict=True)):
        path = args.results / reference["path"]
        require(
            path.name == f"call-{index:02}.json" and file_hash(path) == reference["sha256"],
            "调用返回身份改变",
        )
        record = json.loads(path.read_text())
        require(
            tuple(
                record[k]
                for k in ("dimension", "kind", "constraint_mode", "limit", "mode", "controller")
            )
            == identity,
            "预登记调用次序改变",
        )
        n, kind, constraint, limit, mode, controller = identity
        offset = 16 if kind == "POPMUSIC" else 0
        require(
            record["seeds"] == [17, 29] * 16
            and record["keys"] == [i + offset + 1 for i in range(16) for _ in (17, 29)],
            "图身份/colony/seed映射改变",
        )
        policy = (
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
        require(record["controller_spec"] == policy, "未训练工程控制器被更换")
        for stage in ("before", "after"):
            boundary = record[stage]
            device = next(csv.reader(boundary["device_csv"].splitlines()))
            require(
                device[0].strip() == run["gpu_uuid"] and boundary["pid"] == run["before"]["pid"],
                "GPU/进程身份改变",
            )
            require(
                not any(
                    row and row[0].strip() == run["gpu_uuid"] and int(row[1]) != boundary["pid"]
                    for row in boundary["processes"]
                ),
                "任务边界存在外来GPU进程",
            )
        result = record["result"]
        require(
            all(
                type(result[name]) is int
                for name in (
                    "completed_batches",
                    "launched_batches",
                    "discarded_batches",
                    "evaluation_limit_per_colony",
                    "completed_tour_evaluations_per_colony",
                    "total_tour_evaluations",
                    "completed_construction_steps",
                    "completed_ls_evaluations",
                )
            ),
            "工作计数不是精确整数",
        )
        require(
            result["budget_seconds"] is None
            and result["budget_kind"] == "search_tour_evaluations"
            and result["discarded_batches"]
            == result["overrun_seconds"]
            == result["charged_seconds"]
            == 0,
            "出现隐藏时间限额或丢批",
        )
        require(
            result["completed_batches"] == result["launched_batches"] == limit // 32
            and result["evaluation_limit_per_colony"]
            == result["completed_tour_evaluations_per_colony"]
            == limit
            and result["total_tour_evaluations"] == limit * 32
            and len(result["items"]) == 32,
            "FE账目或面板大小错误",
        )
        require(
            result["preparation_completed"] and result["preparation_mode"] == mode, "原生准备不完整"
        )
        if controller == "baseline":
            require(result["baseline_policy"] == policy, "原生基线身份不同")
        if constraint != "unrestricted":
            require(
                result["constraint_mode"] == constraint
                and result["graph_spec_id"] == 1
                and result["preparation_scope"] == "engine_only; external candidate graph cached",
                "约束身份或准备计费范围错误",
            )
            values = (result["allocated_device_bytes"], result["reserved_escape_device_bytes"])
            require(
                values[1] > 0 and (n not in memory or memory[n] == values),
                "Hard/Escape设备容量预算不匹配",
            )
            memory[n] = values
        if constraint == "escape":
            current = result["escape_counters"]
            require(
                result["escape_spec_id"] == 1
                and result["escape_edge_capacity_per_ant"] == 64
                and all(type(value) is int and value >= 0 for value in current.values()),
                "例外规格或计数格式不正确",
            )
            require(
                current["new_edges"] <= result["total_tour_evaluations"] * 64
                and current["construction_gates"] <= current["construction_opportunities"]
                and result["completed_construction_steps"]
                <= current["construction_opportunities"]
                <= result["completed_construction_steps"] + result["total_tour_evaluations"],
                "例外次数超过固定容量/机会",
            )
            if not limit:
                require(not any(current.values()), "零FE仍存在例外活动")
            counters.update(current)
        for c, item in enumerate(result["items"]):
            problem, value = problems[n, c // 2], records[n, c // 2]["original"]
            require(item["has_incumbent"], "没有返回完整tour")
            actual = tour_cost(problem, item["tour"])
            error = abs(actual - item["cost"])
            require(
                error < 1e-10 and actual <= value["common"]["cost"] + 1e-10,
                "返回成本或incumbent不正确",
            )
            maximum_error = max(maximum_error, error)
            if not limit:
                require(
                    item["tour"] == value["common"]["tour"]
                    and item["cost"] == value["common"]["cost"],
                    "共同初解改变",
                )
            if constraint != "unrestricted":
                allowed = set(map(tuple, graphs[n, c // 2][kind]["edges"]))
                require(result["graph_edges_per_colony"][c] == len(allowed), "完整CSR身份不正确")
                tour = item["tour"]
                out = sum(tuple(sorted((tour[j - 1], tour[j]))) not in allowed for j in range(n))
                outside[f"{n}-{kind}-{constraint}"] += out
                if constraint == "hard":
                    require(out == 0, "Hard返回tour出现图外边")
                    hard_edge_checks += n
            tours += 1
        if controller == "gp" and limit:
            key = (n, kind, constraint)
            if mode == "cached":
                earlier[key] = result
            else:
                prior = earlier[key]
                names = [
                    "control_states",
                    "completed_construction_steps",
                    "completed_ls_evaluations",
                    "completed_constraint_rejections",
                ]
                if constraint == "escape":
                    names.append("escape_counters")
                require(
                    all(prior[name] == result[name] for name in names)
                    and [(v["tour"], v["cost"]) for v in prior["items"]]
                    == [(v["tour"], v["cost"]) for v in result["items"]],
                    "跨控制器/准备模式重放不一致",
                )
                replays += 1
        fe += result["total_tour_evaluations"]
        native_seconds += record["native_wall_seconds"]
    require((fe, tours, replays) == (81920, 1088, 8), "覆盖量不完整")
    report = {
        "status": "passed",
        "scope": "engineering_development; no efficacy or test conclusion",
        "formal_E3_complete": False,
        "instances": 32,
        "matched_nodes": matched["matched_nodes"],
        "solve_calls": 34,
        "returned_tours_verified": tours,
        "search_tour_evaluations": fe,
        "hard_edges_verified": hard_edge_checks,
        "exact_replay_pairs": replays,
        "maximum_cost_error": maximum_error,
        "escape_counters": dict(counters),
        "returned_outside_edge_occurrences": dict(outside),
        "native_call_wall_seconds": native_seconds,
        "device_memory_by_dimension": memory,
        "host": run["host"],
        "gpu_uuid": run["gpu_uuid"],
        "device": run["before"]["device_csv"].strip(),
        "native_binary_sha256": run["native_binary_sha256"],
        "input_manifest_sha256": file_hash(args.inputs / "manifest.json"),
        "run_manifest_sha256": file_hash(args.results / "manifest.json"),
        "source_sha256": file_hash(Path(__file__)),
        "call_receipts": completed["calls"],
        "graphs": slot_counts,
        "limits": [
            "Returned tour costs and Hard edges checked here; every accepted Escape move/footprint "
            "is checked in separate C++ tests.",
            "LKH and graph matching are reused with their earlier receipts; engine EndToEnd "
            "is not a new full LKH preparation measurement.",
            "Untrained controllers at development budgets; formal independent training "
            "and paired tests are still required.",
        ],
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("call_receipts", "graphs")}
        )
    )


if __name__ == "__main__":
    main()
