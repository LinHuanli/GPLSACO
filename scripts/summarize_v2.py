#!/usr/bin/env python3
"""从已有结果更新可读报告；不求解、不重新评分、不计算完整性摘要。"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.checkpoint import atomic_json  # noqa: E402

PROTOCOL = PROJECT / "artifacts/v2/protocol"
RESULTS = PROJECT / "results/v2"
REPORTS = PROJECT / "docs/reports"


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def performance():
    raw = PROJECT / "artifacts/v2/gpu-checks"
    reports = {
        name: read(raw / f"final-v2-{suffix}.json")
        for name, suffix in (("exact", "exact"), ("fp32", "fp32"), ("fp32_fast", "fp32-fast"))
    }
    if any(r is None or len(r["rows"]) != 4 for r in reports.values()):
        return None
    previous = read(raw / "unoptimized.json")
    if previous is None or len(previous["rows"]) != 4:
        return None
    old = {(r["dimension"], r["controller"]): r for r in previous["rows"]}
    values = {}
    for name, report in reports.items():
        values[name] = {(r["dimension"], r["controller"]): r for r in report["rows"]}
    rows = []
    for key, exact in values["exact"].items():
        rows.append(
            {
                "dimension": key[0],
                "controller": key[1],
                "unoptimized_seconds": old[key]["median_seconds"],
                "exact_speedup": old[key]["median_seconds"] / exact["median_seconds"],
                "backends": {
                    backend: {
                        field: groups[key][field]
                        for field in (
                            "median_seconds",
                            "seconds",
                            "cold_seconds",
                            "registration_seconds",
                            "fe_per_second",
                            "device_bytes",
                        )
                    }
                    for backend, groups in values.items()
                },
            }
        )
    value = {
        "implementation": "v2-warp-shared-distance-table-identity-zero",
        "device": reports["exact"]["device"],
        "host": reports["exact"]["host"],
        "gpu_uuid": reports["exact"]["gpu_uuid"],
        "iterations": 100,
        "ants": 128,
        "colonies": 32,
        "warmups": 1,
        "measures": 5,
        "measurement": "complete_solve_and_one_external_final_score_per_tour",
        "raw_directory": str(raw.relative_to(PROJECT)),
        "rows": rows,
    }
    atomic_json(RESULTS / "gpu_performance.json", value)
    return value


def performance_text(value):
    if value is None:
        return "最终三个数值后端的完整测速尚未全部完成，暂不填写最终加速比。"
    lines = [
        f"实测主机 {value['host']}，单张 RTX A5000；32 colonies、128 蚂蚁、100 迭代。",
        "每项 1 次预热、5 次测量，以下为包含最终外部评分的中位耗时；一次调用共 409600 FE。",
        "",
        "| 规模 / 控制器 | 优化前 exact | 当前 exact | exact 加速 | FP32 | FP32 fast-math |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in value["rows"]:
        controller = "Static 区域3" if row["controller"].startswith("static") else "固定代表性 GP"
        b = row["backends"]
        lines.append(
            f"| {row['dimension']} / {controller} | {row['unoptimized_seconds']:.3f} s | "
            f"{b['exact']['median_seconds']:.3f} s | {row['exact_speedup']:.2f}× | "
            f"{b['fp32']['median_seconds']:.3f} s | {b['fp32_fast']['median_seconds']:.3f} s |"
        )
    lines += [
        "",
        "| 规模 / 控制器 | exact FE/s | exact 冷启动 | 引擎显存 |",
        "|---|---:|---:|---:|",
    ]
    for row in value["rows"]:
        controller = "Static" if row["controller"].startswith("static") else "GP"
        b = row["backends"]["exact"]
        lines.append(
            f"| {row['dimension']} / {controller} | {b['fe_per_second']:,.0f} | "
            f"{b['cold_seconds']:.3f} s | {b['device_bytes'] / 2**20:.1f} MiB |"
        )
    lines += [
        "",
        "冷启动包括新面板注册准备和一次初次求解；GPU 显存为引擎数组，不含驱动上下文。",
        "以上比较包含性能优化及恒等 2-opt 假改善的修正；后者会改变搜索轨迹。",
        "FE 相同，实际 LS 工作量可能减少。这不是 GP 对 FACO 的质量提升结论。",
        "近似后端须另外通过完整 5000 迭代的质量门槛才可用于训练。",
        "",
        "以下是此前逐个体提交时的比例估算，**不代表新的种群并行代时**：",
        "",
        "| H | exact 每代 | exact 50代 | FP32 每代 | FP32 50代 |",
        "|---:|---:|---:|---:|---:|",
    ]
    sums = {
        b: sum(
            r["backends"][b]["median_seconds"]
            for r in value["rows"]
            if r["controller"] == "gp_representative"
        )
        for b in ("exact", "fp32")
    }
    for horizon in (100, 500, 1000, 5000):
        costs = {b: seconds * 128 * horizon / 100 for b, seconds in sums.items()}
        lines.append(
            f"| {horizon} | {costs['exact'] / 60:.1f} min | {costs['exact'] * 50 / 3600:.1f} h | "
            f"{costs['fp32'] / 60:.1f} min | {costs['fp32'] * 50 / 3600:.1f} h |"
        )
    lines += [
        "",
        "这是 100 迭代面板的比例估算，包含该测量内的固定调用开销；更长轨迹、不同 GP 动作、",
        "初始化及 CPU 调度都会影响实际速度。表中不含最终验证、监控、baseline、调参和测试，",
        "也不把近似候选尚未通过的状态解释为已选用 FP32。",
        "",
        "精简结果见 [gpu_performance.json](../../results/v2/gpu_performance.json)。",
    ]
    profiles = [read(RESULTS / f"kernel_profile_{b}.json") for b in ("exact", "fp32")]
    if all(profiles):
        lines += [
            "",
            "生产布局下单独 CUDA event 计时（100 迭代、32 colonies；毫秒）：",
            "",
            "| 后端 / 规模 / 控制器 | 构造与 LS | 档案与反馈 | GP/规则评分 |",
            "|---|---:|---:|---:|",
        ]
        for profile in profiles:
            for row in profile["rows"]:
                stages = row["stages_ms"]
                lines.append(
                    f"| {profile['numeric_backend']} / {row['dimension']} / {row['controller']} | "
                    f"{stages.get('construction_and_ls', 0):.2f} | "
                    f"{stages.get('archive_and_feedback', 0):.2f} | "
                    f"{stages.get('gp_scoring', 0):.2f} |"
                )
        lines += [
            "",
            "完整阶段数据见 [exact](../../results/v2/kernel_profile_exact.json)、",
            "[FP32](../../results/v2/kernel_profile_fp32.json)。这些计时均另跑一次同输入的无插桩求解，",
            "确认最终路线相同；实际训练不启用计时事件。",
        ]
    return "\n".join(lines)


def population_performance():
    raw = PROJECT / "artifacts/v2/gpu-checks"
    final = read(raw / "population-exact.json")
    name = (
        "population-exact.json"
        if final and final["status"] == "complete"
        else "population-exact-probe.json"
    )
    value = read(raw / name)
    if not value or value["status"] != "complete":
        return None
    summary = {key: item for key, item in value.items() if key != "programs"}
    summary["source"] = str((raw / name).relative_to(PROJECT))
    summary["measurement_stage"] = (
        "five_repeat_benchmark" if name == "population-exact.json" else "initial_probe"
    )
    summary["rows"] = [{k: v for k, v in row.items() if k != "instances"} for row in value["rows"]]
    atomic_json(RESULTS / "population_gpu_performance.json", summary)
    return summary


def population_text(value):
    if value is None:
        return "种群展开已实现，完整种群测量尚未完成。"
    lines = [
        f"单张 A5000（{value['host']}），真实首代 128 个体、每规模 16 实例×2 seeds、128 蚂蚁，",
        f"每次搜索 {value['iterations']} 迭代；1 次预热、{value['measures']} 次测量。"
        "两种入口均包含最终一次外部评分。",
        "",
        "| 规模 | 逐个体提交 | 整种群展开 | 加速 | 种群引擎显存 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in value["rows"]:
        lines.append(
            f"| {row['dimension']} | {row['serial_median_seconds']:.2f} s | "
            f"{row['population_median_seconds']:.2f} s | {row['speedup']:.2f}× | "
            f"{row['population_device_bytes'] / 2**30:.3f} GiB |"
        )
    lines += [
        "",
        f"两个规模的整代求解与评分合计 **{value['serial_generation_seconds']:.2f} → "
        f"{value['generation_seconds']:.2f} 秒，{value['generation_speedup']:.2f}×**。",
        "全部 8192 条最终路线、成本、反馈和实际工作量逐个体一致；总 FE 没有减少。",
        "此表不含进程启动、离线繁殖、日志写入和最终候选验证；",
        "它与上轮单程序优化使用不同控制器负载，不能直接相乘加速比。",
    ]
    if value["measurement_stage"] == "initial_probe":
        lines += [
            "",
            "这是短搜索初测；100 迭代、5 次重复的完整测速尚在进行。"
            "不能把它直接当成冻结训练 H 下的实测代时。",
        ]
    lines += [
        "",
        "精简记录见 "
        "[population_gpu_performance.json](../../results/v2/population_gpu_performance.json)。",
    ]
    check = read(RESULTS / "population_training_check.json")
    if check and check["status"] == "passed":
        lines += [
            "",
            f"真实 worker 另完成 {check['population']} 个体、{check['generations']} 代训练，"
            f"以及 {check['validation_candidates']} 个候选的独立验证；"
            f"共 {check['costs']['solve_jobs']} 个个体/规模/面板求解、"
            f"{check['costs']['search_tour_evaluations']:,} FE，"
            "覆盖整批暂停恢复，无失败、无重复评价。"
            "该工程检查训练用 H=10、验证用 H=20，仅使用开发数据，"
            "不替代正式训练、5000 迭代验证或 E1–E4 结果。",
            "记录见 "
            "[population_training_check.json](../../results/v2/population_training_check.json)。",
        ]
    return "\n".join(lines)


def status_text(value):
    if value["pipeline"]["stage"] == "campaign":
        return (
            "当前执行三 seed × 50 代自动队列；"
            "[实时状态、曲线与 FACO 对比见三次进化实验](三次进化实验.md)。"
            "旧的五 seed Full/NoFeedback 接续器已经移交，不再作为本轮入口。"
        )
    names = {
        "not_started": "前置实现与验证已准备，自动队列尚未启动",
        "benchmark": "完成三后端性能测量",
        "quality": "正在进行数值质量比较",
        "quality_complete": "数值选择完成",
        "curves": "正在计算 24 控制器的开发集迭代曲线",
        "curves_complete": "数值后端和训练迭代数已经冻结",
        "baselines": "正在准备完整训练、监控和验证的两类 FACO 对照",
        "baselines_complete_training_ready": "两类 FACO 对照完整，训练前置步骤完成",
        "baseline_tuning": "正在搜索完整 Static/Rule 配置族并验证选择",
        "training": "正在进行新的 E1 GP 训练/验证",
        "e1_gp_training_and_validation_complete_test_still_sealed": (
            "E1 训练和验证完成，正式测试仍未执行"
        ),
        "failed": "当前阶段报错，已保留记录并停止依赖阶段",
    }
    state = value["pipeline"]
    text = f"**当前阶段：{names.get(state['stage'], state['stage'])}。**"
    if "backend" in state:
        text += f" 当前计算后端为 `{state['backend']}`。"
    if "condition" in state:
        text += f" 条件 {state['condition']}，进化 seed {state['seed']}。"
    numeric, frozen = value["numeric"], value["frozen"]
    text += "\n\n" + (
        f"已选数值后端：`{numeric['numeric_backend']}`。" if numeric else "数值后端尚未完成选择。"
    )
    text += (
        f"训练 H={frozen['training_iterations']}，验证/测试均为 5000。"
        if frozen
        else "训练 H 尚未冻结，验证/测试固定为 5000。"
    )
    text += "所有阶段均按评价次数推进。"
    if state["stage"] in ("quality", "curves"):
        progress = value.get("stage_progress")
        if progress:
            total = 20 if state["stage"] == "quality" else 240
            text += (
                f"\n\n最近保存：当前后端已完成 {progress['completed_jobs']}/{total} 个 GPU 调用。"
            )
    elif state["stage"] == "baselines" and value.get("baseline_progress"):
        progress = value["baseline_progress"]
        text += f"\n\n最近保存的 baseline 状态：`{json.dumps(progress, ensure_ascii=False)}`。"
    queue = value.get("training_queue")
    if queue and queue.get("status") != "superseded":
        text += (
            f"\n\n已登记自动接续至训练/验证的进程：{queue['host']}、PID {queue['pid']}。"
            "先等待当前前置流程结束，再完成完整 Static/Rule 调参、"
            "Full 与 NoFeedback 各 5 次种群并行训练和验证；等待时不占 GPU。"
            "若前置流程失败，依赖训练停止。接续日志：`artifacts/v2/protocol/training-queue.log`。"
        )
    text += "\n\n实际追加进度见 `artifacts/v2/protocol/` 的阶段进度与日志。正式测试成绩尚未揭盲。"
    return text


def replace_block(path, name, body):
    content = path.read_text()
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    left, rest = content.split(start, 1)
    _, right = rest.split(end, 1)
    path.write_text(left + start + "\n" + body + "\n" + end + right)


def main():
    global PROTOCOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    PROTOCOL = args.directory.resolve()
    if not PROTOCOL.is_relative_to(PROJECT):
        raise ValueError("协议目录必须在项目内")
    now = datetime.now(ZoneInfo("Pacific/Auckland")).isoformat(timespec="seconds")
    perf = performance()
    population = population_performance()
    pipeline = read(PROTOCOL / "pipeline_status.json", {"stage": "not_started"})
    value = {
        "updated_at": now,
        "pipeline": pipeline,
        "numeric": read(PROTOCOL / "numeric_selection.json"),
        "frozen": read(PROTOCOL / "frozen.json"),
        "baseline_progress": read(PROTOCOL / "baseline_progress.json"),
        "baselines_ready": read(PROTOCOL / "baselines_ready.json"),
        "training_queue": read(PROTOCOL / "training_queue.json"),
        "verification": read(RESULTS / "verification.json"),
        "original_faco_runs": 180,
        "formal_test_completed": False,
    }
    if pipeline["stage"] in ("quality", "curves"):
        value["stage_progress"] = read(
            PROTOCOL / pipeline["stage"] / pipeline["backend"] / "progress.json"
        )
    atomic_json(RESULTS / "status.json", value)
    replace_block(REPORTS / "实验说明与结果.md", "performance", performance_text(perf))
    replace_block(REPORTS / "实验说明与结果.md", "population", population_text(population))
    replace_block(REPORTS / "实验说明与结果.md", "status", status_text(value))
    if pipeline["stage"] == "campaign":
        # 新队列负责自己的动态报告，旧汇总不能覆盖为五 seed 或单 GPU 状态。
        print(json.dumps({"updated_at": now, "stage": "campaign"}), flush=True)
        return
    verification = value["verification"]
    tests = (
        f"CUDA CTest {verification['ctest_passed']}/{verification['ctest_total']}；"
        f"真实 A5000 Python {verification['python_passed']} 通过、"
        f"{verification['python_skipped']} 跳过。"
        if verification
        else "当前检查的精简记录尚未生成。"
    )
    (REPORTS / "progress.md").write_text(
        f"# 当前研究进度\n\n更新于 {now}（新西兰）。详细参数和结果读 "
        "[实验说明与结果](实验说明与结果.md)，"
        "操作入口读 [v2 协议](../experiments/protocol_v2.md)。\n\n"
        + status_text(value)
        + "\n\n| 工作 | 可复核状态 |\n|---|---|\n"
        "| 原始 FACO 2022 | 六个实例各 30 次、共 180 次完成；论文蚂蚁数、5000 迭代、8 线程 |\n"
        "| 连续问题原始对照 | 连续距离/候选适配完成，两个开发实例接口检查通过 |\n"
        "| GPU 优化 | warp 与共享内存、距离复用、并行档案、取消生产逐批同步/重算 |\n"
        "| GP 种群并行 | 128 个体×32 colonies×128 蚂蚁在同一 A5000 展开；详见总报告第 9 节 |\n"
        "| 运行管理 | 普通编号、追加日志与代际快照；无完整性哈希；真实故障恢复检查通过 |\n"
        f"| 当前代码验证 | {tests} |\n"
        "| v1 训练 | 原始结果和检查点只读保留；不续接到 v2 |\n"
        "| 正式研究结论 | E1–E4 正式测试尚未完成，当前不能宣称 GP 优于 FACO |\n\n"
        "使用资源：只用空闲 RTX A5000，当前主流程最多使用一张；原始 FACO 为 CPU 对照。\n\n"
        "下一阶段由当前队列依次推进：数值选择 → 开发迭代曲线 → 两类配对 FACO。"
        "设置 `--through training` 时继续完整 Static/Rule 调参与新的 Full/NoFeedback 训练和验证。"
        "E2/E3/E4 保留完整科学范围，按相应协议继续，不能用工程检查替代正式结果。\n\n"
        "[原始 FACO 数字](../../results/v2/faco_2022_tsplib.json)、"
        "[当前机器状态](../../results/v2/status.json)。\n"
    )
    print(json.dumps({"updated_at": now, "stage": pipeline["stage"]}), flush=True)


if __name__ == "__main__":
    main()
