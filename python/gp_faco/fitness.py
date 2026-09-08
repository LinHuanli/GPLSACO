"""协调进程中的独立路线核验与两规模宏平均；标签不进入worker任务。"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from gp_faco.data import Label, tour_cost
from gp_faco.worker import BaselineTask, FactorialTask, SolveTask, WorkerProtocol


@dataclass(frozen=True)
class PanelFitness:
    task_id: str
    protocol_sha256: str
    controller_sha256: str
    dimension: int
    # (实例身份, solve seed, reference gap百分比)。失败保留任务，不生成部分平均。
    members: tuple[tuple[str, int, float], ...] = ()
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


def score_panel(
    task: SolveTask | BaselineTask | FactorialTask,
    protocol: WorkerProtocol,
    outcome: dict,
    labels: dict[str, Label],
) -> PanelFitness:
    identity = task.task_id(protocol)
    problems = {p.instance_id: p for p in task.problems}
    if set(labels) != set(problems):
        raise ValueError("外部标签必须与任务问题精确对应")
    for name, label in labels.items():
        if type(label) is not Label or not math.isfinite(label.cost) or label.cost <= 0:
            raise ValueError("外部标签成本必须正且有限")
        actual = tour_cost(problems[name], label.tour)
        if abs(actual - label.cost) > 1e-8 + 1e-12 * actual:
            raise ValueError("外部标签路线与成本不符")

    def failed(message: str) -> PanelFitness:
        return PanelFitness(
            identity, protocol.sha256, task.controller_sha256, task.dimension, error=message
        )

    if (
        outcome.get("task_id") != identity
        or outcome.get("protocol_sha256") != protocol.sha256
        or any(outcome.get(k) != v for k, v in task.controller_identity().items())
        or outcome.get("dimension") != task.dimension
        or outcome.get("occurrence_id") != task.occurrence_id
    ):
        return failed("返回任务/程序/硬件/规模身份不符")
    if outcome.get("status") != "completed":
        return failed(str(outcome.get("error", "worker未完成预定任务")))
    try:
        result = outcome["native_result"]
        if type(task) is BaselineTask and result.get("baseline_policy") != task.policy.to_dict():
            return failed("原生实际基线配置与预定任务不符")
        if (
            type(task) is FactorialTask
            and result.get("factorial_policy") != task.factorial_policy.to_dict()
        ):
            return failed("原生实际析因配置与预定任务不符")
        if (
            result["budget_seconds"] != task.budget_seconds
            or result["preparation_mode"] != task.preparation_mode
            or len(result["items"]) != len(task.replicas)
        ):
            return failed("返回预算/模式/固定面板大小不符")
        for name in ("actual_seconds", "elapsed_seconds", "charged_seconds", "overrun_seconds"):
            if not math.isfinite(result[name]) or result[name] < 0:
                return failed("无效计时记录")
        if not math.isclose(
            result["elapsed_seconds"],
            result["actual_seconds"] + result["charged_seconds"],
            abs_tol=1e-12,
        ):
            return failed("预算账目不平")
        counted = task.evaluation_limit_per_colony is not None
        expected_overrun = 0 if counted else max(0, result["elapsed_seconds"] - task.budget_seconds)
        if not math.isclose(
            result["overrun_seconds"],
            expected_overrun,
            abs_tol=1e-12,
        ) or (task.preparation_mode == "end_to_end" and result["charged_seconds"] != 0):
            return failed("超限或端到端扣费账目不符")
        if task.preparation_charges is not None:
            expected_charge = math.fsum(
                cheap + preparation for _, cheap, preparation in task.preparation_charges
            )
            if result["charged_seconds"] > expected_charge + 1e-12 or (
                result["preparation_completed"]
                and not math.isclose(result["charged_seconds"], expected_charge, abs_tol=1e-12)
            ):
                return failed("实际扣费与任务冻结费用不符")
        counts = [
            result[name] for name in ("launched_batches", "completed_batches", "discarded_batches")
        ]
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or counts[0] != counts[1] + counts[2]
            or counts[2] > 1
        ):
            return failed("批次账目不平")
        if counted:
            limit = task.evaluation_limit_per_colony
            for key, expected in (
                ("evaluation_limit_per_colony", limit),
                ("completed_tour_evaluations_per_colony", limit),
                ("total_tour_evaluations", limit * protocol.colonies),
            ):
                if type(result.get(key)) is not int or result[key] != expected:
                    return failed("返回FE计数未完整覆盖预定限额")
            if (
                result.get("budget_kind") != "search_tour_evaluations"
                or counts != [limit // protocol.settings.ants, limit // protocol.settings.ants, 0]
                or result["charged_seconds"] != 0
                or result["preparation_completed"] is not True
            ):
                return failed("次数任务出现时间扣费、丢弃、未准备或未完成批次")
        members = []
        for (name, seed), item in zip(task.replicas, result["items"], strict=True):
            if item["has_incumbent"] is not True:
                return failed(f"预定实例/seed缺少截止前可行解: {name}/{seed}")
            completed, reported = item["completed_seconds"], item["cost"]
            if (
                not math.isfinite(completed)
                or not 0
                <= completed
                <= (result["actual_seconds"] if counted else task.budget_seconds)
                or not math.isfinite(reported)
                or reported <= 0
            ):
                return failed("返回迟到或非有限/非正incumbent")
            actual = tour_cost(problems[name], item["tour"])
            if abs(actual - reported) > 1e-8 + 1e-12 * actual:
                return failed("返回路线与成本不符")
            gap = 100 * (actual / labels[name].cost - 1)
            # 参考标签未必有独立最优性证明；更好的合法路线可以具有负reference gap。
            if not math.isfinite(gap):
                return failed("外部gap超出有限范围")
            members.append((name, seed, gap))
        return PanelFitness(
            identity, protocol.sha256, task.controller_sha256, task.dimension, tuple(members)
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return failed(f"返回结构或独立核验失败: {error}")


def aggregate_panels(
    tasks: tuple[SolveTask | BaselineTask | FactorialTask, ...],
    protocol: WorkerProtocol,
    outcomes: tuple[PanelFitness, ...],
    dimensions: tuple[int, ...],
) -> float:
    """按预定任务归集，先平均实例内seed，再平均规模内实例，最后等权平均规模。"""
    expected = {task.task_id(protocol): task for task in tasks}
    if not tasks or len(expected) != len(tasks):
        raise ValueError("预定任务身份为空或重复")
    if len(dimensions) != len(set(dimensions)) or set(dimensions) != {t.dimension for t in tasks}:
        raise ValueError("预定规模与评价面板不符")
    if len({task.controller_sha256 for task in tasks}) != 1:
        raise ValueError("不能把不同控制器的规模结果拼成一个fitness")
    received = {result.task_id: result for result in outcomes}
    if len(received) != len(outcomes) or set(received) != set(expected):
        raise ValueError("实际结果必须精确覆盖预定任务，不能遗漏、重复或加入外来任务")
    all_pairs = set()
    by_dimension: dict[int, dict[str, list[float]]] = {n: {} for n in dimensions}
    any_failure = False
    for identity, task in expected.items():
        result = received[identity]
        if (
            result.protocol_sha256 != protocol.sha256
            or result.controller_sha256 != task.controller_sha256
            or result.dimension != task.dimension
        ):
            raise ValueError("任务结果混用了硬件协议、程序或规模")
        for name, seed in task.replicas:
            key = task.dimension, name, seed
            if key in all_pairs:
                raise ValueError("跨面板重复了同一实例/seed，不能增加其统计权重")
            all_pairs.add(key)
        if result.failed:
            any_failure = True
            continue
        pairs = {(name, seed) for name, seed, _ in result.members}
        if len(result.members) != len(task.replicas) or pairs != set(task.replicas):
            raise ValueError("成功面板的成员遗漏或重复")
        for name, _, value in result.members:
            if not math.isfinite(value):
                raise ValueError("成功面板含非有限fitness")
            by_dimension[task.dimension].setdefault(name, []).append(value)
    if any_failure:
        return math.inf
    scale_means = [
        statistics.mean(statistics.mean(values) for values in by_dimension[n].values())
        for n in dimensions
    ]
    return statistics.mean(scale_means)
