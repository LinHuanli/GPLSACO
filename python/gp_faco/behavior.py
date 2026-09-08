"""逐批行为的独立账目核验与预登记4批窗口；不参与在线控制或fitness数值。"""

import math

WINDOW_BATCHES = 4
COUNTERS = (
    "construction_mne",
    "construction_steps",
    "construction_relocations",
    "construction_exhausted_ants",
    "construction_new_edges",
    "final_new_edges",
    "exact_returns",
    "fingerprint_returns",
    "ls_move_evaluations",
    "ls_accepted_moves",
    "ls_limit_reached_ants",
)
COSTS = (
    "global_before",
    "reference_before",
    "reference_used",
    "global_after",
    "iteration_best_cost",
)
FEEDBACK = {"return_rate", "ls_work", "stagnant_batches", "epoch_batches", "restarts"}
ROW_KEYS = {
    "batch",
    "colony",
    "ants",
    "dimension",
    "alternative",
    "legal_mask",
    "action_mask",
    "action",
    "baseline_requested_action",
    "feedback_before",
    "feedback_after",
    *COUNTERS,
    *COSTS,
}


def _require(condition, message):
    if not condition:
        raise ValueError(f"行为记录: {message}")


def _integer(value, upper=0xFFFFFFFFFFFFFFFF):
    return type(value) is int and 0 <= value <= upper


def _feedback(value):
    _require(type(value) is dict and set(value) == FEEDBACK, "反馈字段不完整")
    for key in ("stagnant_batches", "epoch_batches", "restarts"):
        _require(_integer(value[key]), "反馈计数非法")
    for key in ("return_rate", "ls_work"):
        _require(
            type(value[key]) is float and math.isfinite(value[key]) and 0 <= value[key] <= 1,
            "反馈EMA非法",
        )


def _decision(row, policy, experiment_mask):
    legal = experiment_mask & (0xFFFFFFFF if row["alternative"] < 4 else 0xFFFF)
    _require(row["legal_mask"] == legal, "限制前机会mask错误")
    action, requested = row["action"], row["baseline_requested_action"]
    expected = legal
    if policy.variant == "M11":
        _require(requested is None, "M11没有固定规则请求")
    else:
        _require(_integer(requested, 31), "固定规则请求动作非法")
        base, before = policy.baseline_policy, row["feedback_before"]
        level = base.mne_level
        if base.kind == "rule" and base.stagnation_step:
            level = min(
                base.max_mne_level, level + before["stagnant_batches"] // base.stagnation_step
            )
        _require(requested % 16 == base.region * 4 + level, "固定局部规则与状态不符")
        if base.kind == "rule":
            restart = bool(
                base.restart_stagnation
                and before["stagnant_batches"] >= base.restart_stagnation
                and before["epoch_batches"] >= base.restart_cooldown
            )
        elif base.restart_mode == "periodic":
            restart = row["batch"] > 0 and row["batch"] % base.restart_period == 0
        elif base.restart_mode == "none":
            restart = False
        elif base.restart_probability in (0, 1):
            restart = bool(base.restart_probability)
        else:
            restart = requested >= 16  # 随机抽样本身由完整CUDA诊断与重放独立核验。
        _require((requested >= 16) == restart, "固定重启规则与状态不符")
        if policy.variant == "M00":
            selected = requested if legal & (1 << requested) else requested % 16
            _require(action == selected, "M00动作偏离基线")
        elif policy.variant == "M10":
            expected &= 0xFFFF0000 if restart and legal & 0xFFFF0000 else 0xFFFF
        elif policy.variant == "M01":
            expected &= (1 << (requested % 16)) | (1 << (16 + requested % 16))
    _require(row["action_mask"] == expected and expected & (1 << action), "限制后mask/动作错误")


def validate_behavior(
    result,
    *,
    dimension,
    colonies,
    ants,
    evaluation_limit,
    ls_evaluation_limit,
    policy,
    experiment_mask=0xFFFFFFFF,
):
    """拒绝缺行、重复、非法mask、断开的反馈/GB及与FE/最终结果不一致的行为数据。"""
    _require(
        _integer(evaluation_limit) and ants > 0 and evaluation_limit % ants == 0,
        "预定FE不是完整批次",
    )
    policy.validate_mask(experiment_mask)
    behavior = result["behavior"]
    _require(
        type(behavior) is dict
        and set(behavior) == {"behavior_spec_id", "device_bytes", "host_row_bytes", "rows"},
        "行为字段不完整",
    )
    _require(
        type(behavior["behavior_spec_id"]) is int and behavior["behavior_spec_id"] == 1,
        "未知行为规格",
    )
    _require(
        _integer(behavior["device_bytes"])
        and behavior["device_bytes"] > 0
        and _integer(behavior["host_row_bytes"]),
        "观测内存记录非法",
    )
    rows = behavior["rows"]
    batches = evaluation_limit // ants
    _require(type(rows) is list and len(rows) == batches * colonies, "行数未完整覆盖FE")
    _require(
        result["completed_batches"] == result["launched_batches"] == batches
        and result["discarded_batches"] == 0
        and result["total_tour_evaluations"] == evaluation_limit * colonies,
        "行为与完整FE账目不符",
    )
    previous = [None] * colonies
    for index, row in enumerate(rows):
        _require(type(row) is dict and set(row) == ROW_KEYS, "逐批字段不完整")
        for name, expected in (
            ("batch", index // colonies),
            ("colony", index % colonies),
            ("ants", ants),
            ("dimension", dimension),
        ):
            _require(_integer(row[name]) and row[name] == expected, "行身份/顺序不符")
        for name, maximum in (
            ("alternative", 4),
            ("action", 31),
            ("legal_mask", 0xFFFFFFFF),
            ("action_mask", 0xFFFFFFFF),
        ):
            _require(_integer(row[name], maximum), "动作元数据非法")
        for name in COUNTERS:
            _require(_integer(row[name]), "原始工作计数非法")
        for name in COSTS:
            _require(
                type(row[name]) is float and math.isfinite(row[name]) and row[name] > 0,
                "成本必须正且有限",
            )
        _require(
            row["global_after"] <= row["global_before"]
            and row["global_after"] <= row["iteration_best_cost"]
            and row["global_before"] <= min(row["reference_before"], row["reference_used"]),
            "GB/reference成本关系错误",
        )
        _require(
            row["global_after"] == min(row["global_before"], row["iteration_best_cost"]),
            "GB更新不符",
        )
        if row["action"] < 16:
            _require(row["reference_used"] == row["reference_before"], "保持动作却更换reference")
        for name in ("construction_exhausted_ants", "ls_limit_reached_ants", "fingerprint_returns"):
            _require(row[name] <= ants, "蚂蚁计数超限")
        _require(
            row["exact_returns"] <= row["fingerprint_returns"]
            and 2 * (ants - row["exact_returns"])
            <= row["final_new_edges"]
            <= dimension * (ants - row["exact_returns"]),
            "精确返回与新边计数不符",
        )
        _require(
            row["construction_new_edges"] <= ants * dimension
            and row["construction_mne"] <= ants * (2 << (row["action"] % 4))
            and row["construction_relocations"]
            <= row["construction_steps"]
            <= ants * (dimension - 1)
            and row["ls_move_evaluations"] <= ants * ls_evaluation_limit
            and row["ls_accepted_moves"] <= ants * dimension,
            "内部工作计数超限",
        )
        before, after = row["feedback_before"], row["feedback_after"]
        _feedback(before)
        _feedback(after)
        prior = previous[row["colony"]]
        if prior is None:
            _require(all(value == 0 for value in before.values()), "初始反馈不为零")
        else:
            _require(
                before == prior["feedback_after"] and row["global_before"] == prior["global_after"],
                "逐批状态链断开",
            )
        _decision(row, policy, experiment_mask)
        restart = row["action"] >= 16
        _require(
            after["restarts"] == before["restarts"] + restart
            and after["epoch_batches"] == (0 if restart else before["epoch_batches"]) + 1
            and after["stagnant_batches"]
            == (
                0 if row["global_after"] < row["global_before"] else before["stagnant_batches"] + 1
            ),
            "批后反馈计数不符",
        )
        for key, fraction in (
            ("return_rate", row["fingerprint_returns"] / ants),
            (
                "ls_work",
                row["ls_move_evaluations"] / (ants * ls_evaluation_limit)
                if ls_evaluation_limit
                else 0,
            ),
        ):
            start = 0 if restart else before[key]
            _require(
                math.isclose(after[key], start + (fraction - start) / 16, rel_tol=0, abs_tol=1e-7),
                "批后EMA不符",
            )
        previous[row["colony"]] = row
    _require(
        sum(row["construction_steps"] for row in rows) == result["completed_construction_steps"]
        and sum(row["ls_move_evaluations"] for row in rows) == result["completed_ls_evaluations"],
        "累计工作与native结果不符",
    )
    for colony, row in enumerate(previous):
        if row is not None:
            final = result["control_states"][colony]
            _require(
                all(final[key] == value for key, value in row["feedback_after"].items()),
                "最后一行反馈与最终状态不符",
            )
            _require(
                math.isclose(
                    row["global_after"],
                    result["items"][colony]["cost"],
                    rel_tol=1e-12,
                    abs_tol=1e-8,
                ),
                "最后一行GB与最终解不符",
            )
    return rows


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def window_metrics(rows):
    """保留分子/分母；空窗口不会被解释为零返回、零重启或零改进。"""
    fe = sum(row["ants"] for row in rows)
    totals = {name: sum(row[name] for row in rows) for name in COUNTERS}
    restarts = sum(row["action"] >= 16 for row in rows)
    legal = sum(bool(row["legal_mask"] & 0xFFFF0000) for row in rows)
    allowed = sum(bool(row["action_mask"] & 0xFFFF0000) for row in rows)
    histogram = [sum(row["action"] % 4 == level for row in rows) for level in range(4)]
    return {
        "batches": len(rows),
        "fe": fe,
        "restarts": restarts,
        "legal_restart_opportunities": legal,
        "allowed_restart_opportunities": allowed,
        "no_alternative_batches": sum(row["alternative"] == 4 for row in rows),
        "legal_but_factor_mask_blocks": sum(
            bool(row["legal_mask"] & 0xFFFF0000) and not row["action_mask"] & 0xFFFF0000
            for row in rows
        ),
        "legal_but_kept": sum(
            bool(row["legal_mask"] & 0xFFFF0000) and row["action"] < 16 for row in rows
        ),
        "restart_per_fe": _ratio(restarts, fe),
        "restart_given_legal": _ratio(restarts, legal),
        "mne_levels": [2, 4, 8, 16],
        "mne_action_counts": histogram,
        "mne_action_frequencies": [_ratio(count, len(rows)) for count in histogram],
        "raw_totals": totals,
        "per_ant": {name: _ratio(value, fe) for name, value in totals.items()},
        "relative_gb_improvement": 1 - rows[-1]["global_after"] / rows[0]["global_before"]
        if rows
        else None,
    }


def summarize_behavior(result, **contract):
    rows = validate_behavior(result, **contract)
    summaries = []
    for colony in range(contract["colonies"]):
        solve = rows[colony :: contract["colonies"]]
        events = []
        for r, row in enumerate(solve):
            if row["action"] < 16:
                continue
            before, after = solve[max(0, r - WINDOW_BATCHES) : r], solve[r : r + WINDOW_BATCHES]
            events.append(
                {
                    "batch": r,
                    "complete": len(before) == len(after) == WINDOW_BATCHES,
                    "before": window_metrics(before),
                    "after": window_metrics(after),
                    "additional_restarts_after": sum(v["action"] >= 16 for v in after) - 1,
                }
            )
        complete = [event for event in events if event["complete"]]
        paired = None
        if complete:
            paired = {
                "mean_after_minus_before": {
                    name: math.fsum(
                        event["after"]["per_ant"][name] - event["before"]["per_ant"][name]
                        for event in complete
                    )
                    / len(complete)
                    for name in COUNTERS
                },
                "mean_post_relative_gb_improvement": math.fsum(
                    event["after"]["relative_gb_improvement"] for event in complete
                )
                / len(complete),
            }
        summaries.append(
            {
                "colony": colony,
                "whole_solve": window_metrics(solve),
                "restart_events": events,
                "complete_events": len(complete),
                "boundary_events": len(events) - len(complete),
                "paired_windows": paired,
            }
        )
    return {"behavior_spec_id": 1, "window_batches": WINDOW_BATCHES, "solves": summaries}
