#!/usr/bin/env python3
"""量化 DEAP→IR→Python/C++/CUDA 一致性；不是完整求解器性能实验。"""

from __future__ import annotations

import argparse
import json
import random
import socket
from pathlib import Path

import gp_faco_ext
import numpy as np
from deap import gp
from gp_faco.primitives import make_primitive_set
from gp_faco.program_ir import choose_actions, evaluate, export_tree

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--programs", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if not output_path.is_relative_to(PROJECT) or args.programs < 1:
        parser.error("输出须在项目内，程序数量须为正")
    backends = ["score_cpu"] + (["score_cuda"] if args.cuda else [])
    for backend in backends:
        if not hasattr(gp_faco_ext, backend):
            raise RuntimeError(f"请求的后端未构建: {backend}")
    random.seed(20260908)
    rng = np.random.default_rng(53459699)
    features = rng.random((12, 7, 32), dtype=np.float32)
    features[:, 0, :] = 0
    features[:, 1, :] = 1
    masks = rng.integers(1, 0xFFFFFFFF, size=7, dtype=np.uint32)
    masks[:3] = [1, 1 << 31, 0xFFFFFFFF]
    cases = []
    for no_feedback in (False, True):
        pset = make_primitive_set(no_feedback)
        for _ in range(args.programs // 2):
            tree = gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=0, max_=5))
            cases.append(export_tree(tree))
    pset = make_primitive_set()
    for expression in (
        "0.0",
        "restart",
        "SUB(return_rate,restart)",
        "AQ(ls_work,mne_level)",
        "SUB(MUL(MUL(2.0,2.0),MUL(2.0,2.0)),2.0)",
        "ABS(ABS(ABS(ABS(ABS(-1.0)))))",
    ):
        cases.append(export_tree(gp.PrimitiveTree.from_string(expression, pset)))
    # 完整二叉深度5树覆盖63指令与峰值6槽栈。
    expression = "restart"
    for _ in range(5):
        expression = f"ADD({expression},{expression})"
    cases.append(export_tree(gp.PrimitiveTree.from_string(expression, pset)))
    reports = {}
    for backend in backends:
        maximum_error, mismatches = 0.0, 0
        for program in cases:
            expected = evaluate(program, features)
            result = getattr(gp_faco_ext, backend)(program.to_dict(), features, masks)
            error = float(np.max(np.abs(result["scores"] - expected)))
            maximum_error = max(maximum_error, error)
            np.testing.assert_allclose(result["scores"], expected, atol=2e-6, rtol=2e-6)
            mismatches += int(
                np.count_nonzero(result["actions"] != choose_actions(expected, masks))
            )
        if mismatches:
            raise RuntimeError(f"{backend}: 动作选择不一致 {mismatches}")
        reports[backend] = {
            "programs": len(cases),
            "colonies_per_program": 7,
            "scores_checked": len(cases) * 7 * 32,
            "max_abs_error": maximum_error,
            "action_mismatches": mismatches,
        }
    report = {
        "scope": "program scoring diagnostics; full FACO Engine not yet implemented",
        "hostname": socket.gethostname(),
        "seed": 20260908,
        "numeric_spec_id": 1,
        "atol": 2e-6,
        "rtol": 2e-6,
        "backends": reports,
    }
    if args.cuda:
        report["device"] = gp_faco_ext.cuda_device_info()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
