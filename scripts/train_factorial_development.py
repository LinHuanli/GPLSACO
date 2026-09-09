#!/usr/bin/env python3
"""真实M10/M01工程训练、固定验证与独立审计；正式训练另行冻结。"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.configuration_search import gpu_boundary  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evaluation_run import portable_outcome  # noqa: E402
from gp_faco.evolution import EvolutionSettings  # noqa: E402
from gp_faco.factorial_policy import FactorialPolicy  # noqa: E402
from gp_faco.factorial_training import FactorialTrainingRun  # noqa: E402
from gp_faco.training import TrainingData, TrainingSettings, json_value  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, content_hash, file_hash  # noqa: E402
from summarize_training import summarize  # noqa: E402


class ObservedFactorialRun(FactorialTrainingRun):
    """提交前等待空闲；先保存每个实际返回，随后记录资源查询，不能因争用重做求解。"""

    def _boundary(self):
        try:
            return gpu_boundary(self.protocol, self.state["active_worker"]["pid"])
        except Exception as error:
            return {"foreign_processes": None, "error": f"{type(error).__name__}: {error}"}

    def _before_submit(self):
        while True:
            observation = self._boundary()
            key = self.state["pending"]["key"]
            if (
                type(observation["foreign_processes"]) is int
                and observation["foreign_processes"] == 0
            ):
                self.state.setdefault("admission", []).append({"task_key": key, **observation})
                self._save()
                return
            with (self.directory / "resource-waits.jsonl").open("a") as out:
                out.write(json.dumps({"task_key": key, **observation}) + "\n")
            print(json.dumps({"event": "waiting_for_idle_gpu", **observation}), flush=True)
            time.sleep(5)  # 观察间隔，不是ACO或进化时间上限。

    def _operation(self, kind, description, submit, validate):
        def observed_submit(worker):
            original = submit(worker)
            key = self.state["pending"]["key"]

            class ObservedFuture:
                def result(inner, timeout=None):
                    outcome = original.result(timeout=timeout)
                    self._immutable_record(
                        self.directory / "raw-returns" / f"{key}.json",
                        json_value(portable_outcome(outcome)),
                    )
                    outcome["gpu_boundary_after"] = self._boundary()
                    return outcome

            return ObservedFuture()

        return super()._operation(kind, description, observed_submit, validate)


def run(args, config, policy):
    row = next(
        csv.reader(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={args.gpu_uuid}",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).splitlines()
        )
    )
    settings = TrainingSettings(
        **config["training"],
        evolution=EvolutionSettings(**config["evolution"]),
        scope="engineering_development",
    )
    native_dir = args.native_dir.resolve()
    if not native_dir.is_relative_to(PROJECT):
        raise ValueError("析因原生构建必须位于本工作树")
    protocol = WorkerProtocol(
        args.gpu_uuid,
        row[0].strip(),
        row[1].strip(),
        file_hash(native_dir / "gp_faco_ext.so"),
        colonies=config["colonies"],
        settings=SolverSettings(**config["solver"]),
        extension_directory=str(native_dir.relative_to(PROJECT)),
        maximum_registered_per_dimension=config["maximum_registered_per_dimension"],
    )
    with IndexedDataset(args.database, args.dataset_root) as source:
        training, validation = {}, {}
        for n, _ in settings.budgets:
            ids = sorted(source.record_ids("development", n))
            a, b = (
                config["development_training_per_scale"],
                config["development_validation_per_scale"],
            )
            if len(ids) < a + b:
                raise ValueError("开发池不足，不能借用正式TEST")
            training[n], validation[n] = ids[:a], ids[a : a + b]
        data = TrainingData(
            source,
            training,
            validation,
            {
                "database_sha256": file_hash(args.database),
                "split_sha256": file_hash(PROJECT / "provenance/splits.v1.json"),
                "config_sha256": file_hash(args.config),
                "entrypoint_sha256": file_hash(Path(__file__)),
                "selection": "sorted development IDs; disjoint leading training then validation",
                "reference_status": "user_supplied_not_independently_certified",
            },
        )
        training_run = ObservedFactorialRun(
            args.output,
            settings,
            protocol,
            data,
            factorial_policy=policy,
            resume=args.resume,
            event=lambda record: print(json.dumps(record, ensure_ascii=False), flush=True),
        )
        result = training_run.run(stop_after_tasks=args.stop_after_tasks)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"]}), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


def audit(args, config, policy):
    report = summarize(
        args.output,
        database=args.database,
        dataset_root=args.dataset_root,
        entrypoint=Path(__file__),
        checks=args.checks,
    )
    if report["factorial_policy"] != policy.to_dict():
        raise ValueError("审计变体与训练身份不符")
    selection = load_checkpoint(args.output / "data_selection.json")
    if selection["identity"]["config_sha256"] != file_hash(args.config):
        raise ValueError("工程配置改变")
    if (
        report["training_individuals"]
        != config["evolution"]["population"] * config["evolution"]["generations"]
        or report["work_by_phase"]["training"]["solve_jobs"] != 48
    ):
        raise ValueError("没有完成全部工程训练个体和两规模")
    state = load_checkpoint(args.output / "checkpoint.json")
    raws = {p.stem: p for p in (args.output / "raw-returns").glob("*.json")}
    if set(raws) != set(state["completed"]):
        raise ValueError("原始返回与已完成任务不一一对应")
    deviations = []
    for key, path in raws.items():
        record = load_checkpoint(args.output / "tasks" / f"{key}.json")
        raw = load_checkpoint(path)
        outcome = {k: v for k, v in record["outcome"].items() if k != "gpu_boundary_after"}
        if raw != outcome:
            raise ValueError("实际原始返回与入账记录不符")
        post = record["outcome"]["gpu_boundary_after"]
        if type(post["foreign_processes"]) is not int or post["foreign_processes"] != 0:
            deviations.append({"task_key": key, **post})
    admitted = {
        r["task_key"]
        for r in state["admission"]
        if type(r["foreign_processes"]) is int and r["foreign_processes"] == 0
    }
    if admitted != set(state["completed"]) or len(state["admission"]) != len(admitted):
        raise ValueError("提交许可有遗漏或重复")
    resources = [json.loads(p.read_text()) for p in args.resources]
    if len(resources) != 2 or any(r["exit_code"] != 0 for r in resources):
        raise ValueError("需要原始暂停和完整恢复两次CLI成功回执")
    report.update(
        scope="E2 engineering only; M00 remains unselected; no formal TEST",
        original_raw_returns=len(raws),
        resource_deviations=deviations,
        timing_class="observed_exclusive"
        if not deviations
        else "exclude aggregate clean-speed comparison",
        cli_resources=resources,
        raw_manifest_sha256=content_hash({k: file_hash(v) for k, v in sorted(raws.items())}),
    )
    atomic_json(args.output / "audit.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "solve_jobs",
                    "valid_members",
                    "search_tour_evaluations",
                    "validation_candidates",
                    "max_independent_cost_error",
                )
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "audit"))
    parser.add_argument("--variant", choices=("M10", "M01"), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/e2_engineering_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, default=PROJECT / "build/e2")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int, choices=(5,))
    parser.add_argument("--checks", type=Path)
    parser.add_argument("--resources", type=Path, nargs="*")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(PROJECT):
        parser.error("产物必须位于本GPLSACO工作树")
    config = json.loads(args.config.read_text())
    if config["scope"] != "engineering_development" or config["wall_clock_limit"] is not None:
        parser.error("此入口仅执行次数工程验收，不能代替正式E2")
    policy = FactorialPolicy.from_dict(
        {
            "factorial_spec_id": 1,
            "variant": args.variant,
            "baseline_policy": config["baseline_policy"],
        }
    )
    if args.stage == "run":
        if not args.gpu_uuid:
            parser.error("run需要gpu UUID")
        run(args, config, policy)
    else:
        if args.checks is None or args.resources is None:
            parser.error("audit需要原生检查及两次CLI原始资源回执")
        audit(args, config, policy)


if __name__ == "__main__":
    main()
