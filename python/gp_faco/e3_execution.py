"""正式E3执行组装；固定图/面板/资源边界，复用既有评价与恢复事务。"""

from __future__ import annotations

import json

from gp_faco.checkpoint import atomic_json
from gp_faco.configuration_search import ConfigurationRun, SearchData, SearchSettings, gpu_boundary
from gp_faco.e3_protocol import CONDITIONS, require, static_policies, training_settings
from gp_faco.evaluation_run import json_value, portable_outcome
from gp_faco.training import TrainingData, TrainingRun
from gp_faco.worker import PROJECT, SolverSettings, WorkerProtocol, file_hash


class ObservedGraphOperations:
    """真实返回先独立写入，再观察GPU；查询失败不允许再次求解同一任务。"""

    def _before_submit(self):
        observation = self._boundary(self.protocol, self.state["active_worker"]["pid"])
        self.state.setdefault("admission", []).append(
            {"task_key": self.state["pending"]["key"], **observation}
        )
        self._save()
        require(
            type(observation["foreign_processes"]) is int and observation["foreign_processes"] == 0,
            "目标GPU占用未知或存在外来进程，尚未提交当前任务",
        )

    def _operation(self, kind, description, submit, validate):
        def observed_submit(worker):
            original = submit(worker)

            class ObservedFuture:
                def result(inner, timeout=None):
                    # TimeoutError只能来自同一个Future；不附加算法截止，也不新建任务。
                    outcome = portable_outcome(original.result(timeout=timeout))
                    path = self.directory / "raw_returns" / f"{self.state['pending']['key']}.json"
                    if path.exists():
                        require(
                            json.loads(path.read_text()) == json_value(outcome),
                            "同一任务已有不同真实返回，不能覆写或择优",
                        )
                    else:
                        atomic_json(path, outcome)
                    try:
                        observation = self._boundary(
                            self.protocol, self.state["active_worker"]["pid"]
                        )
                    except Exception as error:
                        observation = {
                            "foreign_processes": None,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    # 不修改刚保存的原始字典，使重复读取Future也不会改变原始返回身份。
                    return {**outcome, "gpu_boundary_after": observation}

            return ObservedFuture()

        def checked(outcome):
            if "gpu_boundary_after" in outcome:
                require(
                    type(outcome["gpu_boundary_after"].get("foreign_processes")) is int
                    and outcome["gpu_boundary_after"]["foreign_processes"] == 0,
                    "任务结束时GPU占用未知或有外来进程，原始结果已保留",
                )
            else:
                # 基础设施重试耗尽是显式失败位置；没有实际返回时不能伪造边界或原生费用。
                require(
                    outcome.get("status") == "failed"
                    and outcome.get("error") == "基础设施重试耗尽；保留完整任务失败",
                    "真实返回缺少GPU边界观察",
                )
            return validate(outcome)

        return super()._operation(kind, description, observed_submit, checked)


class RegisteredGraphTrainingRun(ObservedGraphOperations, TrainingRun):
    def __init__(self, *args, expected_panels, boundary=gpu_boundary, **kwargs):
        self.expected_panels, self._boundary = expected_panels, boundary
        super().__init__(*args, **kwargs)

    def _begin_generation(self):
        super()._begin_generation()
        require(
            self.state["panels"] == self.expected_panels[self.evolution.generation],
            "实际代面板与预登记随机流不符，尚未提交评价",
        )


class RegisteredGraphStaticRun(ObservedGraphOperations, ConfigurationRun):
    def _submit(self, worker, task):
        # 统一由ObservedGraphOperations记录边界，不能让旧Static包装器先修改真实返回。
        return worker.submit(task)


def condition_spec(name):
    require(name in {row[0] for row in CONDITIONS}, "E3条件未登记")
    _, kind, mode = next(row for row in CONDITIONS if row[0] == name)
    return kind, mode


def worker_protocol(plan, execution, condition, gpu_uuid, model, driver, host, *, static=False):
    config = plan["config"]
    kind, mode = condition_spec(condition)
    require(
        all(type(value) is str and value.strip() for value in (gpu_uuid, model, driver, host)),
        "每个正式运行都须记录实际GPU及host身份",
    )
    return WorkerProtocol(
        gpu_uuid,
        model,
        driver,
        plan["native_binary_sha256"],
        execution_host=host,
        dimensions=tuple(config["dimensions"]),
        colonies=config["colonies"],
        settings=SolverSettings(**config["solver"]),
        maximum_registered_per_dimension=(
            config["static_search"]["maximum_registered_per_dimension"]
            if static
            else config["maximum_registered_per_dimension"]
        ),
        constraint_mode=mode,
        graph_prior_kind=kind,
        graph_catalog_path=execution["catalog_path"],
        graph_catalog_sha256=execution["catalog_sha256"],
    )


def training_identity(plan, execution, condition):
    condition_spec(condition)
    return {
        "database_sha256": plan["database_sha256"],
        "split_sha256": plan["split_sha256"],
        "config_sha256": plan["config_sha256"],
        "entrypoint_sha256": execution["entrypoint_sha256"],
        "e3_plan_sha256": plan["sha256"],
        "e3_execution_sha256": execution["sha256"],
        "catalog_sha256": execution["catalog_sha256"],
        "condition": condition,
        "selection": "complete frozen train and validation splits; no development or test members",
        "reference_status": "user_supplied_not_independently_certified",
    }


def make_training(source, plan, execution, members, condition, seed):
    config = plan["config"]
    settings = training_settings(config, seed)
    data = TrainingData(
        source,
        {int(n): value["train"] for n, value in members.items()},
        {int(n): value["validation"] for n, value in members.items()},
        training_identity(plan, execution, condition),
    )
    return settings, data


def static_config(plan, condition):
    condition_spec(condition)
    config = plan["config"]
    return {
        "scope": "E3_development_static_tuning",
        "condition": condition,
        "e3_plan_sha256": plan["sha256"],
        "search": config["static_search"]["settings"],
        "data_pools": config["static_search"]["data_pools"],
        "solver": config["solver"],
        "dimensions": config["dimensions"],
        "colonies": config["colonies"],
        "maximum_registered_per_dimension": config["static_search"][
            "maximum_registered_per_dimension"
        ],
        "policies": [p.to_dict() for p in static_policies(config)],
    }


def make_static(source, plan, execution, development, condition):
    name = execution["static_configs"][condition]["path"]
    path = PROJECT / name
    require(
        file_hash(path) == execution["static_configs"][condition]["sha256"], "Static冻结配置改变"
    )
    config = json.loads(path.read_text())
    require(config == static_config(plan, condition), "Static配置不是完整预登记族")
    pools = {
        role: {
            int(n): ids[selection["offset"] : selection["offset"] + selection["count"]]
            for n, ids in development.items()
        }
        for role, selection in config["data_pools"].items()
    }
    identity = {
        "database_sha256": plan["database_sha256"],
        "split_sha256": plan["split_sha256"],
        "config_sha256": file_hash(path),
        "config_path": name,
        "entrypoint_sha256": execution["entrypoint_sha256"],
        "e3_plan_sha256": plan["sha256"],
        "e3_execution_sha256": execution["sha256"],
        "catalog_sha256": execution["catalog_sha256"],
        "condition": condition,
        "selection": "explicit disjoint ranges of sorted development IDs",
        "reference_status": "user_supplied_not_independently_certified",
        "scope": config["scope"],
    }
    return (
        SearchSettings(**config["search"]),
        static_policies(plan["config"]),
        SearchData(source, pools, identity),
    )
