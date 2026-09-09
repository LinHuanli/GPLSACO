"""离线标准代际训练、固定验证和可恢复任务日志；在线求解全部交给spawn worker。"""

from __future__ import annotations

import fcntl
import importlib.metadata
import math
import platform
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gp_faco.checkpoint import load_checkpoint, save_checkpoint
from gp_faco.evaluation_run import EvaluationRun, json_value
from gp_faco.evolution import Evolution, EvolutionSettings, _tuples, individual_from_program
from gp_faco.fitness import PanelFitness, aggregate_panels, score_panel
from gp_faco.program_ir import Program, export_tree
from gp_faco.worker import (
    PROJECT,
    PersistentGpuWorker,
    SolveTask,
    WorkerProtocol,
    freeze_problems,
    preparation_id,
)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class TrainingSettings:
    evolution: EvolutionSettings = field(default_factory=EvolutionSettings)
    evolution_seed: int = 1103
    panel_seed: int = 73001
    instances_per_panel: int = 16
    solver_seeds_per_instance: int = 2
    validation_seeds: tuple[int, ...] = (17, 29)
    budgets: tuple[tuple[int, float], ...] = ((500, 0.6), (1000, 1.2))
    preparation_mode: str = "cached_charged"
    experiment_mask: int = 0xFFFFFFFF
    infrastructure_retries: int = 1
    scope: str = "engineering_development"
    budget_kind: str = "wall_clock"
    validation_budgets: tuple[tuple[int, int], ...] = ()
    monitoring_every: int = 0
    population_batch_size: int = 1
    portable_a5000: bool = False

    def __post_init__(self):
        object.__setattr__(self, "validation_seeds", tuple(self.validation_seeds))
        object.__setattr__(self, "budgets", tuple(tuple(v) for v in self.budgets))
        object.__setattr__(
            self, "validation_budgets", tuple(tuple(v) for v in self.validation_budgets)
        )
        if self.validation_budgets and (
            {n for n, _ in self.validation_budgets} != {n for n, _ in self.budgets}
            or any(type(v) is not int or v < 0 for _, v in self.validation_budgets)
        ):
            raise ValueError("验证 FE 预算必须覆盖训练规模")
        if type(self.monitoring_every) is not int or self.monitoring_every < 0:
            raise ValueError("监控间隔必须非负整数")
        if type(self.evolution) is not EvolutionSettings:
            raise TypeError("训练需要显式演化配置")
        if self.budget_kind not in ("wall_clock", "search_tour_evaluations"):
            raise ValueError("未知训练预算单位")
        counted = self.budget_kind == "search_tour_evaluations"
        if (
            type(self.population_batch_size) is not int
            or not 1 <= self.population_batch_size <= 128
        ):
            raise ValueError("GPU 种群批次大小必须在 1..128")
        if self.population_batch_size > 1 and (not counted or self.preparation_mode != "cached"):
            raise ValueError("GPU 种群并行需要 cached 的评价次数协议")
        if self.evolution.feature_spec_id != (2 if counted else 1):
            raise ValueError("进化特征版本与预算单位不符")
        for value in (self.evolution_seed, self.panel_seed, *self.validation_seeds):
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError("训练、面板和验证seed必须为uint64")
        if any(
            type(v) is not int or v < 1
            for v in (self.instances_per_panel, self.solver_seeds_per_instance)
        ) or (
            len(self.validation_seeds) != self.solver_seeds_per_instance
            or len(set(self.validation_seeds)) != len(self.validation_seeds)
        ):
            raise ValueError("固定验证seed数量与训练形状不符")
        if (
            not self.budgets
            or any(
                len(v) != 2
                or type(v[0]) is not int
                or not 3 <= v[0] <= 10000
                or type(v[1]) not in (int, float)
                or not math.isfinite(v[1])
                or v[1] < 0
                or (counted and type(v[1]) is not int)
                for v in self.budgets
            )
            or len({v[0] for v in self.budgets}) != len(self.budgets)
        ):
            raise ValueError("规模预算必须非负有限且规模不得重复")
        if (
            self.preparation_mode
            not in (("cached", "end_to_end") if counted else ("cached_charged", "end_to_end"))
            or type(self.infrastructure_retries) is not int
            or not 0 <= self.infrastructure_retries <= 3
            or type(self.experiment_mask) is not int
            or not 0 < self.experiment_mask <= 0xFFFFFFFF
            or not self.experiment_mask & 0xFFFF
            or type(self.scope) is not str
            or not self.scope
        ):
            raise ValueError("准备模式、重试、动作mask或研究范围无效")


class TrainingData:
    """协调端的数据访问；身份记录与数据源分离，标签仅在外部评分时读取。"""

    def __init__(
        self,
        source,
        training: dict[int, list[str]],
        validation: dict[int, list[str]],
        identity: dict,
        *,
        shared_panels=None,
        monitoring_panels=None,
        baseline_cache=None,
    ):
        self.source = source
        self.training = {n: tuple(sorted(ids)) for n, ids in training.items()}
        self.validation = {n: tuple(sorted(ids)) for n, ids in validation.items()}
        self.identity = json_value(identity)
        self.shared_panels = json_value(shared_panels) if shared_panels is not None else None
        self.monitoring_panels = json_value(monitoring_panels or [])
        self.baseline_cache = baseline_cache
        if not identity or set(self.training) != set(self.validation):
            raise ValueError("数据身份或训练/验证规模不完整")
        all_ids = []
        for pool in (self.training, self.validation):
            for n, ids in pool.items():
                if type(n) is not int or not ids or any(type(v) is not str or not v for v in ids):
                    raise ValueError("数据池身份无效")
                all_ids.extend(ids)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("训练、验证或规模之间有重复实例身份")
        self._instances = {}
        if self.shared_panels is not None:
            for generation in self.shared_panels:
                for panel in generation["panels"]:
                    if not set(panel["ids"]).issubset(self.training[panel["dimension"]]):
                        raise ValueError("预先登记的训练面板不属于正式训练池")

    def problems(self, ids) -> tuple:
        for name in ids:
            if name not in self._instances:
                value = self.source.load_instance(name)
                if value.instance_id != name:
                    raise ValueError("数据源返回了其他实例身份")
                self._instances[name] = value
        return freeze_problems(tuple(self._instances[name] for name in ids))

    def labels(self, problems) -> dict:
        return {p.instance_id: self.source.load_label(p.instance_id) for p in problems}

    def manifest(self) -> dict:
        # 大训练池只保存一次完整成员表；每次checkpoint保留其摘要，避免反复重写数万ID。
        return json_value(
            {
                "identity": self.identity,
                **{
                    role: {n: {"count": len(ids)} for n, ids in pool.items()}
                    for role, pool in (("training", self.training), ("validation", self.validation))
                },
            }
        )


def training_manifest(settings: TrainingSettings, protocol: WorkerProtocol, data: TrainingData):
    if settings.population_batch_size > 1 and (
        protocol.constraint_mode != "unrestricted"
        or any(n > 1500 or protocol.settings.ants_for(n) % 4 for n in protocol.dimensions)
    ):
        raise ValueError("种群展开要求 n<=1500 的普通 FACO 和完整 warp 蚂蚁组")
    if protocol.graph_catalog is not None and settings.budget_kind != "search_tour_evaluations":
        raise ValueError("图约束训练必须按evaluation次数终止")
    n_values = tuple(n for n, _ in settings.budgets)
    if set(n_values) != set(protocol.dimensions) or set(n_values) != set(data.training):
        raise ValueError("数据、预算与worker规模不符")
    if settings.instances_per_panel * settings.solver_seeds_per_instance != protocol.colonies:
        raise ValueError("每个程序必须使用worker的完整固定形状")
    if settings.budget_kind == "search_tour_evaluations" and any(
        limit % protocol.settings.ants_for(n) or limit // protocol.settings.ants_for(n) > 0xFFFFFFFF
        for n, limit in (*settings.budgets, *settings.validation_budgets)
    ):
        raise ValueError("训练FE必须为完整ants批次，且批次数不超出uint32")
    for n in n_values:
        train_count, val_count = len(data.training[n]), len(data.validation[n])
        if train_count < settings.instances_per_panel or val_count % settings.instances_per_panel:
            raise ValueError("训练不足一个完整面板或验证不能整分为固定面板")
        needed = max(
            min(train_count, settings.evolution.generations * settings.instances_per_panel),
            val_count,
        )
        if needed > protocol.maximum_registered_per_dimension:
            raise ValueError("worker缓存不足覆盖整个阶段，禁止中途更换准备费用")
    worker_manifest = protocol.manifest()
    if settings.portable_a5000:
        # 物理设备是执行记录；同一构建与算法参数在其他 A5000 上仍是同一实验。
        for name in ("gpu_uuid", "execution_host", "driver_version"):
            worker_manifest.pop(name)
    return json_value(
        {
            "training_version": 2,
            "settings": asdict(settings),
            "worker_protocol": worker_manifest,
            "data": data.manifest(),
            "baseline_protocol": data.baseline_cache.manifest if data.baseline_cache else None,
            "software": {
                "python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in ("deap", "numpy")},
            },
            "fitness_policy": "all_individuals_full_common_panel_each_generation_v1",
            "selection_policy": "fitness_then_nodes_then_opcode_tuple_v2",
            "retry_policy": "same_identity_only_after_confirmed_broken_pool_or_dead_worker",
        }
    )


class TrainingPaused(Exception):
    """仅在已完成任务边界暂停；活动Future从不被取消。"""


class TrainingRun(EvaluationRun):
    def __init__(
        self,
        directory: Path,
        settings: TrainingSettings,
        protocol: WorkerProtocol,
        data: TrainingData,
        *,
        resume: bool = False,
        worker_factory=PersistentGpuWorker,
        event=None,
    ):
        self.directory = directory.resolve()
        if not self.directory.is_relative_to(PROJECT):
            raise ValueError("训练目录必须在GPLSACO内")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lease = (self.directory / ".run.lock").open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lease.close()
            raise RuntimeError("同一训练目录已有协调进程；不能并发恢复") from None
        self._worker = None
        self._closed = False
        self._event_callback = event or (lambda record: None)
        self._worker_factory = worker_factory
        self.settings, self.protocol, self.data = settings, protocol, data
        self._new_solves = 0
        self._stop_after = None
        self.path = self.directory / "checkpoint.json"
        try:
            manifest = self._training_manifest()
            self.run_id = self.directory.name
            if resume:
                self.state = load_checkpoint(self.path)
                if (
                    self.state.get("version") != 2
                    or type(self.state["version"]) is not int
                    or self.state.get("phase")
                    not in ("training", "validation", "complete", "failed")
                ):
                    raise ValueError("未知训练checkpoint版本或阶段")
                if self.state["manifest"] != manifest or self.state["run_id"] != self.run_id:
                    raise ValueError("恢复时软件、数据、配置或硬件参数发生变化")
                selection = load_checkpoint(self.directory / "data_selection.json")
                if selection != json_value(
                    {
                        "training": data.training,
                        "validation": data.validation,
                        "identity": data.identity,
                    }
                ):
                    raise ValueError("恢复时数据成员归档与当前身份不符")
                if load_checkpoint(self.directory / "manifest.json") != manifest:
                    raise ValueError("恢复时run manifest归档不符")
                self.evolution = Evolution.from_state_dict(self.state["evolution"])
                self.panel_rng = random.Random()
                self.panel_rng.setstate(_tuples(self.state["panel_rng"]))
                active = self.state["active_worker"]
                if active is not None:
                    # 同一host和目录租约均已核查；旧PID仍存在时不猜测其任务是否完成。
                    if self._previous_worker_active(active):
                        raise RuntimeError("旧worker仍可能活动，保留checkpoint，不能重新提交")
                    self.state["active_worker"] = None
                if self.state["pending"] and self.state["pending"]["attempts"]:
                    last = self.state["pending"]["attempts"][-1]
                    receipt = self.directory / "tasks" / f"{self.state['pending']['key']}.json"
                    if last["status"] == "running" and not receipt.exists():
                        last["status"] = "worker_terminated_unobserved"
                        self.state["costs"]["infrastructure_failures"] += 1
                        self.state["costs"]["unobserved_terminated_attempts"] += 1
                self._save()
            else:
                if any(p.name != ".run.lock" for p in self.directory.iterdir()):
                    raise FileExistsError("新训练要求空目录；已有训练须显式resume")
                self.evolution = Evolution(settings.evolution, settings.evolution_seed)
                self.evolution.initialize()
                self.panel_rng = random.Random(settings.panel_seed)
                self.state = {
                    "version": 2,
                    "run_id": self.run_id,
                    "manifest": manifest,
                    "phase": "training",
                    "active_worker": None,
                    "worker_history": [],
                    "pending": None,
                    "completed": {},
                    "fees": {},
                    "panels": [],
                    "training_panels": [],
                    "validation_panels": self._validation_panels(),
                    "shortlist": [],
                    "validation_results": {},
                    "generation_metrics": {},
                    "monitoring_results": {},
                    "selected": None,
                    "costs": {
                        "solve_jobs": 0,
                        "preparation_jobs": 0,
                        "failed_solves": 0,
                        "valid_members": 0,
                        "worker_seconds": 0.0,
                        "evaluator_seconds": 0.0,
                        "charged_seconds": 0.0,
                        "native_actual_seconds": 0.0,
                        "search_tour_evaluations": 0,
                        "overrun_seconds": 0.0,
                        "infrastructure_failures": 0,
                        "unobserved_terminated_attempts": 0,
                        "observer_timeouts": 0,
                    },
                }
                save_checkpoint(self.directory / "manifest.json", manifest)
                save_checkpoint(
                    self.directory / "data_selection.json",
                    {
                        "training": self.data.training,
                        "validation": self.data.validation,
                        "identity": self.data.identity,
                    },
                )
            self._open_journal()
            if data.shared_panels is not None:
                if len(data.shared_panels) != settings.evolution.generations:
                    raise ValueError("预先登记的面板代数与训练不一致")
                self._immutable_record(
                    self.directory / "shared_panels.json",
                    {"training": data.shared_panels, "monitor": data.monitoring_panels},
                )
            self._save(force=True)
        except BaseException:
            self._lease.close()
            raise

    def _training_manifest(self):
        return training_manifest(self.settings, self.protocol, self.data)

    def _save(self, force=False):
        if not force and self.path.exists():
            return
        if hasattr(self, "_journal"):
            self.state["journal_position"] = self._journal.position
        self.state["evolution"] = self.evolution.state_dict()
        self.state["panel_rng"] = self.panel_rng.getstate()
        save_checkpoint(self.path, self.state)

    def _event(self, kind, **values):
        self._event_callback({"event": kind, "phase": self.state["phase"], **values})

    def _validation_panels(self):
        width = self.settings.instances_per_panel
        return [
            {
                "dimension": n,
                "ids": list(self.data.validation[n][start : start + width]),
                "seeds": list(self.settings.validation_seeds),
                "evaluation_limit": dict(self.settings.validation_budgets or self.settings.budgets)[
                    n
                ],
            }
            for n, _ in self.settings.budgets
            for start in range(0, len(self.data.validation[n]), width)
        ]

    def _begin_generation(self):
        self.state["generation_started_unix"] = time.time()
        self.state["generation_start_costs"] = dict(self.state["costs"])
        panels = []
        for n, _ in () if self.data.shared_panels is not None else self.settings.budgets:
            ids = sorted(
                self.panel_rng.sample(self.data.training[n], self.settings.instances_per_panel)
            )
            seeds = []
            while len(seeds) < self.settings.solver_seeds_per_instance:
                value = self.panel_rng.getrandbits(64)
                if value not in seeds:
                    seeds.append(value)
            panels.append({"dimension": n, "ids": ids, "seeds": seeds})
        if self.data.shared_panels is not None:
            panels = self.data.shared_panels[self.evolution.generation]["panels"]
        identity = f"panel-{self.evolution.generation + 1:03d}"
        self.evolution.begin_panel(identity)
        self.state["panels"] = panels
        self.state["training_panels"].append(
            {"generation": self.evolution.generation, "panel_id": identity, "panels": panels}
        )
        self._save(force=True)  # 面板和solve seed先落盘，之后才允许任何准备/求解。

    def _prepare_fees(self, panels):
        if self.settings.preparation_mode == "end_to_end":
            return
        for panel in panels:
            missing = [name for name in panel["ids"] if name not in self.state["fees"]]
            if not missing:
                continue
            problems = self.data.problems(missing)
            description = {
                "protocol_id": self.protocol.identifier,
                "dimension": panel["dimension"],
                "problems": [p.instance_id for p in problems],
                "preparation_id": preparation_id(problems),
                **self.protocol.graph_identity(problems),
            }

            def validate(outcome, description=description, missing=missing):
                started = time.perf_counter()
                if (
                    outcome.get("status") != "completed"
                    or outcome.get("preparation_id") != description["preparation_id"]
                    or any(
                        json_value(outcome.get(k)) != json_value(v) for k, v in description.items()
                    )
                ):
                    raise RuntimeError(f"面板准备失败，保留身份等待排查: {outcome.get('error')}")
                fees = outcome["registration_fees"]
                if set(fees) != set(missing) or any(
                    type(fees[name].get(field)) not in (int, float)
                    or not math.isfinite(fees[name][field])
                    or fees[name][field] < 0
                    for name in missing
                    for field in ("cheap_seconds", "preparation_seconds")
                ):
                    raise ValueError("准备费用表无效")
                return {
                    name: {
                        "cheap_seconds": fees[name]["cheap_seconds"],
                        "preparation_seconds": fees[name]["preparation_seconds"],
                    }
                    for name in missing
                }, time.perf_counter() - started

            record, _ = self._operation(
                "preparation", description, lambda worker, p=problems: worker.prepare(p), validate
            )
            for problem in problems:
                self.state["fees"][problem.instance_id] = {
                    **record["checked"][problem.instance_id],
                    "preparation_record": record["key"],
                    "protocol_id": self.protocol.identifier,
                }
            self._save()  # 全部面板完成冻结之后才进入第一个个体。

    def _task(self, program, occurrence, panel, index):
        problems = self.data.problems(panel["ids"])
        if any(p.dimension != panel["dimension"] for p in problems):
            raise ValueError("实例规模与已确定面板不符")
        charges = None
        if self.settings.preparation_mode == "cached_charged":
            fees = self.state["fees"]
            charges = tuple(
                (
                    p.instance_id,
                    fees[p.instance_id]["cheap_seconds"],
                    fees[p.instance_id]["preparation_seconds"],
                )
                for p in problems
            )
        return SolveTask(
            f"{self.run_id}:{occurrence}:panel{index}",
            program,
            problems,
            tuple((p.instance_id, seed) for p in problems for seed in panel["seeds"]),
            dict(self.settings.budgets)[panel["dimension"]]
            if self.settings.budget_kind == "wall_clock"
            else None,
            self.settings.preparation_mode,
            self.settings.experiment_mask,
            charges,
            panel.get("evaluation_limit", dict(self.settings.budgets)[panel["dimension"]])
            if self.settings.budget_kind == "search_tour_evaluations"
            else None,
        )

    def _evaluate(self, program, occurrence, panels):
        self._immutable_record(
            self.directory / "programs" / f"{program.identifier}.json",
            {
                "program": program.to_dict(),
                "program_id": program.identifier,
                "expression": str(individual_from_program(program, self.evolution.pset)),
            },
        )
        tasks, scores = [], []
        for index, panel in enumerate(panels):
            task = self._task(program, occurrence, panel, index)
            labels = self.data.labels(task.problems)

            def validate(outcome, task=task, labels=labels):
                started = time.perf_counter()
                score = score_panel(task, self.protocol, outcome, labels)
                return json_value(asdict(score)), time.perf_counter() - started

            key = task.task_id(self.protocol)
            if key in self.state["completed"]:
                checked = self._completed_score(key)
                newly_completed = False
            else:
                record, newly_completed = self._operation(
                    "solve",
                    task.manifest(self.protocol),
                    lambda worker, t=task: worker.submit(t),
                    validate,
                )
                checked = record["checked"]
            score = PanelFitness(
                **{**checked, "members": tuple(tuple(v) for v in checked["members"])}
            )
            tasks.append(task)
            scores.append(score)
            if newly_completed:
                self._new_solves += 1
                self._event(
                    "solve_completed",
                    occurrence=occurrence,
                    panel=index,
                    task_id=task.task_id(self.protocol),
                    failed=score.failed,
                    total=self.state["costs"]["solve_jobs"],
                )
                if self._stop_after is not None and self._new_solves >= self._stop_after:
                    raise TrainingPaused("按预定任务边界暂停，活动任务已完成")
        value = aggregate_panels(
            tuple(tasks), self.protocol, tuple(scores), self.protocol.dimensions
        )
        self._last_scores = scores
        return value, [task.task_id(self.protocol) for task in tasks]

    def _evaluate_population(self, members, panels):
        """每个规模一次展开个体×实例×seed×蚂蚁；整批事务保存，沿用逐个体评分记录。"""
        for program, _ in members:
            self._immutable_record(
                self.directory / "programs" / f"{program.identifier}.json",
                {
                    "program": program.to_dict(),
                    "program_id": program.identifier,
                    "expression": str(individual_from_program(program, self.evolution.pset)),
                },
            )
        for index, panel in enumerate(panels):
            tasks = [
                self._task(program, occurrence, panel, index) for program, occurrence in members
            ]
            missing = [
                task for task in tasks if task.task_id(self.protocol) not in self.state["completed"]
            ]
            size = self.settings.population_batch_size
            for start in range(0, len(missing), size):
                chunk = tuple(missing[start : start + size])
                labels = self.data.labels(chunk[0].problems)
                description = {
                    "occurrence_id": "population:" + "|".join(task.occurrence_id for task in chunk),
                    "members": [task.manifest(self.protocol) for task in chunk],
                }

                def validate(outcome, chunk=chunk, labels=labels):
                    if len(outcome["members"]) != len(chunk):
                        raise ValueError("种群返回成员数不完整")
                    checked = []
                    for task, member in zip(chunk, outcome["members"], strict=True):
                        before = time.perf_counter()
                        score = score_panel(task, self.protocol, member, labels)
                        checked.append(
                            {
                                "checked": json_value(asdict(score)),
                                "evaluation_seconds": time.perf_counter() - before,
                            }
                        )
                    return checked, sum(item["evaluation_seconds"] for item in checked)

                _, completed = self._operation(
                    "population",
                    description,
                    lambda worker, chunk=chunk: worker.submit_population(chunk),
                    validate,
                )
                if completed:
                    self._new_solves += len(chunk)
                    self._event(
                        "population_completed",
                        individuals=len(chunk),
                        panel=index,
                        total=self.state["costs"]["solve_jobs"],
                    )
                    if self._stop_after is not None and self._new_solves >= self._stop_after:
                        raise TrainingPaused("完整 GPU 种群批次已保存，按任务边界暂停")

    def _paired_metrics(self, scores, panels):
        if self.data.baseline_cache is None:
            return None
        from gp_faco.experiment_v2 import paired_progress

        iterations = {
            panel.get("evaluation_limit", dict(self.settings.budgets)[panel["dimension"]])
            // self.protocol.settings.ants_for(panel["dimension"])
            for panel in panels
        }
        if len(iterations) != 1:
            raise ValueError("配对学习曲线要求两个规模使用相同迭代数")
        return paired_progress(scores, panels, self.data.baseline_cache, iterations.pop())

    def _generation_report(self):
        """只读取已经评分的冠军记录，不重新运行或重复评分该代 fitness。"""
        winner = self.evolution.winners[self.evolution.generation]
        program = Program.from_dict(winner["program"])
        index = next(
            i
            for i, individual in enumerate(self.evolution.population)
            if individual.program_id == program.identifier
        )
        _, task_ids = self._evaluate(
            program,
            f"generation{self.evolution.generation}:individual{index}",
            self.state["panels"],
        )
        metrics = self._paired_metrics(self._last_scores, self.state["panels"])
        previous = self.state["generation_metrics"].get(str(self.evolution.generation), {})
        start = self.state.get("generation_started_unix")
        wall_seconds = previous.get("wall_seconds")
        if wall_seconds is None and start is not None:
            wall_seconds = max(0.0, time.time() - start)
        start_costs = self.state.get("generation_start_costs", {})
        fitnesses = [v.fitness.values[0] for v in self.evolution.population]
        row = {
            "generation": self.evolution.generation + 1,
            "program_id": program.identifier,
            "fitness": winner["fitness"],
            "population_mean_gap": finite_or_none(statistics.mean(fitnesses)),
            "population_median_gap": finite_or_none(statistics.median(fitnesses)),
            "population_failed": sum(not math.isfinite(v) for v in fitnesses),
            "champion_nodes": len(program.opcode),
            "paired_baselines": metrics,
            "costs_cumulative": dict(self.state["costs"]),
            "task_ids": task_ids,
            "wall_seconds": wall_seconds,
            "time_scope": "panel_selection_through_champion_including_pauses_excluding_monitor",
            "generation_costs": previous.get(
                "generation_costs",
                {
                    name: value - start_costs.get(name, 0)
                    for name, value in self.state["costs"].items()
                },
            ),
        }
        self.state["generation_metrics"][str(self.evolution.generation)] = row
        save_checkpoint(
            self.directory / "learning_curve.json",
            {
                "generations": list(self.state["generation_metrics"].values()),
                "monitoring": list(self.state["monitoring_results"].values()),
            },
        )
        self._event("generation_metrics", **row)
        every = self.settings.monitoring_every
        key = str(self.evolution.generation)
        if (
            every
            and (self.evolution.generation + 1) % every == 0
            and key not in self.state["monitoring_results"]
        ):
            panels = self.data.monitoring_panels
            if not panels:
                raise ValueError("启用代际监控需要固定开发面板")
            self._prepare_fees(panels)
            fitness, _ = self._evaluate(program, f"monitor:g{self.evolution.generation}", panels)
            self.state["monitoring_results"][key] = {
                "generation": self.evolution.generation + 1,
                "program_id": program.identifier,
                "fitness": finite_or_none(fitness),
                "paired_baselines": self._paired_metrics(self._last_scores, panels),
                "costs_cumulative": dict(self.state["costs"]),
            }
            save_checkpoint(
                self.directory / "learning_curve.json",
                {
                    "generations": list(self.state["generation_metrics"].values()),
                    "monitoring": list(self.state["monitoring_results"].values()),
                },
            )
        self._save(force=True)

    def run(self, *, stop_after_tasks: int | None = None, through: str = "validation"):
        if self._closed:
            raise RuntimeError("训练句柄已经关闭")
        if stop_after_tasks is not None and (
            type(stop_after_tasks) is not int or stop_after_tasks < 1
        ):
            raise ValueError("暂停界限必须为正整数任务数")
        self._stop_after = stop_after_tasks
        if through not in ("training", "validation"):
            raise ValueError("训练终点必须为 training 或 validation")
        try:
            while self.state["phase"] == "training":
                if self.evolution.panel_id is None:
                    self._begin_generation()
                self._prepare_fees(self.state["panels"])
                if self.settings.population_batch_size > 1:
                    self._evaluate_population(
                        [
                            (
                                export_tree(individual),
                                f"generation{self.evolution.generation}:individual{index}",
                            )
                            for index, individual in enumerate(self.evolution.population)
                            if not individual.fitness.valid
                        ],
                        self.state["panels"],
                    )
                for index, individual in enumerate(self.evolution.population):
                    if individual.fitness.valid:
                        continue
                    program = export_tree(individual)
                    value, _ = self._evaluate(
                        program,
                        f"generation{self.evolution.generation}:individual{index}",
                        self.state["panels"],
                    )
                    self.evolution.assign(index, value, self.evolution.panel_id, program.identifier)
                    self._save()
                if len(self.evolution.winners) == self.evolution.generation:
                    winner = self.evolution.finish_generation()
                    self._save(force=True)
                    self._event(
                        "generation_completed",
                        generation=self.evolution.generation,
                        winner=winner["program"],
                        fitness=winner["fitness"],
                    )
                self._generation_report()
                self._immutable_record(
                    self.directory / "generations" / f"{self.evolution.generation:04d}.json",
                    {
                        "run_id": self.run_id,
                        "evolution": self.evolution.state_dict(),
                        "panels": self.state["panels"],
                    },
                )
                if self.evolution.advance():
                    self.state["panels"] = []
                    self._save()
                else:
                    self.state["shortlist"] = [p.to_dict() for p in self.evolution.shortlist()]
                    self._close_worker()
                    self.state["phase"] = "validation"
                    self._save()
            if self.state["phase"] == "validation":
                if through == "training":
                    self._close_worker()
                    self._save(force=True)
                    report = self.summary()
                    save_checkpoint(self.directory / "training_summary.json", report)
                    return report
                self._prepare_fees(self.state["validation_panels"])
                if self.settings.population_batch_size > 1:
                    self._evaluate_population(
                        [
                            (
                                Program.from_dict(value),
                                f"validation:{Program.from_dict(value).identifier}",
                            )
                            for value in self.state["shortlist"]
                            if Program.from_dict(value).identifier
                            not in self.state["validation_results"]
                        ],
                        self.state["validation_panels"],
                    )
                for value in self.state["shortlist"]:
                    program = Program.from_dict(value)
                    if program.identifier in self.state["validation_results"]:
                        continue
                    fitness, identities = self._evaluate(
                        program, f"validation:{program.identifier}", self.state["validation_panels"]
                    )
                    self.state["validation_results"][program.identifier] = {
                        "fitness": finite_or_none(fitness),
                        "task_ids": identities,
                        "nodes": len(program.opcode),
                        "expression": str(individual_from_program(program, self.evolution.pset)),
                        "paired_baselines": self._paired_metrics(
                            self._last_scores, self.state["validation_panels"]
                        ),
                    }
                    self._save()
                self._select()
            self._close_worker()
            self._save(force=True)
            report = self.summary()
            save_checkpoint(self.directory / "summary.json", report)
            return report
        except TrainingPaused as error:
            self._close_worker()
            self._save(force=True)
            return {
                "status": "paused",
                "run_id": self.run_id,
                "reason": str(error),
                "solve_jobs": self.state["costs"]["solve_jobs"],
            }
        finally:
            # 异常路径不写回内存状态：保留结果落盘/checkpoint未前进的真实恢复窗口。
            try:
                self._close_worker()
            finally:
                self._journal.close()
                self._lease.close()
                self._closed = True

    def summary(self):
        return {
            "status": self.state["phase"],
            "run_id": self.run_id,
            "scope": self.settings.scope,
            "selected": self.state["selected"],
            "validation_results": self.state["validation_results"],
            "training_panels": self.state["training_panels"],
            "costs": self.state["costs"],
            "worker_history": self.state["worker_history"],
            "variation_counts": self.evolution.variation_counts,
            "generation_metrics": self.state["generation_metrics"],
            "monitoring_results": self.state["monitoring_results"],
            "completed_records": len(self.state["completed"]),
        }

    def _select(self):
        results = self.state["validation_results"]
        finite = [name for name, result in results.items() if result["fitness"] is not None]
        if not finite:
            self.state["phase"] = "failed"
            self._save()
            return
        selected = min(
            finite, key=lambda name: (results[name]["fitness"], results[name]["nodes"], name)
        )
        program = next(p for p in self.evolution.shortlist() if p.identifier == selected)
        self.state["selected"] = {"program_id": selected, **results[selected]}
        save_checkpoint(
            self.directory / "selected_program.json",
            {
                "export_version": 1,
                "run_id": self.run_id,
                "scope": self.settings.scope,
                "program": program.to_dict(),
                "selection": self.state["selected"],
                "manifest": self.state["manifest"],
                "costs": self.state["costs"],
            },
        )
        self.state["phase"] = "complete"
        self._save(force=True)
