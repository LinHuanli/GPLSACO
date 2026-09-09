"""M10/M01独立重训：复用标准DEAP、全量评价与恢复，仅改变任务动作限制。"""

from dataclasses import fields

from gp_faco.factorial_policy import FactorialPolicy
from gp_faco.training import TrainingRun
from gp_faco.worker import FactorialTask, SolveTask


class FactorialTrainingRun(TrainingRun):
    def __init__(self, *args, factorial_policy, **kwargs):
        if type(factorial_policy) is not FactorialPolicy or factorial_policy.variant not in (
            "M10",
            "M01",
        ):
            raise ValueError("单因素独立重训仅接受M10/M01")
        self.factorial_policy = factorial_policy
        super().__init__(*args, **kwargs)

    def _training_manifest(self):
        if self.settings.budget_kind != "search_tour_evaluations":
            raise ValueError("析因重训不接受墙钟预算")
        if self.settings.evolution.no_feedback:
            raise ValueError("E2单因素训练使用Full终端集")
        self.factorial_policy.validate_mask(self.settings.experiment_mask)
        manifest = super()._training_manifest()
        manifest["factorial_policy"] = self.factorial_policy.to_dict()
        manifest["factorial_policy_id"] = self.factorial_policy.identifier
        return manifest

    def _task(self, program, occurrence, panel, index):
        task = super()._task(program, occurrence, panel, index)
        return FactorialTask(
            **{f.name: getattr(task, f.name) for f in fields(SolveTask)},
            factorial_policy=self.factorial_policy,
        )
