"""CPU代际协调器向共享A5000池提交独立种群分片；按原顺序收齐后才推进进化。"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import TimeoutError

from gp_faco.campaign import read, source
from gp_faco.experiment_v2 import write_once
from gp_faco.remote import process_receipt
from gp_faco.training import TrainingRun
from gp_faco.worker import _population_members, faco_ants


class PopulationFuture:
    def __init__(self, directory, jobs, *, single=False):
        self.directory, self.jobs, self.single = directory, jobs, single
        self.completed = {}

    def result(self, timeout=None):
        until = None if timeout is None else time.monotonic() + timeout
        while True:
            aborted = read(self.directory / "campaign_abort.json")
            if aborted:
                raise RuntimeError(f"本轮依赖任务失败，停止提交新评价：{aborted['error']}")
            for job in self.jobs:
                if job in self.completed:
                    continue
                path = self.directory / "jobs" / job
                failed = read(path / "terminal_failure.json")
                if failed:
                    raise RuntimeError(f"GPU分片 {job} 失败：{failed['error']}")
                result = read(path / "result.json")
                if result is not None:
                    if result.get("status") != "completed":
                        raise RuntimeError(f"GPU分片 {job} 没有完整完成")
                    self.completed[job] = result
            if len(self.completed) == len(self.jobs):
                members = []
                for job in self.jobs:
                    for index, member in enumerate(self.completed[job]["members"]):
                        member = {**member, "raw_record": {
                            "path": str(self.directory / "jobs" / job / "result.json"), "member": index}}
                        members.append(member)
                return members[0] if self.single else {"members": members, "distributed": True}
            if until is not None and time.monotonic() >= until:
                raise TimeoutError()
            time.sleep(min(0.25, max(0, until - time.monotonic())) if until else 0.25)


class PopulationClient:
    def __init__(self, directory, seed, protocol):
        self.directory, self.seed, self.protocol = directory, seed, protocol
        self.chunk = read(directory / "campaign.json")["population_shard_size"]
        (directory / "requests").mkdir(exist_ok=True)

    def ready(self, timeout=None):
        # 这是CPU协调端身份；物理GPU与实际worker身份记在每个分片的execution/device中。
        return {**process_receipt(), "protocol_id": self.protocol.identifier, "kind": "shared_gpu_pool_client"}

    def submit_population(self, tasks):
        return self._submit(tuple(tasks))

    def submit(self, task):
        return self._submit((task,), single=True)

    def _submit(self, tasks, single=False):
        first = _population_members(tasks, self.protocol)
        jobs = []
        match = re.search(r"generation(\d+):", first.occurrence_id)
        generation = int(match[1]) if match else int(re.search(r"monitor:g(\d+)", first.occurrence_id)[1])
        role = "training" if match else "monitor"
        for start in range(0, len(tasks), self.chunk):
            chunk = tasks[start:start + self.chunk]
            # occurrence普通编号同时区分精英、重复程序和恢复位置，不按程序复用fitness。
            suffix = re.sub(r"[^A-Za-z0-9_.-]", "-", chunk[0].occurrence_id)
            job_id = f"population-{suffix}"
            job = {"id": job_id, "kind": "population", "resource": "gpu", "seed": self.seed,
                   "generation": generation, "shard_index": start // self.chunk, "role": role, "dimension": first.dimension,
                   "pairs": list(first.replicas), "iterations": first.evaluation_limit_per_colony // faco_ants(first.dimension),
                   "experiment_mask": first.experiment_mask, "programs": [t.program.to_dict() for t in chunk],
                   "occurrences": [t.occurrence_id for t in chunk], "protocol_id": self.protocol.identifier}
            write_once(self.directory / "requests" / f"{job_id}.json", job)
            jobs.append(job_id)
        return PopulationFuture(self.directory, jobs, single=single)

    def close(self):
        # 共享worker由全轮调度器管理，一个seed结束不会关闭其他seed正在使用的GPU。
        pass


class DistributedTrainingRun(TrainingRun):
    def _prepare_fees(self, panels):
        # cached FE模式不扣准备费。注册在实际接单GPU上执行并计时，避免先绑卡做重复准备。
        if self.settings.preparation_mode != "cached" or self.settings.budget_kind != "search_tour_evaluations":
            raise ValueError("共享GPU池只支持cached FE训练")


def solve_population_job(directory, job, session):
    session.set_colonies(len(job["pairs"]))
    with source() as dataset:
        problems = {name: dataset.load_instance(name) for name, _ in job["pairs"]}
        native, timing = session.solve_population(problems, job["pairs"], job["iterations"], job["programs"],
                                                  experiment_mask=job["experiment_mask"])
    count = len(native)
    members = []
    for occurrence, program, result in zip(job["occurrences"], job["programs"], native, strict=True):
        members.append({"task_id": occurrence, "occurrence_id": occurrence, "protocol_id": job["protocol_id"],
                        "program_id": program["program_id"], "dimension": job["dimension"],
                        "status": "completed", "native_result": result, "worker_pid": os.getpid(),
                        "device": session.device, "registration_seconds": timing["registration_seconds"] / count,
                        "worker_seconds": (timing["registration_seconds"] + timing["solve_seconds"]) / count,
                        "population_worker_seconds": timing["registration_seconds"] + timing["solve_seconds"]})
    return {"status": "completed", "kind": "population", "members": members,
            "total_fe": sum(r["total_tour_evaluations"] for r in native),
            "solve_seconds": timing["solve_seconds"], "device": session.device}
