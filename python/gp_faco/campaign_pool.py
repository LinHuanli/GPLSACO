"""三次独立进化共享全部空闲A5000；训练、验证与测试均按完整分片记账。"""

from __future__ import annotations

import json
import subprocess
import time

from gp_faco.campaign import (GPU_BASELINE, NATIVE_BASELINE, baseline_groups, baseline_manifest,
    candidate_programs, comparison_statistics, freeze_methods, publish_baselines, read, training_directory)
from gp_faco.campaign_scheduler import CampaignScheduler
from gp_faco.campaign_tsp500 import select_light_validation, validation_ranking
from gp_faco.checkpoint import atomic_json
from gp_faco.experiment_v2 import BaselineCache, batches
from gp_faco.remote import PROJECT, discover_a5000, process_alive, remote_python


class PopulationScheduler(CampaignScheduler):
    def __init__(self, directory, *, hosts=(), gpu_limit=None, poll_seconds=1):
        super().__init__(directory, hosts=hosts, gpu_limit=gpu_limit, poll_seconds=poll_seconds)
        self.workers = {p.parent.name: read(p) for p in (directory / "workers").glob("*/worker.json")}
        self._worker_alive = {}
        self._baseline_published = (directory / "baselines_ready.json").exists()

    def add(self, kind, suffix, **payload):
        if kind == "training":
            payload["resource"] = "coordinator"
        key = super().add(kind, suffix, **payload)
        if kind == "training":
            # 训练进程只维护DEAP和外部评分，不占GPU，也不占原生FACO的CPU任务槽。
            self.jobs[key]["resource"] = "coordinator"
        return key

    def inherited_baselines(self):
        if self._inherited is not None:
            return self._inherited
        self._inherited = {}
        for origin in self.config["inherited_baseline_directories"]:
            path = PROJECT / origin / "baselines.sqlite"
            if not path.exists():
                continue
            cache = BaselineCache(path, baseline_manifest(), readonly=True)
            try:
                for values in cache.connection.execute("SELECT method,dimension,instance_id,seed,iterations,cost,gap,seconds,tour FROM scores WHERE dimension=500"):
                    method, n, name, seed, it, cost, gap, seconds, tour = values
                    self._inherited.setdefault((method, n, name, int(seed), it), {
                        "method": method, "dimension": n, "instance_id": name, "seed": int(seed),
                        "iterations": it, "cost": cost, "gap_percent": gap, "seconds": seconds,
                        "tour": json.loads(tour), "status": "completed"})
            finally:
                cache.close()
        return self._inherited

    def advance(self):
        frozen = read(self.directory / "frozen.json")
        self.stage = "baselines"
        if not self._baseline_published:
            inherited = self.inherited_baselines()
            for index, group in enumerate(baseline_groups(self.config, self.declaration, frozen)):
                for kind, method in (("baseline_gpu", GPU_BASELINE), ("baseline_native", NATIVE_BASELINE)):
                    reused, missing = [], []
                    for name, seed in group["pairs"]:
                        row = inherited.get((method, group["dimension"], name, seed, group["iterations"]))
                        (reused if row else missing).append(row if row else (name, seed))
                    self.add(kind, f"{index:04d}", **{**group, "pairs": missing, "inherited_rows": reused})
            base_jobs = self.group(("baseline_gpu", "baseline_native"))
            if not self.finished(base_jobs):
                return
            publish_baselines(self.directory, self.results(base_jobs))
            self._baseline_published = True
            self._inherited = {}  # 后续训练直接读取只读baseline表，释放含tour的导入缓存。
        self.stage = "training_and_validation"
        for seed in self.config["seeds"]:
            training = self.add("training", str(seed), seed=seed)
            if self.states[training] != "completed":
                continue
            if seed not in self._candidates:
                self._candidates[seed] = candidate_programs(self.directory, seed)
            programs = self._candidates[seed]
            run = training_directory(self.directory, seed)
            for pindex, panel in enumerate(self.declaration["validation"]):
                for bindex, pairs in enumerate(batches(panel, self.config["gpu_replicas"])):
                    for start in range(0, len(programs), self.config["population_shard_size"]):
                        self.add("validation_quick", f"{seed}-p{pindex:02d}-b{bindex:02d}-c{start:03d}",
                            seed=seed, dimension=500, pairs=pairs,
                            iterations=self.config["validation_screen_iterations"],
                            programs=programs[start:start + self.config["population_shard_size"]])
            quick_jobs = self.group(("validation_quick",), seed)
            if not self.finished(quick_jobs):
                continue
            if not (run / "validation_screen.json").exists():
                scores, order = validation_ranking(self.directory, seed, list(self.results(quick_jobs)), programs, "validation")
                ids = order[:self.config["validation_finalists"]]
                atomic_json(run / "validation_screen.json", {"evaluations": scores,
                    "finalists": [next(p for p in programs if p["program_id"] == key) for key in ids]})
            finalists = read(run / "validation_screen.json")["finalists"]
            for pindex, panel in enumerate(self.declaration["validation_final"]):
                for bindex, pairs in enumerate(batches(panel, self.config["gpu_replicas"])):
                    self.add("validation_final", f"{seed}-p{pindex:02d}-b{bindex:02d}", seed=seed,
                        dimension=500, pairs=pairs, iterations=self.config["validation_iterations"], programs=finalists)
            final_jobs = self.group(("validation_final",), seed)
            if self.finished(final_jobs) and not (run / "selected_program.json").exists():
                select_light_validation(self.directory, seed, self.results(quick_jobs), self.results(final_jobs), programs)
        if not all((training_directory(self.directory, s) / "selected_program.json").exists() for s in self.config["seeds"]):
            return
        if not (self.directory / "methods_frozen.json").exists():
            freeze_methods(self.directory)
        self.stage = "test_and_explanation"
        for panel in self.declaration["test"]:
            for index, pairs in enumerate(batches(panel, self.config["gpu_replicas"])):
                for kind in ("test_gpu", "test_native"):
                    self.add(kind, f"n{panel['dimension']}-p{index:03d}", dimension=panel["dimension"],
                        pairs=pairs, iterations=self.config["test_iterations"], order_rotation=index % 4)
        for panel in self.declaration["explain"]:
            self.add("explain", f"n{panel['dimension']}", dimension=panel["dimension"],
                pairs=[(name, 17) for name in panel["ids"][:2]], iterations=self.config["test_iterations"])
        tests = self.group(("test_gpu", "test_native"))
        explanation = self.group(("explain",))
        if self.finished(tests) and not (self.directory / "statistics.json").exists():
            rows = [r for result in self.results(tests) for r in result["rows"]]
            atomic_json(self.directory / "statistics.json", comparison_statistics(rows, self.config, self.declaration["test"]))
        if self.finished(tests + explanation):
            self.stage = "complete"
            atomic_json(self.directory / "stop_workers.json", {"reason": "campaign_complete"})

    def reconcile(self):
        # 新分片请求仅首次读入；普通编号就是任务身份，不扫描或摘要程序内容。
        for path in (self.directory / "requests").glob("*.json"):
            if path.stem not in self.jobs:
                job = read(path)
                self.add(job["kind"], job["id"][len(job["kind"]) + 1:],
                    **{k: v for k, v in job.items() if k not in ("id", "kind", "resource")})
        # 常驻worker执行失败后退出。训练客户端必须看到最终失败，不能永远等一个缺失分片。
        next_discovery = self.next_discovery
        changed = super().reconcile()
        # 常驻worker结束分片后直接接下一项；不要继承短进程队列每次完成就扫描全集群的行为。
        # 新空闲GPU仍按固定60秒周期发现，故障也不会在每个轮询上触发重复SSH探测。
        self.next_discovery = next_discovery
        for key, state in self.states.items():
            if state == "failed" and self.jobs[key]["kind"] == "population":
                path = self.directory / "jobs" / key
                if not (path / "terminal_failure.json").exists():
                    atomic_json(path / "terminal_failure.json", {"error": read(path / "runtime.json", {}).get("error", "worker失效且重试耗尽")})
        failed = [k for k, v in self.states.items() if v == "failed"]
        if failed and not (self.directory / "campaign_abort.json").exists():
            atomic_json(self.directory / "campaign_abort.json", {"jobs": failed, "error": f"失败任务：{failed[0]}"})
        return changed

    def worker_available(self, uuid):
        info = self.workers[uuid]
        runtime = read(self.directory / "workers" / uuid / "runtime.json")
        if not runtime:
            return None
        if runtime["status"] in ("failed", "resource_wait", "stopped"):
            return None
        cached = self._worker_alive.get(uuid)
        if cached is None or time.monotonic() >= cached[0]:
            alive = process_alive(runtime)
            self._worker_alive[uuid] = (time.monotonic() + 60, alive)
        else:
            alive = cached[1]
        if not alive:
            return None
        return info if runtime["status"] == "idle" else None

    def launch_worker(self, device):
        uuid = device["uuid"]
        path = self.directory / "workers" / uuid
        path.mkdir(parents=True, exist_ok=True)
        atomic_json(path / "worker.json", device)
        self.workers[uuid] = device
        self._worker_alive.pop(uuid, None)
        remote_python(device["host"],
            "import subprocess,sys\n"
            f"root={str(PROJECT)!r}\n"
            "print(subprocess.check_output([sys.executable,root+'/scripts/campaign_gpu_worker.py',"
            "'--campaign',sys.argv[1],'--gpu',sys.argv[2],'--launch'],text=True,cwd=root),end='')\n",
            (self.directory, uuid))

    def submit_pool(self, key, device):
        path = self.directory / "jobs" / key
        previous = read(path / "assigned.json", {})
        assigned = {"host": device["host"], "gpu_uuid": device["uuid"],
            "attempt": previous.get("attempt", 0) + 1, "launched_unix": time.time(), "persistent_worker": True}
        atomic_json(path / "assigned.json", assigned)
        self.active[key] = assigned
        self.states[key] = "running"
        worker = read(self.directory / "workers" / device["uuid"] / "runtime.json")
        # 先由协调器发布已派单进程记录，避免远端首次创建文件时的可见性延迟被误判为未启动。
        atomic_json(path / "runtime.json", {
            **worker, "job": key, "attempt": assigned["attempt"], "status": "assigned",
            "assigned_unix": assigned["launched_unix"]})
        atomic_json(self.directory / "workers" / device["uuid"] / "request.json", {
            "job": key, "attempt": assigned["attempt"], "worker_pid": worker["pid"], "worker_start_ticks": worker["start_ticks"]})

    def dispatch(self):
        pending = [k for k, state in self.states.items() if state == "pending"]
        for key in pending:
            if self.jobs[key]["kind"] == "training":
                self.submit(key, self.config["cpu_host"])
        self.dispatch_native(pending)
        gpu_jobs = [k for k in pending if self.jobs[k]["resource"] == "gpu"]

        def priority(key):
            job = self.jobs[key]
            rank = {"baseline_gpu": 0, "population": 1, "validation_quick": 2, "validation_final": 2,
                    "test_gpu": 3, "explain": 4}[job["kind"]]
            return rank, job.get("generation", 0), job.get("shard_index", 0), job.get("seed", 0), key

        gpu_jobs.sort(key=priority)
        busy = {a["gpu_uuid"] for a in self.active.values() if a["gpu_uuid"]}
        available = [device for uuid in self.workers if uuid not in busy
                     for device in [self.worker_available(uuid)] if device]
        for key, device in zip(gpu_jobs, available, strict=False):
            self.submit_pool(key, device)
        remaining = [k for k in gpu_jobs if self.states[k] == "pending"]
        if not remaining or time.monotonic() < self.next_discovery:
            return
        self.next_discovery = time.monotonic() + self.config["gpu_discovery_seconds"]
        try:
            devices = discover_a5000(self.hosts)
        except (subprocess.SubprocessError, OSError, ValueError):
            return
        for device in devices:
            if device["host_cpus"] == self.config["native_initial_routes"] and device["host"] not in self.cpu_hosts:
                self.cpu_hosts.append(device["host"])
        atomic_json(self.directory / "native_hosts.json", self.cpu_hosts)
        self.dispatch_native(pending)
        busy = {a["gpu_uuid"] for a in self.active.values() if a["gpu_uuid"]}
        active_workers = sum(1 for uuid in self.workers if uuid in busy or self.worker_available(uuid))
        capacity = len(remaining) if self.gpu_limit is None else max(0, self.gpu_limit - active_workers)
        for device in devices[:capacity]:
            if device["uuid"] not in busy:
                try:
                    self.launch_worker(device)
                except (subprocess.SubprocessError, OSError, ValueError):
                    pass  # 启动应答丢失时保留worker目录，下一轮依据实际进程和租约恢复。
