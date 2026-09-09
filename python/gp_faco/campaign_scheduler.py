"""单协调器、多主机 A5000 队列；任务结束后派下一项，恢复只读取已有结果。"""

from __future__ import annotations

import fcntl
import json
import subprocess
import time
from collections import Counter

from gp_faco.campaign import (
    GPU_BASELINE,
    NATIVE_BASELINE,
    baseline_groups,
    baseline_manifest,
    candidate_programs,
    comparison_statistics,
    freeze_methods,
    publish_baselines,
    read,
    select_validation,
    training_directory,
)
from gp_faco.campaign_calibration import calibration_jobs, summarize_calibration
from gp_faco.checkpoint import atomic_json
from gp_faco.experiment_v2 import BaselineCache, batches, write_once
from gp_faco.remote import (
    PROJECT,
    discover_a5000,
    process_alive,
    process_receipt,
    remote_python,
)


def launch_remote(host, directory, job_id, gpu, attempt):
    return remote_python(
        host,
        "import subprocess,sys\n"
        f"root={str(PROJECT)!r}\n"
        "command=[sys.executable,root+'/scripts/campaign_job_v2.py',"
        "'--campaign',sys.argv[1],'--job',sys.argv[2],"
        "'--attempt',sys.argv[3],'--launch']\n"
        "if sys.argv[4]!='None': command+=['--gpu',sys.argv[4]]\n"
        "print(subprocess.check_output(command,text=True,cwd=root),end='')\n",
        (directory, job_id, attempt, gpu),
    )


def job_lock_active(host, directory):
    return remote_python(
        host,
        "import fcntl,json,sys\n"
        "f=open(sys.argv[1]+'/running.lock','a')\n"
        "try:\n fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n active=False\n"
        "except BlockingIOError:\n active=True\n"
        "print(json.dumps(active))\n",
        (directory,),
    )


class CampaignScheduler:
    def __init__(self, directory, *, hosts=(), gpu_limit=None, poll_seconds=5):
        self.directory = directory.resolve()
        self.config = read(self.directory / "campaign.json")
        self.declaration = read(self.directory / "panels.json")
        self.hosts, self.gpu_limit = hosts, gpu_limit
        if gpu_limit is not None and self.config["scope"] != "engineering_campaign_only":
            raise ValueError("正式本轮不限制可使用的空闲 A5000 总数")
        self.poll_seconds = poll_seconds
        self.lease = (self.directory / "campaign.lock").open("a")
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.jobs, self.states, self.active = {}, {}, {}
        for path in sorted((self.directory / "jobs").glob("*/job.json")):
            job = read(path)
            self.jobs[job["id"]] = job
            assigned = read(path.parent / "assigned.json")
            self.states[job["id"]] = (
                "completed" if (path.parent / "result.json").exists() else "pending"
            )
            if assigned and self.states[job["id"]] != "completed":
                self.active[job["id"]] = assigned
                self.states[job["id"]] = "running"
        self.next_discovery = 0.0
        self.stage = "waiting_horizon"
        self.started = read(self.directory / "coordinator.json", {}).get(
            "campaign_started", time.time()
        )
        atomic_json(
            self.directory / "coordinator.json",
            {**process_receipt(), "campaign_started": self.started, "status": "running"},
        )
        self._inherited = None
        self._last_report = 0.0
        self._next_liveness = {}
        self._candidates = {}
        self._calibration_manifest = read(self.directory / "calibration/manifest.json")
        self.cpu_hosts = read(
            self.directory / "native_hosts.json", [self.config.get("cpu_host", "cuda07")]
        )

    def add(self, kind, suffix, **payload):
        job_id = f"{kind}-{suffix}"
        if job_id not in self.jobs:
            job = {
                "id": job_id,
                "kind": kind,
                "resource": "cpu" if kind.endswith("_native") else "gpu",
                **payload,
            }
            path = self.directory / "jobs" / job_id
            path.mkdir(parents=True, exist_ok=True)
            write_once(path / "job.json", job)
            self.jobs[job_id] = job
            self.states[job_id] = "completed" if (path / "result.json").exists() else "pending"
            if not payload.get("pairs", [True]) and payload.get("inherited_rows"):
                atomic_json(
                    path / "result.json",
                    {
                        "status": "completed",
                        "kind": kind,
                        "job": job_id,
                        "rows": payload["inherited_rows"],
                        "reused_members": len(payload["inherited_rows"]),
                        "total_fe": 0,
                        "solve_seconds": 0.0,
                        "wall_seconds": 0.0,
                    },
                )
                self.states[job_id] = "completed"
        return job_id

    def group(self, kinds, seed=None):
        return [
            k
            for k, j in self.jobs.items()
            if j["kind"] in kinds and (seed is None or j.get("seed") == seed)
        ]

    def finished(self, jobs):
        return bool(jobs) and all(self.states[k] == "completed" for k in jobs)

    def results(self, jobs):
        for key in jobs:
            result = read(self.directory / "jobs" / key / "result.json")
            if not result or result["status"] != "completed":
                raise ValueError("任务结果尚未齐备")
            yield result

    def prepare_horizon(self):
        frozen = read(self.directory / "frozen.json")
        if frozen is not None:
            return frozen
        origin = PROJECT / self.config["source_protocol"]
        numeric = read(origin / "numeric_selection.json")
        if self._calibration_manifest:
            curve = read(self.directory / "calibration/summary.json")
            if curve is None:
                for job in calibration_jobs(self._calibration_manifest):
                    payload = {k: v for k, v in job.items() if k not in ("id", "kind", "resource")}
                    self.add("calibration", job["source_job_id"], **payload)
                jobs = self.group(("calibration",))
                if self.finished(jobs):
                    curve = summarize_calibration(self.directory, self.results(jobs))
        else:
            curve = read(origin / "curves/exact/summary.json")
        if not numeric or not curve:
            return None
        if numeric["numeric_backend"] != "exact":
            raise ValueError("本轮固定已经选定的 exact 后端")
        if curve["status"] != "complete" or any(r["status"] != "completed" for r in curve["rows"]):
            raise ValueError("开发曲线有失败，不能启动依赖训练")
        frozen = {
            "experiment_version": 2,
            "numeric_backend": "exact",
            "training_iterations": curve["selection"]["iterations"],
            "validation_iterations": 5000,
            "test_iterations": 5000,
            "horizon_selection": curve["selection"],
            "wall_clock_limit": None,
        }
        write_once(self.directory / "frozen.json", frozen)
        return frozen

    def inherited_baselines(self):
        if self._inherited is not None:
            return self._inherited
        self._inherited = {}
        if self.config["scope"] == "engineering_campaign_only":
            return self._inherited
        path = PROJECT / self.config["source_protocol"] / "baselines.sqlite"
        if not path.exists():
            return self._inherited
        cache = BaselineCache(path, baseline_manifest(), readonly=True)
        try:
            for row in cache.connection.execute(
                "SELECT method,dimension,instance_id,seed,iterations,cost,gap,seconds,tour"
                " FROM scores"
            ):
                method, n, name, seed, iterations, cost, gap, seconds, tour = row
                self._inherited[method, n, name, int(seed), iterations] = {
                    "method": method,
                    "dimension": n,
                    "instance_id": name,
                    "seed": int(seed),
                    "iterations": iterations,
                    "cost": cost,
                    "gap_percent": gap,
                    "seconds": seconds,
                    "tour": json.loads(tour),
                    "status": "completed",
                }
        finally:
            cache.close()
        return self._inherited

    def advance(self):
        frozen = self.prepare_horizon()
        self.stage = "baselines" if frozen else "calibration_and_fixed_baselines"
        inherited = self.inherited_baselines()
        # 固定 5000 迭代的监控/验证对照不依赖训练 H，可与标定同时执行。
        # 按用途编号，H 冻结后追加训练对照，不改变已派发任务的编号。
        if self.config["scope"] == "full_three_seed_v2":
            fixed = {**self.declaration, "training": []}
            groups = [
                (f"fixed-{i:04d}", g) for i, g in enumerate(baseline_groups(self.config, fixed, {}))
            ]
            if frozen:
                training = {**self.declaration, "monitor": [], "validation": []}
                groups += [
                    (f"train-{i:04d}", g)
                    for i, g in enumerate(baseline_groups(self.config, training, frozen))
                ]
        elif frozen:
            groups = [
                (f"{i:04d}", g)
                for i, g in enumerate(baseline_groups(self.config, self.declaration, frozen))
            ]
        else:
            return
        for suffix, group in groups:
            for kind, method in (
                ("baseline_gpu", GPU_BASELINE),
                ("baseline_native", NATIVE_BASELINE),
            ):
                reused, missing = [], []
                for name, seed in group["pairs"]:
                    row = inherited.get(
                        (method, group["dimension"], name, seed, group["iterations"])
                    )
                    if row is None:
                        missing.append((name, seed))
                    else:
                        reused.append(row)
                self.add(kind, suffix, **{**group, "pairs": missing, "inherited_rows": reused})
        if frozen is None:
            return
        base_jobs = self.group(("baseline_gpu", "baseline_native"))
        if not self.finished(base_jobs):
            return
        if not (self.directory / "baselines_ready.json").exists():
            publish_baselines(self.directory, self.results(base_jobs))
        self.stage = "training_and_validation"
        width = self.config["instances_per_panel"] * 2
        for seed in self.config["seeds"]:
            training = self.add("training", str(seed), seed=seed)
            if self.states[training] != "completed":
                continue
            if seed not in self._candidates:
                self._candidates[seed] = candidate_programs(self.directory, seed)
            programs = self._candidates[seed]
            for panel in self.declaration["validation"]:
                for panel_index, pairs in enumerate(batches(panel, width)):
                    for start in range(0, len(programs), 128):
                        self.add(
                            "validation",
                            f"{seed}-n{panel['dimension']}-p{panel_index:02d}-c{start:03d}",
                            seed=seed,
                            dimension=panel["dimension"],
                            pairs=pairs,
                            iterations=self.config["validation_iterations"],
                            programs=programs[start : start + 128],
                        )
            selected = training_directory(self.directory, seed) / "selected_program.json"
            validation = self.group(("validation",), seed)
            if self.finished(validation) and not selected.exists():
                select_validation(self.directory, seed, self.results(validation))
        if not all(
            (training_directory(self.directory, s) / "selected_program.json").exists()
            for s in self.config["seeds"]
        ):
            return
        if not (self.directory / "methods_frozen.json").exists():
            freeze_methods(self.directory)
        self.stage = "test_and_explanation"
        for panel in self.declaration["test"]:
            for index, pairs in enumerate(batches(panel, width)):
                for kind in ("test_gpu", "test_native"):
                    self.add(
                        kind,
                        f"n{panel['dimension']}-p{index:03d}",
                        dimension=panel["dimension"],
                        pairs=pairs,
                        iterations=self.config["test_iterations"],
                        order_rotation=index % 4,
                    )
        for panel in self.declaration["monitor"]:
            self.add(
                "explain",
                f"n{panel['dimension']}",
                dimension=panel["dimension"],
                pairs=[(name, 17) for name in panel["ids"][:2]],
                iterations=self.config["validation_iterations"],
            )
        tests = self.group(("test_gpu", "test_native"))
        explanation = self.group(("explain",))
        if self.finished(tests) and not (self.directory / "statistics.json").exists():
            rows = [r for result in self.results(tests) for r in result["rows"]]
            atomic_json(
                self.directory / "statistics.json",
                comparison_statistics(rows, self.config, self.declaration["test"]),
            )
        if self.finished(tests + explanation):
            self.stage = "complete"

    def reconcile(self):
        changed = False
        for key, assigned in list(self.active.items()):
            path = self.directory / "jobs" / key
            runtime = read(path / "runtime.json")
            result = read(path / "result.json") if (path / "result.json").exists() else None
            if result:
                self.states[key] = "completed"
                self.active.pop(key)
                if assigned["gpu_uuid"]:
                    self.next_discovery = 0.0
                changed = True
                continue
            if (
                runtime
                and runtime["status"] == "running"
                and (time.monotonic() < self._next_liveness.get(key, 0))
            ):
                continue
            alive = process_alive(runtime) if runtime else None
            self._next_liveness[key] = time.monotonic() + 60
            if result and runtime and not alive:
                self.states[key] = "completed"
                self.active.pop(key)
                if assigned["gpu_uuid"]:
                    self.next_discovery = 0.0
                changed = True
                continue
            if alive:
                continue
            # 未拿到进程记录时询问远端租约；SSH 断开不代表任务失败。
            if runtime is None:
                if time.time() - assigned["launched_unix"] < 60:
                    continue
                try:
                    if job_lock_active(assigned["host"], path):
                        continue
                except (subprocess.SubprocessError, OSError, ValueError):
                    continue
            if runtime and runtime["status"] == "completed":
                # 等待共享文件系统上的完成结果可见，不根据缓存的文件属性重算。
                continue
            if result:
                self.states[key] = "completed"
            elif runtime and runtime["status"] == "resource_wait":
                self.states[key] = "pending"
            else:
                failures = read(path / "failures.json", [])
                attempt = assigned["attempt"]
                if not any(v["attempt"] == attempt for v in failures):
                    failures.append(
                        {
                            "attempt": attempt,
                            "host": assigned["host"],
                            "error": runtime.get("error", "process_exited_without_result")
                            if runtime
                            else "launch_not_observed",
                        }
                    )
                    atomic_json(path / "failures.json", failures)
                self.states[key] = (
                    "pending"
                    if len(failures) <= self.config["infrastructure_retries"]
                    and (runtime is None or runtime.get("retryable", True))
                    else "failed"
                )
            self.active.pop(key)
            if assigned["gpu_uuid"]:
                self.next_discovery = 0.0
            changed = True
        return changed

    def submit(self, key, host, gpu=None):
        path = self.directory / "jobs" / key
        previous = read(path / "assigned.json", {})
        assigned = {
            "host": host,
            "gpu_uuid": gpu,
            "attempt": previous.get("attempt", 0) + 1,
            "launched_unix": time.time(),
        }
        atomic_json(path / "assigned.json", assigned)
        self.active[key] = assigned
        self.states[key] = "running"
        try:
            launch_remote(host, self.directory, key, gpu, assigned["attempt"])
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            # 派单的应答丢失也可能已启动；保留 reservation，下一轮查实际终态。
            atomic_json(path / "launch_observation.json", {"error": str(error), **assigned})

    def dispatch(self):
        pending = [k for k, value in self.states.items() if value == "pending"]
        priority = {
            "calibration": -1,
            "baseline_gpu": 0,
            "baseline_native": 0,
            "training": 1,
            "validation": 2,
            "test_gpu": 3,
            "test_native": 3,
            "explain": 4,
        }
        pending.sort(key=lambda k: (priority[self.jobs[k]["kind"]], k))
        self.dispatch_native(pending)
        gpu_pending = [k for k in pending if self.jobs[k]["resource"] == "gpu"]
        gpu_busy = {a["gpu_uuid"] for a in self.active.values() if a["gpu_uuid"]}
        if self.gpu_limit is not None and len(gpu_busy) >= self.gpu_limit:
            return
        cpu_pending = any(
            self.jobs[k]["resource"] == "cpu" and self.states[k] == "pending" for k in pending
        )
        if (not gpu_pending and not cpu_pending) or time.monotonic() < self.next_discovery:
            return
        self.next_discovery = time.monotonic() + self.config["gpu_discovery_seconds"]
        try:
            devices = discover_a5000(self.hosts)
        except (subprocess.SubprocessError, OSError, ValueError):
            return
        if self.config["scope"] == "full_three_seed_v2":
            previous = list(self.cpu_hosts)
            for device in devices:
                # 作者初始化条数取实际逻辑 CPU 数，只接纳同为24的主机，每机8线程。
                if (
                    device.get("host_cpus") == self.config["native_initial_routes"]
                    and device["host"] not in self.cpu_hosts
                ):
                    self.cpu_hosts.append(device["host"])
            if previous != self.cpu_hosts:
                atomic_json(self.directory / "native_hosts.json", self.cpu_hosts)
            self.dispatch_native(pending)
        devices = [d for d in devices if d["uuid"] not in gpu_busy]
        if self.gpu_limit is not None:
            devices = devices[: max(0, self.gpu_limit - len(gpu_busy))]
        for key, device in zip(gpu_pending, devices, strict=False):
            self.submit(key, device["host"], device["uuid"])

    def dispatch_native(self, pending):
        busy = {a["host"] for k, a in self.active.items() if self.jobs[k]["resource"] == "cpu"}
        jobs = iter(
            k for k in pending if self.jobs[k]["resource"] == "cpu" and self.states[k] == "pending"
        )
        for host in self.cpu_hosts:
            if host not in busy:
                key = next(jobs, None)
                if key is not None:
                    self.submit(key, host)

    def status(self):
        return {
            "stage": self.stage,
            "updated_unix": time.time(),
            "campaign_started": self.started,
            "jobs": dict(Counter(self.states.values())),
            "by_kind": {
                kind: dict(
                    Counter(self.states[k] for k, j in self.jobs.items() if j["kind"] == kind)
                )
                for kind in sorted({j["kind"] for j in self.jobs.values()})
            },
            "active": {k: {**v, "kind": self.jobs[k]["kind"]} for k, v in self.active.items()},
            "failed": [k for k, state in self.states.items() if state == "failed"],
            "scope": self.config["scope"],
        }

    def run(self):
        from gp_faco.campaign_report import write_report

        try:
            while True:
                changed = self.reconcile()
                try:
                    self.advance()
                except FileNotFoundError as error:
                    # 多主机共享目录可能晚于完成通知可见；等待文件，绝不重新求解。
                    self.stage = "waiting_shared_files"
                    atomic_json(
                        self.directory / "file_visibility.json", {"path": str(error.filename)}
                    )
                if any(v == "failed" for v in self.states.values()):
                    # 其他已提交任务仍可完成；保留结果，不能跳过失败成员写出最终成绩。
                    self.stage = "failed"
                elif self.stage != "complete":
                    self.dispatch()
                status = self.status()
                atomic_json(self.directory / "status.json", status)
                if self.stage == "complete":
                    write_report(self.directory, final=True)
                    print(json.dumps(status, ensure_ascii=False), flush=True)
                    return status
                if (
                    changed
                    or time.monotonic() - self._last_report >= 60
                    or self.stage == "complete"
                ):
                    write_report(self.directory)
                    self._last_report = time.monotonic()
                    print(json.dumps(status, ensure_ascii=False), flush=True)
                if self.stage == "failed" and not self.active:
                    write_report(self.directory)
                    raise RuntimeError("有任务用尽基础设施重试；保留全部结果，修复前停止依赖阶段")
                time.sleep(self.poll_seconds)
        except BaseException as error:
            self.stage = "failed"
            atomic_json(self.directory / "status.json", {**self.status(), "error": str(error)})
            write_report(self.directory)
            raise
        finally:
            atomic_json(
                self.directory / "coordinator.json",
                {**process_receipt(), "campaign_started": self.started, "status": self.stage},
            )
            self.lease.close()
