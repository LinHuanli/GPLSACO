"""v2 的预先登记、统计选择与 FACO 配对结果。普通编号，不计算完整性摘要。"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from gp_faco.checkpoint import atomic_json
from gp_faco.program_ir import Program
from gp_faco.worker import faco_ants

SCALES = (500, 1000)
EVOLUTION_SEEDS = (1103, 2207, 3313, 4409, 5519)
CURVE_SEEDS = (17, 29, 41, 53, 67)
TEST_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109, 127, 149)
CHECKPOINTS = (0, 100, 250, 500, 1000, 2500, 5000)
GPU_BASELINE = "gpu_faco_mne8_uniform_no_restart"
NATIVE_BASELINE = "faco_2022_continuous_identity_guard"


def representative_program():
    # 固定表达式在查看近似质量结果之前登记；不是按结果挑出的冠军。
    return Program((0, 0, 2), (3, 8, 0), feature_spec_id=2, program_id="representative-01")


def write_once(path: Path, value: dict):
    """已有声明直接比较参数；同目录不能静默换成另一个实验。"""
    normalized = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"已有实验参数不同，请使用新的普通版本目录: {path}")
    else:
        atomic_json(path, normalized)


def predeclare(source, directory: Path) -> dict:
    path = directory / "panels.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(73001)
    training = {n: source.record_ids("training", n) for n in SCALES}
    if any(len(ids) < 16 for ids in training.values()):
        # 旧索引的正式训练角色是 train。
        training = {n: source.record_ids("train", n) for n in SCALES}
    generations = []
    for generation in range(50):
        panels = []
        for n in SCALES:
            ids = sorted(rng.sample(training[n], 16))
            seeds = []
            while len(seeds) < 2:
                seed = rng.getrandbits(64)
                if seed not in seeds:
                    seeds.append(seed)
            panels.append({"dimension": n, "ids": ids, "seeds": seeds})
        generations.append({"generation": generation, "panels": panels})
    value = {
        "experiment_version": 2,
        "panel_seed": 73001,
        "training": generations,
        "validation": [],
        "test": [],
        "monitor": [],
        "quality": [],
        "curves": [],
        "representative_program": representative_program().to_dict(),
        "reference_status": "user_supplied_not_independently_certified",
    }
    for n in SCALES:
        development = source.record_ids("development", n)
        if len(development) < 128:
            raise ValueError("需要每规模 128 个开发实例")
        value["monitor"].append({"dimension": n, "ids": development[64:80], "seeds": [17, 29]})
        value["quality"].append(
            {"dimension": n, "ids": development[64:96], "seeds": list(CURVE_SEEDS)}
        )
        value["curves"].append(
            {"dimension": n, "ids": development[96:128], "seeds": list(CURVE_SEEDS)}
        )
        for role, seeds, count in (("validation", (17, 29), 256), ("test", TEST_SEEDS, 128)):
            ids = source.record_ids(role, n)
            if len(ids) != count:
                raise ValueError(f"{role} 的 TSP{n} 实例数量应为 {count}")
            value[role].append({"dimension": n, "ids": ids, "seeds": list(seeds)})
    write_once(path, value)
    return value


def replicas(panel):
    return [(name, seed) for name in panel["ids"] for seed in panel["seeds"]]


def batches(panel, width=32):
    items = replicas(panel)
    if len(items) % width:
        raise ValueError("预先声明的面板必须能够整分，不补齐或丢弃实例")
    return [items[start : start + width] for start in range(0, len(items), width)]


def paired_quality(exact, approximate, *, max_increase_pp=0.01):
    """先平均实例的所有 seed，再用配对实例差值的单侧 t 上界作非劣效门槛。"""
    from scipy.stats import t

    def collect(rows):
        result = {}
        for row in rows:
            key = (row["dimension"], row["controller"], row["instance_id"], row["seed"])
            if key in result:
                raise ValueError("质量面板含重复成员")
            result[key] = row
        return result

    left, right = collect(exact), collect(approximate)
    if not left or left.keys() != right.keys():
        return {"passed": False, "reason": "missing_or_extra_paired_members", "rows": []}
    groups = defaultdict(lambda: defaultdict(list))
    for key, a in left.items():
        b = right[key]
        if (
            a.get("status") != "completed"
            or b.get("status") != "completed"
            or not all(math.isfinite(v.get("gap_percent", math.inf)) for v in (a, b))
        ):
            return {"passed": False, "reason": "failed_or_nonfinite_member", "rows": []}
        groups[key[:2]][key[2]].append(b["gap_percent"] - a["gap_percent"])
    rows = []
    for (n, controller), instances in sorted(groups.items()):
        values = [statistics.mean(v) for v in instances.values()]
        if len(values) < 2:
            return {"passed": False, "reason": "insufficient_paired_instances", "rows": []}
        mean = statistics.mean(values)
        se = statistics.stdev(values) / math.sqrt(len(values))
        upper = mean + float(t.ppf(0.95, len(values) - 1)) * se
        rows.append(
            {
                "dimension": n,
                "controller": controller,
                "paired_instances": len(values),
                "mean_increase_pp": mean,
                "upper_95_percent_pp": upper,
                "passed": upper <= max_increase_pp,
            }
        )
    return {
        "passed": all(v["passed"] for v in rows),
        "rows": rows,
        "confidence": 0.95,
        "max_increase_pp": max_increase_pp,
        "unit": "instance_after_seed_average",
        "interval": "one_sided_paired_student_t",
    }


def choose_horizon(
    rows, *, controllers, expected_panels, checkpoints=CHECKPOINTS, retained=0.95, correlation=0.90
):
    """只使用预先登记的完整曲线；缺失、失败或无改进时不能自动缩短预算。"""
    from scipy.stats import spearmanr

    expected = {
        (p["dimension"], controller, name, seed, it)
        for p in expected_panels
        for controller in controllers
        for name, seed in replicas(p)
        for it in checkpoints
    }
    found = {}
    for row in rows:
        key = (
            row["dimension"],
            row["controller"],
            row["instance_id"],
            row["seed"],
            row["iterations"],
        )
        if key in found:
            raise ValueError("曲线面板出现重复成员")
        found[key] = row
    if found.keys() != expected:
        raise ValueError("曲线尚未完整；不能用缺失的结果决定训练预算")
    maximum = max(checkpoints)
    if any(
        r.get("status") != "completed" or not math.isfinite(r.get("gap_percent", math.inf))
        for r in rows
    ):
        return {"iterations": maximum, "reason": "failed_member", "rows": []}
    grouped = defaultdict(list)
    for (n, controller, _name, _seed, it), row in found.items():
        grouped[n, controller, it].append(row["gap_percent"])
    means = {key: statistics.mean(values) for key, values in grouped.items()}
    evidence = []
    selected = maximum
    for it in checkpoints:
        if it == 0:
            continue
        row = {"iterations": it, "scales": [], "passed": True}
        for n in sorted({p["dimension"] for p in expected_panels}):
            initial = statistics.mean(means[n, c, 0] for c in controllers)
            final = statistics.mean(means[n, c, maximum] for c in controllers)
            current = statistics.mean(means[n, c, it] for c in controllers)
            improvement = initial - final
            fraction = (initial - current) / improvement if improvement > 0 else None
            last_ranks = [means[n, c, maximum] for c in controllers]
            current_ranks = [means[n, c, it] for c in controllers]
            rho = (
                float(spearmanr(current_ranks, last_ranks).statistic)
                if len(set(current_ranks)) > 1 and len(set(last_ranks)) > 1
                else math.nan
            )
            passed = fraction is not None and fraction >= retained and rho >= correlation
            row["scales"].append(
                {
                    "dimension": n,
                    "initial_mean_gap_percent": initial,
                    "final_mean_gap_percent": final,
                    "mean_gap_percent": current,
                    "retained_improvement": fraction,
                    "spearman": rho if math.isfinite(rho) else None,
                    "passed": passed,
                }
            )
            row["passed"] &= passed
        evidence.append(row)
        if row["passed"] and selected == maximum:
            selected = it
    return {
        "iterations": selected,
        "reason": "development_curve_rule",
        "rows": evidence,
        "required_retained_improvement": retained,
        "required_spearman": correlation,
    }


class BaselineCache:
    """按方法、实例、seed 和评价预算索引原始评分；SQLite 的 B-tree 不计算内容摘要。"""

    def __init__(self, path: Path, manifest: dict, *, readonly=False, immutable=False):
        self.path = path
        self.manifest = json.loads(json.dumps(manifest))
        if readonly:
            self.connection = sqlite3.connect(
                f"file:{path}?mode=ro" + ("&immutable=1" if immutable else ""), uri=True
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS scores (
                    method TEXT, dimension INTEGER, instance_id TEXT, seed TEXT,
                    iterations INTEGER, ants INTEGER, backend TEXT, initialization TEXT,
                    cost REAL, gap REAL, seconds REAL, tour TEXT,
                    PRIMARY KEY(method, dimension, instance_id, seed, iterations, ants,
                                backend, initialization));
            """)
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE name='manifest'"
        ).fetchone()
        if existing:
            if json.loads(existing[0]) != self.manifest:
                raise ValueError("baseline 缓存的距离、求解设置或数据版本不同")
        elif readonly:
            raise ValueError("baseline 缓存缺少实验参数")
        else:
            self.connection.execute(
                "INSERT INTO metadata VALUES ('manifest', ?)", (json.dumps(self.manifest),)
            )
            self.connection.commit()
        self._panels = {}

    def key(self, method, n, name, seed, iterations):
        backend = self.manifest["numeric_backend"] if method == GPU_BASELINE else "exact"
        initialization = (
            self.manifest["gpu_initialization"]
            if method == GPU_BASELINE
            else self.manifest["native_initialization"]
        )
        return method, n, name, str(seed), iterations, faco_ants(n), backend, initialization

    def get(self, method, n, name, seed, iterations):
        row = self.connection.execute(
            "SELECT cost, gap, seconds FROM scores WHERE method=? AND dimension=? AND instance_id=?"
            " AND seed=? AND iterations=? AND ants=? AND backend=? AND initialization=?",
            self.key(method, n, name, seed, iterations),
        ).fetchone()
        return None if row is None else {"cost": row[0], "gap_percent": row[1], "seconds": row[2]}

    def put(self, method, n, name, seed, iterations, *, cost, gap_percent, seconds, tour):
        if (
            not all(math.isfinite(v) for v in (cost, gap_percent, seconds))
            or cost <= 0
            or seconds < 0
        ):
            raise ValueError("不能将失败求解写成有效 baseline")
        self.connection.execute(
            "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *self.key(method, n, name, seed, iterations),
                cost,
                gap_percent,
                seconds,
                json.dumps(tour),
            ),
        )

    def commit(self):
        self.connection.commit()

    def panel(self, method, panel, iterations):
        key = (method, panel["dimension"], tuple(panel["ids"]), tuple(panel["seeds"]), iterations)
        if key not in self._panels:
            result = {}
            for name, seed in replicas(panel):
                row = self.get(method, panel["dimension"], name, seed, iterations)
                if row is None:
                    raise ValueError(
                        f"尚缺配对 baseline: {method} TSP{panel['dimension']} {name} seed={seed}"
                    )
                result[name, seed] = row["gap_percent"]
            self._panels[key] = result
        return self._panels[key]

    def close(self):
        self.connection.close()


def paired_progress(scores, panels, cache: BaselineCache, iterations):
    """胜率先平均同实例的 seed；两个规模等权，保持原有 raw gap fitness。"""
    report = {}
    for method in (NATIVE_BASELINE, GPU_BASELINE):
        by_scale = defaultdict(lambda: defaultdict(list))
        for score, panel in zip(scores, panels, strict=True):
            if score.failed:
                return {"status": "failed", "reason": score.error}
            baseline = cache.panel(method, panel, iterations)
            for name, seed, gap in score.members:
                by_scale[panel["dimension"]][name].append((gap, baseline[name, seed]))
        scale_rows = []
        for n, instances in sorted(by_scale.items()):
            gp = [statistics.mean(v[0] for v in values) for values in instances.values()]
            faco = [statistics.mean(v[1] for v in values) for values in instances.values()]
            scale_rows.append(
                {
                    "dimension": n,
                    "gp_gap_percent": statistics.mean(gp),
                    "faco_gap_percent": statistics.mean(faco),
                    "improvement_pp": statistics.mean(b - a for a, b in zip(gp, faco, strict=True)),
                    "win_rate": statistics.mean(a < b for a, b in zip(gp, faco, strict=True)),
                }
            )
        report[method] = {
            key: statistics.mean(row[key] for row in scale_rows)
            for key in ("gp_gap_percent", "faco_gap_percent", "improvement_pp", "win_rate")
        }
        report[method]["scales"] = scale_rows
    return {"status": "completed", "comparisons": report}
