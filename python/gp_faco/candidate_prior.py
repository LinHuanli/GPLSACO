"""外部LKH只生成候选先验；保留原始先验顺序与原问题FP64距离顺序。"""

from __future__ import annotations

import json
import math
import os
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gp_faco.checkpoint import atomic_json
from gp_faco.data import Instance
from gp_faco.worker import PROJECT, content_hash, coordinate_hash, file_hash


@dataclass(frozen=True)
class PriorSettings:
    kind: str = "ALPHA"
    maximum_candidates: int = 80
    seed: int = 73001
    distance_scale: int = 1000000
    popmusic_solutions: int = 50
    popmusic_sample_size: int = 10
    popmusic_max_neighbors: int = 5
    popmusic_trials: int = 1

    def __post_init__(self):
        if self.kind not in ("ALPHA", "POPMUSIC"):
            raise ValueError("先验只支持ALPHA或POPMUSIC")
        for name, value in asdict(self).items():
            if name == "kind":
                continue
            if type(value) is not int or not 1 <= value <= 0x7FFFFFFF:
                raise ValueError("先验种子、整数缩放及工作量参数必须为正int32")
        if self.maximum_candidates > 9999:
            raise ValueError("先验候选数量超出10K支持范围")


def parse_candidates(problem: Instance, settings: PriorSettings, raw: dict) -> dict:
    """核验1-based节点、先验成本/Pi关系及整数距离，再生成独立FP64 LS视图。"""
    n = problem.dimension
    required = {
        "candidate_export_version",
        "dimension",
        "kind",
        "scale",
        "precision",
        "seed",
        "maximum_candidates",
        "native_preparation_cpu_seconds",
        "nodes",
    }
    if set(raw) != required:
        raise ValueError("未知原生候选输出字段")
    for key, expected in (
        ("candidate_export_version", 1),
        ("dimension", n),
        ("kind", settings.kind),
        ("scale", settings.distance_scale),
        ("precision", 1),
        ("seed", settings.seed),
        ("maximum_candidates", settings.maximum_candidates),
    ):
        if type(raw[key]) is not type(expected) or raw[key] != expected:
            raise ValueError(f"原生候选身份不符: {key}")
    elapsed = raw["native_preparation_cpu_seconds"]
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("原生候选资源记录无效")
    nodes = raw["nodes"]
    if type(nodes) is not list or len(nodes) != n:
        raise ValueError("候选未覆盖全部节点")
    for i, node in enumerate(nodes):
        if (
            set(node) != {"id", "pi", "dad", "count", "edges"}
            or any(type(node[k]) is not int for k in ("id", "pi", "dad", "count"))
            or node["id"] != i + 1
            or not 0 <= node["dad"] <= n
        ):
            raise ValueError("节点编号/Pi/MST父节点身份不符")
        if not 0 <= node["count"] < n or len(node["edges"]) != node["count"]:
            raise ValueError("候选数量声明不符")
    rows, distance_rows, degrees, undirected = [], [], [], set()
    max_quantization_error = 0.0
    for i, node in enumerate(nodes):
        seen, row = set(), []
        for edge in node["edges"]:
            if type(edge) is not list or len(edge) != 4 or any(type(v) is not int for v in edge):
                raise ValueError("候选边必须是四项整数字段")
            destination, alpha, transformed, quantized = edge
            j = destination - 1
            if not 0 <= j < n or j == i or j in seen:
                raise ValueError("候选含未知节点、自环或重复")
            seen.add(j)
            dx = problem.coordinates[i][0] - problem.coordinates[j][0]
            dy = problem.coordinates[i][1] - problem.coordinates[j][1]
            distance = math.sqrt(dx * dx + dy * dy)
            if quantized != math.floor(settings.distance_scale * distance + 0.5):
                raise ValueError("LKH整数先验距离与原问题坐标不符")
            if transformed != quantized + node["pi"] + nodes[j]["pi"]:
                raise ValueError("原生先验成本与距离/Pi关系不符")
            max_quantization_error = max(
                max_quantization_error, abs(quantized / settings.distance_scale - distance)
            )
            row.append(
                {
                    "to": j,
                    "alpha": alpha,
                    "transformed_cost": transformed,
                    "quantized_distance": quantized,
                    "distance": distance,
                }
            )
            undirected.add((min(i, j), max(i, j)))
        rows.append(row)  # 保留LKH实际优先级顺序；不依照标注tour改成员。
        distance_rows.append([e["to"] for e in sorted(row, key=lambda e: (e["distance"], e["to"]))])
        degrees.append(len(row))
    return {
        "prior_spec_id": 1,
        "dimension": n,
        "coordinate_sha256": coordinate_hash(problem),
        "settings": asdict(settings),
        "rows": rows,
        "distance_rows": distance_rows,
        "native_nodes_sha256": content_hash(nodes),
        "node_pi": [v["pi"] for v in nodes],
        "directed_slots": sum(degrees),
        "undirected_edges": len(undirected),
        "degree_min": min(degrees),
        "degree_max": max(degrees),
        "max_distance_quantization_error": max_quantization_error,
        "distance_role": "LKH integer approximation selects members; FP64 source distance for LS",
        "initial_tour_exported": False,
    }


def prepare_candidates(problem: Instance, settings: PriorSettings, binary: Path, directory: Path):
    if type(problem) is not Instance or type(settings) is not PriorSettings:
        raise TypeError("候选准备只接受无标签Instance与明确设置")
    n = problem.dimension
    if not 3 <= n <= 10000 or settings.maximum_candidates >= n:
        raise ValueError("规模/候选宽度越界")
    # 对称归一化之外的数据也只按原坐标传入；超出安全int范围时明确拒绝。
    xs, ys = zip(*problem.coordinates, strict=True)
    upper = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) * settings.distance_scale
    if not math.isfinite(upper) or upper > 10000000:
        raise ValueError("量化边长超出保守int32先验安全范围，须显式另订距离规格")
    binary, directory = binary.resolve(), directory.resolve()
    if not binary.is_relative_to(PROJECT) or not directory.is_relative_to(PROJECT):
        raise ValueError("候选工具及产物必须在项目内")
    build = json.loads((binary.parent / "build-manifest.json").read_text())
    if build["exit_code"] != 0 or file_hash(binary) != build["binary_sha256"]:
        raise ValueError("LKH适配器与实际构建身份不符")
    directory.mkdir(parents=True, exist_ok=False)
    lines = [
        "NAME: gpfaco_prior",
        "TYPE: TSP",
        f"DIMENSION: {n}",
        "EDGE_WEIGHT_TYPE: EUC_2D",
        "NODE_COORD_SECTION",
    ]
    lines.extend(f"{i + 1} {x:.17g} {y:.17g}" for i, (x, y) in enumerate(problem.coordinates))
    (directory / "problem.tsp").write_text("\n".join([*lines, "EOF", ""]))
    params = [
        "PROBLEM_FILE = problem.tsp",
        f"CANDIDATE_SET_TYPE = {settings.kind}",
        f"MAX_CANDIDATES = {settings.maximum_candidates}",
        f"ASCENT_CANDIDATES = {max(50, settings.maximum_candidates)}",
        f"SEED = {settings.seed}",
        f"SCALE = {settings.distance_scale}",
        "PRECISION = 1",
        "EXCESS = 1",
        "MAX_TRIALS = 1",
        "RUNS = 1",
        "INITIAL_TOUR_ALGORITHM = WALK",
        "POPMUSIC_INITIAL_TOUR = NO",
        f"POPMUSIC_SOLUTIONS = {settings.popmusic_solutions}",
        f"POPMUSIC_SAMPLE_SIZE = {settings.popmusic_sample_size}",
        f"POPMUSIC_MAX_NEIGHBORS = {settings.popmusic_max_neighbors}",
        f"POPMUSIC_TRIALS = {settings.popmusic_trials}",
        "TRACE_LEVEL = 1",
    ]
    (directory / "parameters.par").write_text("\n".join([*params, ""]))
    manifest = {
        "instance_id": problem.instance_id,
        "coordinate_sha256": coordinate_hash(problem),
        "dimension": n,
        "settings": asdict(settings),
        "binary_sha256": file_hash(binary),
        "build_manifest_sha256": file_hash(binary.parent / "build-manifest.json"),
        "module_sha256": file_hash(Path(__file__)),
        "wall_clock_limit": None,
        "problem_file_sha256": file_hash(directory / "problem.tsp"),
        "parameter_file_sha256": file_hash(directory / "parameters.par"),
    }
    atomic_json(directory / "manifest.json", manifest)
    (PROJECT / ".tmp").mkdir(exist_ok=True)
    before, started = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    with (directory / "native.log").open("x") as stream:
        outcome = subprocess.run(
            [str(binary), "parameters.par", "native-candidates.json"],
            cwd=directory,
            env={**os.environ, "TMPDIR": str(PROJECT / ".tmp")},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    resources = {
        "exit_code": outcome.returncode,
        "wall_seconds": time.perf_counter() - started,
        "cpu_user_seconds": after.ru_utime - before.ru_utime,
        "cpu_system_seconds": after.ru_stime - before.ru_stime,
        "max_rss_children_kib": after.ru_maxrss,
        "rss_note": "maximum over child processes of this coordinator, not additive",
        "log_sha256": file_hash(directory / "native.log"),
    }
    atomic_json(directory / "resources.json", resources)
    if outcome.returncode != 0:
        raise RuntimeError("候选原生进程失败；保留原始日志，不根据质量重试")
    raw = json.loads((directory / "native-candidates.json").read_text())
    prior = parse_candidates(problem, settings, raw)
    prior["manifest_sha256"] = content_hash(manifest)
    prior["native_output_sha256"] = file_hash(directory / "native-candidates.json")
    atomic_json(directory / "prior.json", prior)
    return prior
