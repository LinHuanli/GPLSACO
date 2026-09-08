"""不读标签地匹配先验实际图边数；完整合法图与固定枚举/距离视图分离。"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from gp_faco.data import Instance, validate_tour
from gp_faco.worker import content_hash, coordinate_hash


def engine_graph_spec(graph: dict) -> dict:
    """核验缓存身份后，只向在线Engine交付六个无标签图字段。"""
    if graph.get("sha256") != content_hash({k: v for k, v in graph.items() if k != "sha256"}):
        raise ValueError("图缓存内容与冻结身份不符")
    return {
        field: graph[field]
        for field in ("graph_spec_id", "common_initial_tour", "edges", "primary", "backup", "ls")
    }


@dataclass(frozen=True)
class GraphSettings:
    primary_width: int = 16
    backup_width: int = 64
    ls_width: int = 20
    uniform_backup_slots: int = 16
    preparation_seed: int = 80471

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()) or (
            self.primary_width < 2
            or self.ls_width < 1
            or self.uniform_backup_slots > self.backup_width
        ):
            raise ValueError("图枚举宽度、均匀槽位或准备seed无效")


def match_graphs(problem: Instance, initial_tour: tuple[int, ...], priors: dict, settings=None):
    """先完成两个先验的固定匹配，再返回四条件可共享的两份图资料。"""
    if settings is None:
        settings = GraphSettings()
    if type(problem) is not Instance or type(settings) is not GraphSettings:
        raise TypeError("需要无标签Instance与显式图设置")
    n, fingerprint = problem.dimension, coordinate_hash(problem)
    validate_tour(initial_tour, n)
    if set(priors) != {"ALPHA", "POPMUSIC"}:
        raise ValueError("图匹配必须同时提供两个已核验先验")
    width = min(settings.primary_width, n - 1)
    backup_width = min(settings.backup_width, n - 1 - width)
    ls_width = min(settings.ls_width, n - 1)
    initial_edges = {tuple(sorted((initial_tour[i - 1], initial_tour[i]))) for i in range(n)}
    ranks, extras = {}, {}
    for kind, prior in priors.items():
        if (
            type(prior.get("prior_spec_id")) is not int
            or prior["prior_spec_id"] != 1
            or type(prior["dimension"]) is not int
            or prior["dimension"] != n
            or prior["coordinate_sha256"] != fingerprint
            or (prior["settings"]["kind"] != kind or len(prior["rows"]) != n)
        ):
            raise ValueError("先验与坐标/方法身份不符")
        ranks[kind], candidates = [], set()
        for i, row in enumerate(prior["rows"]):
            ids = [e["to"] for e in row]
            if any(type(j) is not int or not 0 <= j < n or j == i for j in ids) or (
                len(ids) != len(set(ids))
            ):
                raise ValueError("先验有未知节点、自环或重复")
            ranks[kind].append({j: rank for rank, j in enumerate(ids)})
            candidates.update(tuple(sorted((i, j))) for j in ids[:width])
        extras[kind] = candidates - initial_edges
    target = min(len(v) for v in extras.values())
    output = {}
    for kind, prior in priors.items():
        rank = ranks[kind]

        def priority(edge, rank=rank):
            a, b = edge
            left, right = rank[a].get(b, n), rank[b].get(a, n)
            return min(left, right), max(left, right), a, b

        edges = initial_edges | set(sorted(extras[kind], key=priority)[:target])
        neighbors = [[] for _ in range(n)]
        for a, b in sorted(edges):
            neighbors[a].append(b)
            neighbors[b].append(a)
        primary, backup, ls = [], [], []
        for i, row in enumerate(neighbors):
            chosen = sorted(row, key=lambda j: (rank[i].get(j, n), j))[:width]
            primary.append(chosen + [n] * (width - len(chosen)))

            def distance(j, i=i):
                dx = problem.coordinates[i][0] - problem.coordinates[j][0]
                dy = problem.coordinates[i][1] - problem.coordinates[j][1]
                return math.sqrt(dx * dx + dy * dy), j

            ordered = sorted(row, key=distance)[:ls_width]
            ls.append(ordered + [n] * (ls_width - len(ordered)))
            uniforms = min(settings.uniform_backup_slots, backup_width)
            tail_limit = backup_width - uniforms
            # 只取原生前width之后的尾部；主行和自身不能重复成为备用。
            forbidden = {i, *chosen}
            tail = [e["to"] for e in prior["rows"][i][width:] if e["to"] not in forbidden][
                :tail_limit
            ]
            forbidden.update(tail)
            # 使用与ACO隔离的Python MT19937准备流；版本和全部输出入图身份。
            seed = int(
                content_hash(
                    {"coordinates": fingerprint, "seed": settings.preparation_seed, "node": i}
                )[:16],
                16,
            )
            pool = [j for j in range(n) if j not in forbidden]
            sampled = random.Random(seed).sample(pool, min(uniforms, len(pool)))
            values = tail + sampled
            backup.append(values + [n] * (backup_width - len(values)))
        spec = {
            "graph_spec_id": 1,
            "dimension": n,
            "coordinate_sha256": fingerprint,
            "common_initial_tour": list(initial_tour),
            "edges": [list(e) for e in sorted(edges)],
            "primary": primary,
            "backup": backup,
            "ls": ls,
            "settings": asdict(settings),
            "prior_kind": kind,
            "prior_sha256": content_hash(prior),
            "extra_edge_budget": target,
            "discarded_extra_edges": len(extras[kind]) - target,
            "degrees": [len(v) for v in neighbors],
            "preparation_random_stream": "Python Random/MT19937; coordinate/node/seed SHA domain",
        }
        output[kind] = {**spec, "sha256": content_hash(spec)}
    return output
