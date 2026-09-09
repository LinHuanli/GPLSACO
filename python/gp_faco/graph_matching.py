"""不读标签地匹配先验实际图边数；完整合法图与固定枚举/距离视图分离。"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from gp_faco.data import Instance, validate_tour


def engine_graph_spec(graph: dict) -> dict:
    """只向在线 Engine 交付六个无标签图字段；实际节点检查在首次注册完成。"""
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
    n = problem.dimension
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
            or prior.get("instance_id", problem.instance_id) != problem.instance_id
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
    edge_sets, neighbor_sets = {}, {}
    for kind in priors:
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
        edge_sets[kind], neighbor_sets[kind] = edges, neighbors
    # 完整合法图按总边数匹配；物理行内的实际访问量再按同一节点的双先验较小值匹配。
    # 不能把哨兵当作有效候选，也不能让较长的原生尾部增加Escape的访问机会。
    primary_quota = [
        min(width, *(len(rows[i]) for rows in neighbor_sets.values())) for i in range(n)
    ]
    ls_quota = [min(ls_width, *(len(rows[i]) for rows in neighbor_sets.values())) for i in range(n)]
    main_rows = {
        kind: [
            sorted(row, key=lambda j, i=i, rank=ranks[kind]: (rank[i].get(j, n), j))[
                : primary_quota[i]
            ]
            for i, row in enumerate(neighbors)
        ]
        for kind, neighbors in neighbor_sets.items()
    }
    tail_rows = {
        kind: [
            [e["to"] for e in prior["rows"][i][width:] if e["to"] not in {i, *main_rows[kind][i]}]
            for i in range(n)
        ]
        for kind, prior in priors.items()
    }
    uniforms = min(settings.uniform_backup_slots, backup_width)
    tail_quota = [
        min(backup_width - uniforms, *(len(rows[i]) for rows in tail_rows.values()))
        for i in range(n)
    ]
    output = {}
    for kind in priors:
        edges, neighbors = edge_sets[kind], neighbor_sets[kind]
        primary, backup, ls = [], [], []
        for i, row in enumerate(neighbors):
            chosen = main_rows[kind][i]
            primary.append(chosen + [n] * (width - len(chosen)))

            def distance(j, i=i):
                dx = problem.coordinates[i][0] - problem.coordinates[j][0]
                dy = problem.coordinates[i][1] - problem.coordinates[j][1]
                return math.sqrt(dx * dx + dy * dy), j

            ordered = sorted(row, key=distance)[: ls_quota[i]]
            ls.append(ordered + [n] * (ls_width - len(ordered)))
            # 只取原生前width之后的尾部；主行和自身不能重复成为备用。
            forbidden = {i, *chosen}
            tail = tail_rows[kind][i][: tail_quota[i]]
            forbidden.update(tail)
            # 使用与ACO隔离的Python MT19937准备流；版本和全部输出入图身份。
            seed = (settings.preparation_seed << 96) | (problem.numeric_id << 32) | i
            pool = [j for j in range(n) if j not in forbidden]
            # 固定备用宽度不超过n-1-width，故两个先验均有足够的不重复均匀候选。
            sampled = random.Random(seed).sample(pool, uniforms)
            values = tail + sampled
            backup.append(values + [n] * (backup_width - len(values)))
        spec = {
            "graph_spec_id": 1,
            "matching_spec_id": 2,
            "dimension": n,
            "instance_id": problem.instance_id,
            "common_initial_tour": list(initial_tour),
            "edges": [list(e) for e in sorted(edges)],
            "primary": primary,
            "backup": backup,
            "ls": ls,
            "settings": asdict(settings),
            "prior_kind": kind,
            "prior_id": f"{problem.instance_id}-{kind}",
            "extra_edge_budget": target,
            "discarded_extra_edges": len(extras[kind]) - target,
            "degrees": [len(v) for v in neighbors],
            "matched_actual_slots": {
                "primary": list(primary_quota),
                "ls": list(ls_quota),
                "native_backup": list(tail_quota),
                "uniform_backup": [uniforms] * n,
            },
            "matching_policy": "paired-node minimum actual slots; full E0 edges retained",
            "preparation_random_stream": (
                "Python Random/MT19937; explicit preparation seed, instance ordinal, node"
            ),
        }
        output[kind] = {**spec, "graph_id": f"{problem.instance_id}-{kind}"}
    return output
