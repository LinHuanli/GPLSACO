"""冻结匹配图目录；在线输入不含标签，图资料只在首次注册时读取。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gp_faco.data import Instance, validate_tour


@dataclass(frozen=True)
class GraphEntry:
    instance_id: str
    dimension: int
    path: str
    # 固定顺序ALPHA、POPMUSIC，内部使用不可变元组，避免修改调用方字典。
    graphs: tuple[tuple[str, str, int], ...]


def project_file(root: Path, relative: str) -> Path:
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError("图目录只接受工作树内的相对文件路径")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("图文件越过GPLSACO工作树")
    return path


class GraphCatalog:
    """图只在实例首次注册时读取，历史附加元数据直接忽略。"""

    def __init__(self, root: Path, relative: str):
        from gp_faco.graph_matching import GraphSettings

        self.root = root.resolve()
        value = json.loads(project_file(self.root, relative).read_text())
        if value["graph_spec_id"] != 1 or value["matching_spec_id"] != 2:
            raise ValueError("未知候选图格式")
        self.settings = GraphSettings(**value["settings"])
        self.relative = relative
        self._entries = {}
        for row in value["entries"]:
            name, n = row["instance_id"], row["dimension"]
            if name in self._entries or type(n) is not int or n < 3:
                raise ValueError("重复实例编号或无效规模")
            graphs = tuple(
                (kind, f"{name}-{kind}", row["graphs"][kind]["edges"])
                for kind in ("ALPHA", "POPMUSIC")
            )
            if graphs[0][2] != graphs[1][2]:
                raise ValueError("两个先验的实际边数未匹配")
            self._entries[name] = GraphEntry(name, n, row["path"], graphs)
        self._feasibility = {}

    def require_settings(self, settings) -> None:
        if any(
            getattr(settings, name) != getattr(self.settings, name)
            for name in ("primary_width", "backup_width", "ls_width")
        ):
            raise ValueError("worker候选形状与冻结图目录不符")

    def entry(self, problem: Instance) -> GraphEntry:
        entry = self._entries.get(problem.instance_id)
        if entry is None or entry.dimension != problem.dimension:
            raise ValueError("任务实例不在冻结图目录中或坐标改变")
        return entry

    def describe(self, problems, kind: str) -> list[dict]:
        if kind not in ("ALPHA", "POPMUSIC"):
            raise ValueError("需要固定ALPHA或POPMUSIC先验")
        descriptions = []
        for problem in problems:
            entry = self.entry(problem)
            _, graph_id, edges = entry.graphs[0 if kind == "ALPHA" else 1]
            descriptions.append(
                {
                    "instance_id": entry.instance_id,
                    "prior_kind": kind,
                    "graph_id": graph_id,
                    "edges": edges,
                }
            )
        return descriptions

    def load_graph(self, problem: Instance, kind: str) -> dict:
        from gp_faco.graph_matching import engine_graph_spec

        description = self.describe((problem,), kind)[0]
        entry = self.entry(problem)
        path = project_file(self.root, entry.path)
        data = path.read_bytes()
        graph = json.loads(data)["graphs"][kind]
        if (
            type(graph["dimension"]) is not int
            or graph["dimension"] != entry.dimension
            or graph["prior_kind"] != kind
            or type(graph["matching_spec_id"]) is not int
            or graph["matching_spec_id"] != 2
            or type(graph["graph_spec_id"]) is not int
            or graph["graph_spec_id"] != 1
            or graph["settings"] != asdict(self.settings)
            or len(graph["edges"]) != description["edges"]
        ):
            raise ValueError("匹配图内容与目录的实例/先验/格式身份不符")
        payload = engine_graph_spec(graph)
        validate_tour(payload["common_initial_tour"], problem.dimension)
        return payload

    def feasible_reference(self, problem: Instance, kind: str):
        """协调端一次核验后缓存不可变E0/共同初解，避免每个个体重复读磁盘。"""
        self.entry(problem)
        key = problem.instance_id, kind
        if key not in self._feasibility:
            graph = self.load_graph(problem, kind)
            self._feasibility[key] = (
                frozenset(map(tuple, graph["edges"])),
                tuple(graph["common_initial_tour"]),
            )
        return self._feasibility[key]
