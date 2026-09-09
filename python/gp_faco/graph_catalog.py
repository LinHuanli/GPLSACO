"""冻结匹配图目录；在线输入不含标签，图资料只在首次注册时读取。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from gp_faco.data import Instance, validate_tour


@dataclass(frozen=True)
class GraphEntry:
    instance_id: str
    dimension: int
    coordinate_sha256: str
    path: str
    file_sha256: str
    # 固定顺序ALPHA、POPMUSIC，内部使用不可变元组，避免修改调用方字典。
    graphs: tuple[tuple[str, str, int], ...]


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def project_file(root: Path, relative: str) -> Path:
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError("图目录只接受工作树内的相对文件路径")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("图文件越过GPLSACO工作树")
    return path


class GraphCatalog:
    """协议持有完整文件身份；摘要进入run/task，原始图不经每次任务重复传输。"""

    def __init__(self, root: Path, relative: str, expected_sha256: str):
        # 避免worker/graph_matching在模块导入阶段互相初始化。
        from gp_faco.graph_matching import GraphSettings
        from gp_faco.worker import file_hash

        self.root = root.resolve()
        path = project_file(self.root, relative)
        payload = path.read_bytes()
        if not _digest(expected_sha256) or hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("图目录文件与冻结SHA256不符")
        value = json.loads(payload)
        fields = {
            "graph_catalog_version",
            "graph_spec_id",
            "matching_spec_id",
            "settings",
            "entries",
            "source_manifest",
            "source_manifest_sha256",
        }
        if (
            type(value) is not dict
            or set(value) != fields
            or any(
                type(value[name]) is not int or value[name] != expected
                for name, expected in (
                    ("graph_catalog_version", 1),
                    ("graph_spec_id", 1),
                    ("matching_spec_id", 2),
                )
            )
        ):
            raise ValueError("图目录字段或格式/匹配版本不符")
        source = project_file(self.root, value["source_manifest"])
        if not _digest(value["source_manifest_sha256"]) or (
            file_hash(source) != value["source_manifest_sha256"]
        ):
            raise ValueError("图目录来源manifest身份不符")
        self.settings = GraphSettings(**value["settings"])
        self.relative, self.sha256 = relative, expected_sha256
        if type(value["entries"]) is not list or not value["entries"]:
            raise ValueError("图目录必须含非空实例表")
        entries = {}
        coordinates = set()
        for row in value["entries"]:
            if type(row) is not dict or set(row) != set(GraphEntry.__dataclass_fields__):
                raise ValueError("图目录实例字段不符")
            n = row["dimension"]
            if (
                type(row["instance_id"]) is not str
                or not row["instance_id"]
                or row["instance_id"] in entries
                or type(n) is not int
                or not 3 <= n <= 10000
                or not _digest(row["coordinate_sha256"])
                or row["coordinate_sha256"] in coordinates
                or not _digest(row["file_sha256"])
                or type(row["graphs"]) is not dict
                or set(row["graphs"]) != {"ALPHA", "POPMUSIC"}
            ):
                raise ValueError("图目录实例身份重复、遗漏或无效")
            project_file(self.root, row["path"])
            graphs = []
            for kind in ("ALPHA", "POPMUSIC"):
                graph = row["graphs"][kind]
                if (
                    type(graph) is not dict
                    or set(graph) != {"sha256", "edges"}
                    or (
                        not _digest(graph["sha256"])
                        or type(graph["edges"]) is not int
                        or not n <= graph["edges"] <= n * (n - 1) // 2
                    )
                ):
                    raise ValueError("先验图摘要或边数无效")
                graphs.append((kind, graph["sha256"], graph["edges"]))
            if graphs[0][2] != graphs[1][2]:
                raise ValueError("两个先验的实际E0边数未匹配")
            entry = GraphEntry(**{**row, "graphs": tuple(graphs)})
            entries[entry.instance_id] = entry
            coordinates.add(entry.coordinate_sha256)
        self._entries = entries
        self._feasibility = {}

    def require_settings(self, settings) -> None:
        if any(
            getattr(settings, name) != getattr(self.settings, name)
            for name in ("primary_width", "backup_width", "ls_width")
        ):
            raise ValueError("worker候选形状与冻结图目录不符")

    def entry(self, problem: Instance) -> GraphEntry:
        from gp_faco.worker import coordinate_hash

        entry = self._entries.get(problem.instance_id)
        if (
            entry is None
            or entry.dimension != problem.dimension
            or (entry.coordinate_sha256 != coordinate_hash(problem))
        ):
            raise ValueError("任务实例不在冻结图目录中或坐标改变")
        return entry

    def describe(self, problems, kind: str) -> list[dict]:
        if kind not in ("ALPHA", "POPMUSIC"):
            raise ValueError("需要固定ALPHA或POPMUSIC先验")
        descriptions = []
        for problem in problems:
            entry = self.entry(problem)
            _, digest, edges = entry.graphs[0 if kind == "ALPHA" else 1]
            descriptions.append(
                {
                    "instance_id": entry.instance_id,
                    "coordinate_sha256": entry.coordinate_sha256,
                    "prior_kind": kind,
                    "graph_sha256": digest,
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
        if hashlib.sha256(data).hexdigest() != entry.file_sha256:
            raise ValueError("匹配图缓存文件被修改")
        graph = json.loads(data)["graphs"][kind]
        if (
            graph["sha256"] != description["graph_sha256"]
            or graph["coordinate_sha256"] != entry.coordinate_sha256
            or type(graph["dimension"]) is not int
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
