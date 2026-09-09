"""严格读取原始对称TSPLIB及独立tour；不把显示坐标或COMMENT当作求解输入。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from gp_faco.data import validate_tour
from gp_faco.distance import GEO, TSPLIB_COORDINATE_TYPES, coordinate_distance, geo_radians

EXPLICIT = "tsplib_explicit_int_v1"
FORMATS = (
    "FULL_MATRIX",
    "UPPER_ROW",
    "LOWER_ROW",
    "UPPER_DIAG_ROW",
    "LOWER_DIAG_ROW",
    "UPPER_COL",
    "LOWER_COL",
    "UPPER_DIAG_COL",
    "LOWER_DIAG_COL",
)


class UnsupportedTSPLIB(ValueError):
    """无法保持原问题语义的类型/约束，必须显式列出。"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(token: str) -> int:
    require(bool(re.fullmatch(r"[+-]?\d+", token)), "应为整数，不能截断小数")
    return int(token)


@dataclass(frozen=True)
class OriginalProblem:
    """只含问题定义；reference成本、最优tour及原COMMENT不属于此对象。"""

    name: str
    dimension: int
    distance_spec: str
    coordinates: tuple[tuple[float, float], ...] = ()
    weights: tuple[int, ...] = ()

    def __post_init__(self):
        require(type(self.name) is str and bool(self.name), "缺少实例名")
        require(type(self.dimension) is int and self.dimension >= 3, "TSP至少需要3个节点")
        require(
            isinstance(self.coordinates, tuple) and isinstance(self.weights, tuple),
            "定义必须不可变",
        )
        require(
            all(
                isinstance(p, tuple) and len(p) == 2 and all(math.isfinite(v) for v in p)
                for p in self.coordinates
            ),
            "坐标必须为有限二维数值",
        )
        n = self.dimension
        if self.distance_spec == EXPLICIT:
            require(
                len(self.coordinates) in (0, n) and len(self.weights) == n * n, "显式距离维数错误"
            )
            require(
                all(type(v) is int and 0 <= v <= 2**31 - 1 for v in self.weights),
                "显式边权必须为非负32位整数",
            )
            for a in range(n):
                require(self.weights[a * n + a] == 0, "显式对角必须为0")
                for b in range(a):
                    require(
                        self.weights[a * n + b] == self.weights[b * n + a], "显式矩阵不是对称TSP"
                    )
        else:
            require(self.distance_spec in TSPLIB_COORDINATE_TYPES.values(), "未知原始距离")
            require(len(self.coordinates) == n and not self.weights, "坐标距离定义不完整")
            if self.distance_spec == GEO:
                for latitude, longitude in self.coordinates:
                    require(
                        abs(latitude - math.trunc(latitude)) < 0.6
                        and abs(longitude - math.trunc(longitude)) < 0.6,
                        "GEO分钟字段超出范围",
                    )
                    require(
                        abs(geo_radians(latitude)) <= 3.141592 / 2
                        and abs(geo_radians(longitude)) <= 3.141592,
                        "GEO经纬度超出范围",
                    )

    def distance(self, a: int, b: int) -> int:
        require(
            type(a) is int
            and type(b) is int
            and 0 <= a < self.dimension
            and 0 <= b < self.dimension,
            "节点下标越界",
        )
        if self.distance_spec == EXPLICIT:
            return self.weights[a * self.dimension + b]
        return coordinate_distance(self.coordinates, a, b, self.distance_spec)

    def tour_cost(self, tour) -> int:
        validate_tour(tour, self.dimension)
        return sum(self.distance(tour[i - 1], tour[i]) for i in range(self.dimension))


def _sections(text: str):
    headers, sections, active = {}, {}, None
    eof = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        require(not eof, "EOF之后仍有内容")
        if line in ("EOF", "EOF :", "EOF:"):
            eof = True
            continue
        if re.fullmatch(r"[A-Z_]+_SECTION", line):
            require(line not in sections, "重复数据段")
            sections[line], active = [], line
            continue
        if active is not None:
            sections[active].append(line)
            continue
        match = re.fullmatch(r"([A-Z_]+)\s*:\s*(.*)", line)
        require(match is not None, "无效TSPLIB头部")
        key, value = match.groups()
        if key == "COMMENT":
            # COMMENT可能包含已知最优值，只保留在原始文件，不进入问题对象。
            continue
        require(key not in headers and bool(value), "重复或空头部字段")
        headers[key] = value
    return headers, sections


def _coordinates(lines, n):
    require(len(lines) == n, "坐标行数不符")
    by_id = {}
    for line in lines:
        tokens = line.split()
        require(len(tokens) == 3, "每个节点应含编号及二维坐标")
        node = integer(tokens[0])
        require(1 <= node <= n and node not in by_id, "节点编号重复或越界")
        xy = tuple(float(t) for t in tokens[1:])
        require(all(math.isfinite(v) for v in xy), "坐标非有限")
        by_id[node] = xy
    return tuple(by_id[node] for node in range(1, n + 1))


def _matrix_indices(n, layout):
    if layout == "FULL_MATRIX":
        return ((a, b) for a in range(n) for b in range(n))
    if layout == "UPPER_ROW":
        return ((a, b) for a in range(n) for b in range(a + 1, n))
    if layout == "LOWER_ROW":
        return ((a, b) for a in range(n) for b in range(a))
    if layout == "UPPER_DIAG_ROW":
        return ((a, b) for a in range(n) for b in range(a, n))
    if layout == "LOWER_DIAG_ROW":
        return ((a, b) for a in range(n) for b in range(a + 1))
    if layout == "UPPER_COL":
        return ((a, b) for b in range(n) for a in range(b))
    if layout == "LOWER_COL":
        return ((a, b) for b in range(n) for a in range(b + 1, n))
    if layout == "UPPER_DIAG_COL":
        return ((a, b) for b in range(n) for a in range(b + 1))
    if layout == "LOWER_DIAG_COL":
        return ((a, b) for b in range(n) for a in range(b, n))
    raise UnsupportedTSPLIB("未支持的显式矩阵布局")


def _weights(lines, n, layout):
    require(n * n <= 16_000_000, "显式输入超出当前CPU解析容量")
    values = [integer(t) for line in lines for t in line.split()]
    expected = n * n if layout == "FULL_MATRIX" else n * (n + (1 if "DIAG" in layout else -1)) // 2
    require(len(values) == expected, "显式矩阵边权数量不符")
    matrix = [0] * (n * n)
    for (a, b), value in zip(_matrix_indices(n, layout), values, strict=True):
        matrix[a * n + b] = value
        if layout != "FULL_MATRIX":
            matrix[b * n + a] = value
    return tuple(matrix)


def parse_problem(text: str) -> OriginalProblem:
    headers, sections = _sections(text)
    if headers.get("TYPE") != "TSP":
        raise UnsupportedTSPLIB("只接受原始对称TYPE=TSP")
    require(
        "NAME" in headers and "DIMENSION" in headers and "EDGE_WEIGHT_TYPE" in headers,
        "缺少原始身份、规模或距离类型",
    )
    n = integer(headers["DIMENSION"])
    require(3 <= n <= 100_000, "无效或超过解析容量的DIMENSION")
    allowed_headers = {
        "NAME",
        "TYPE",
        "DIMENSION",
        "EDGE_WEIGHT_TYPE",
        "EDGE_WEIGHT_FORMAT",
        "NODE_COORD_TYPE",
        "DISPLAY_DATA_TYPE",
    }
    if set(headers) - allowed_headers:
        raise UnsupportedTSPLIB("存在未支持的头部约束")
    if set(sections) - {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION"}:
        raise UnsupportedTSPLIB("固定边、稀疏图或其他约束不能被忽略")
    require(
        headers.get("NODE_COORD_TYPE", "TWOD_COORDS") in ("TWOD_COORDS", "NO_COORDS"),
        "只支持二维或无坐标表示",
    )
    if "DISPLAY_DATA_SECTION" in sections:
        _coordinates(sections["DISPLAY_DATA_SECTION"], n)
    coordinates = (
        _coordinates(sections["NODE_COORD_SECTION"], n) if "NODE_COORD_SECTION" in sections else ()
    )
    if headers.get("NODE_COORD_TYPE") == "NO_COORDS":
        require(not coordinates, "NO_COORDS与实际坐标冲突")
    metric, layout = headers["EDGE_WEIGHT_TYPE"], headers.get("EDGE_WEIGHT_FORMAT")
    if metric == "EXPLICIT":
        if layout not in FORMATS:
            raise UnsupportedTSPLIB("EXPLICIT缺少有效矩阵布局")
        require("EDGE_WEIGHT_SECTION" in sections, "缺少显式边权")
        return OriginalProblem(
            headers["NAME"],
            n,
            EXPLICIT,
            coordinates,
            _weights(sections["EDGE_WEIGHT_SECTION"], n, layout),
        )
    if metric not in TSPLIB_COORDINATE_TYPES:
        raise UnsupportedTSPLIB("未支持该原始距离类型")
    require(
        layout in (None, "FUNCTION") and "EDGE_WEIGHT_SECTION" not in sections,
        "坐标距离与显式边权冲突",
    )
    require(bool(coordinates), "缺少原坐标；显示坐标不能替代")
    return OriginalProblem(headers["NAME"], n, TSPLIB_COORDINATE_TYPES[metric], coordinates)


def parse_tours(text: str, dimension: int) -> tuple[tuple[int, ...], ...]:
    headers, sections = _sections(text)
    require(headers.get("TYPE") == "TOUR" and set(sections) == {"TOUR_SECTION"}, "不是独立TOUR文件")
    require("DIMENSION" in headers and integer(headers["DIMENSION"]) == dimension, "TOUR维数不符")
    tours, current, ended = [], [], False
    for token in (v for line in sections["TOUR_SECTION"] for v in line.split()):
        value = integer(token)
        if value == -1:
            if current:
                tour = tuple(v - 1 for v in current)
                validate_tour(tour, dimension)
                tours.append(tour)
                current = []
            else:
                require(bool(tours) and not ended, "TOUR结束标记无效")
                ended = True
        else:
            require(not ended, "TOUR段结束后仍有节点")
            current.append(value)
    require(not current and bool(tours), "TOUR缺少终止标记或为空")
    return tuple(tours)
