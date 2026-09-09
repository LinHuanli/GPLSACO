"""数据读取与独立距离核验；标签不会进入 Instance 对象。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Instance:
    instance_id: str
    coordinates: tuple[tuple[float, float], ...]
    distance_spec: str = "continuous_euclidean_fp64"
    numeric_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", tuple(tuple(p) for p in self.coordinates))
        if len(self.coordinates) < 3:
            raise ValueError("TSP 至少需要三个节点")
        if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in self.coordinates):
            raise ValueError("坐标必须是有限二维数值")
        if self.distance_spec != "continuous_euclidean_fp64":
            raise ValueError("此读取器只处理连续欧氏目标，原始 TSPLIB 需独立适配")

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    def distance(self, i: int, j: int) -> float:
        return math.dist(self.coordinates[i], self.coordinates[j])


@dataclass(frozen=True)
class Label:
    tour: tuple[int, ...]
    cost: float
    declared_status: str = "user_supplied_optimal"
    certificate_status: str = "not_independently_verified"


def validate_tour(tour: tuple[int, ...] | list[int], n: int) -> None:
    """排列与闭环表示分开：内部 tour 不重复存储首节点。"""
    if len(tour) != n or any(type(v) is not int for v in tour) or set(tour) != set(range(n)):
        raise ValueError("tour 必须是 0..n-1 的完整排列")


def tour_cost(instance: Instance, tour: tuple[int, ...] | list[int]) -> float:
    validate_tour(tour, instance.dimension)
    return math.fsum(instance.distance(tour[i - 1], tour[i]) for i in range(len(tour)))


def parse_record(line: str, instance_id: str) -> tuple[Instance, Label]:
    """严格解析坐标＋output＋1-based tour，不推断最优性证明。"""
    parts = line.split("output")
    if len(parts) != 2:
        raise ValueError("每条记录必须恰有一个 output 分隔符")
    values = [float(v) for v in parts[0].split()]
    if len(values) % 2:
        raise ValueError("坐标数量必须为偶数")
    instance = Instance(instance_id, tuple(zip(values[::2], values[1::2], strict=True)))
    nodes = [int(v) - 1 for v in parts[1].split()]
    if len(nodes) == instance.dimension + 1:
        if nodes[0] != nodes[-1]:
            raise ValueError("tour 闭合端点不一致")
        nodes.pop()
    tour = tuple(nodes)
    cost = tour_cost(instance, tour)
    if cost <= 0:
        raise ValueError("标签成本必须为正，零长度实例需单独处理")
    return instance, Label(tour, cost)


def read_record(path: Path, row: int = 0) -> tuple[Instance, Label]:
    if row < 0:
        raise ValueError("行号不能为负")
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == row:
                return parse_record(line, f"{path.name}:{row}")
    raise IndexError(f"文件中不存在行 {row}: {path}")


def write_explicit_tsplib(instance: Instance, path: Path) -> None:
    """只用于小规模原生 CPU 核验，17 位数字保留 FP64 边权往返。"""
    if instance.dimension > 1000:
        raise ValueError("诊断矩阵导出限 n<=1000，不能用作 10K 求解器表示")
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "NAME : gpfaco_diagnostic\nTYPE : TSP\n"
            f"DIMENSION : {instance.dimension}\nEDGE_WEIGHT_TYPE : EXPLICIT\n"
            "EDGE_WEIGHT_FORMAT : UPPER_DIAG_ROW\nEDGE_WEIGHT_SECTION\n"
        )
        for i in range(instance.dimension):
            stream.write(
                " ".join(
                    format(instance.distance(i, j), ".17g") for j in range(i, instance.dimension)
                )
                + "\n"
            )
        stream.write("EOF\n")
