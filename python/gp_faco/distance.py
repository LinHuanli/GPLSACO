"""原目标距离定义；只读取坐标和类型，不读取解或最优值。"""

from __future__ import annotations

import math

CONTINUOUS = "continuous_euclidean_fp64"
EUC_2D = "tsplib_euc_2d_v1"
CEIL_2D = "tsplib_ceil_2d_v1"
ATT = "tsplib_att_v1"
GEO = "tsplib_geo_faq_v1"
COORDINATE_SPECS = (CONTINUOUS, EUC_2D, CEIL_2D, ATT, GEO)
TSPLIB_COORDINATE_TYPES = {"EUC_2D": EUC_2D, "CEIL_2D": CEIL_2D, "ATT": ATT, "GEO": GEO}


def geo_radians(value: float) -> float:
    # 官方FAQ纠正旧PDF：度数向零截断；PI常数也保留官方数值。
    degrees = math.trunc(value)
    minutes = value - degrees
    return 3.141592 * (degrees + 5.0 * minutes / 3.0) / 180.0


def coordinate_distance(coordinates, a: int, b: int, spec: str) -> float | int:
    if spec not in COORDINATE_SPECS:
        raise ValueError("不支持的坐标距离类型")
    if a == b:
        return 0.0 if spec == CONTINUOUS else 0
    if spec == CONTINUOUS:
        return math.dist(coordinates[a], coordinates[b])
    if spec == GEO:
        lat_a, lon_a = map(geo_radians, coordinates[a])
        lat_b, lon_b = map(geo_radians, coordinates[b])
        q1, q2, q3 = math.cos(lon_a - lon_b), math.cos(lat_a - lat_b), math.cos(lat_a + lat_b)
        cosine = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
        if not math.isfinite(cosine) or abs(cosine) > 1.0 + 8 * math.ulp(1.0):
            raise ValueError("GEO反余弦输入无效")
        # 只修正浮点舍入越界，不更换球面距离公式。
        value = 6378.388 * math.acos(max(-1.0, min(1.0, cosine))) + 1.0
        return math.floor(value)
    dx = coordinates[a][0] - coordinates[b][0]
    dy = coordinates[a][1] - coordinates[b][1]
    square = dx * dx + dy * dy
    value = math.sqrt(square / 10.0 if spec == ATT else square)
    if not math.isfinite(value):
        raise ValueError("原始坐标距离溢出")
    if spec == CEIL_2D:
        result = math.ceil(value)
    else:
        result = math.floor(value + 0.5)
        if spec == ATT and result < value:
            result += 1
    if result > 2**31 - 1:
        raise ValueError("TSPLIB边权超出32位整数范围")
    return result
