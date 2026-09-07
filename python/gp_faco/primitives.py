"""有界 FP32 grammar；仅用于离线树生成和数值诊断。"""

import random

import numpy as np
from deap import gp

FEATURE_NAMES = (
    "elapsed",
    "stagnation",
    "return_rate",
    "ls_work",
    "restart",
    "mne_level",
    "ref_gap",
    "ref_diff",
    "region_excess",
    "archive_disagreement",
    "pheromone_strength",
    "region_dispersion",
)
FEATURE_IDS = {name: index for index, name in enumerate(FEATURE_NAMES)}


def clip(value):
    return np.clip(np.asarray(value, dtype=np.float32), np.float32(-8), np.float32(8))


def add(left, right):
    return clip(np.float32(left) + np.float32(right))


def sub(left, right):
    return clip(np.float32(left) - np.float32(right))


def mul(left, right):
    return clip(np.float32(left) * np.float32(right))


def minimum(left, right):
    return clip(np.minimum(np.float32(left), np.float32(right)))


def maximum(left, right):
    return clip(np.maximum(np.float32(left), np.float32(right)))


def absolute(value):
    return clip(np.abs(np.float32(value)))


def aq(left, right):
    # CPU oracle 用标准 sqrt；GPU 的 rsqrtf 允许预声明误差。
    right = np.float32(right)
    reciprocal = np.float32(1) / np.sqrt(np.float32(1) + right * right)
    return clip(np.float32(left) * reciprocal)


FUNCTIONS = {
    "ADD": (2, 2, add),
    "SUB": (3, 2, sub),
    "MUL": (4, 2, mul),
    "MIN": (5, 2, minimum),
    "MAX": (6, 2, maximum),
    "ABS": (7, 1, absolute),
    "AQ": (8, 2, aq),
}


def ephemeral_constant() -> float:
    """模块级工厂可序列化，常数在生成节点时立即量化。"""
    return float(np.float32(random.uniform(-2, 2)))


def make_primitive_set(no_feedback: bool = False) -> gp.PrimitiveSet:
    names = [
        name
        for index, name in enumerate(FEATURE_NAMES)
        if not no_feedback or index not in (1, 2, 3)
    ]
    pset = gp.PrimitiveSet("MAIN", len(names))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(names)})
    for name, (_, arity, function) in FUNCTIONS.items():
        pset.addPrimitive(function, arity, name=name)
    for value in (-1.0, 0.0, 1.0):
        pset.addTerminal(value)
    pset.addEphemeralConstant("ERC", ephemeral_constant)
    return pset
