"""稳定 JSON 后缀 IR；导出与导入都验证深度、栈和数值范围。"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from gp_faco.primitives import FUNCTIONS, feature_names

OPERATORS = {opcode: (arity, function) for opcode, arity, function in FUNCTIONS.values()}
_program_numbers = count(1)


@dataclass(frozen=True)
class Program:
    opcode: tuple[int, ...]
    operand: tuple[int, ...]
    constant_bits: tuple[int, ...] = ()
    ir_version: int = 1
    numeric_spec_id: int = 1
    feature_spec_id: int = 1
    program_id: str = field(
        default_factory=lambda: f"program-{next(_program_numbers):06d}", compare=False
    )

    def __post_init__(self) -> None:
        # 防止外部 list 在校验后被原地改写而绕过冻结契约。
        for name in ("opcode", "operand", "constant_bits"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            any(type(v) is not int or v != 1 for v in (self.ir_version, self.numeric_spec_id))
            or type(self.feature_spec_id) is not int
            or self.feature_spec_id not in (1, 2)
        ):
            raise ValueError("未知 IR/数值/特征版本")
        if not 1 <= len(self.opcode) <= 63 or len(self.opcode) != len(self.operand):
            raise ValueError("指令与操作数长度不合法")
        if len(self.constant_bits) > 63:
            raise ValueError("常数池超限")
        for bits in self.constant_bits:
            if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF:
                raise ValueError("常数必须是 uint32 FP32 位模式")
        if any(not math.isfinite(c) or not -2 <= c <= 2 for c in self.constants):
            raise ValueError("ERC 必须有限且在 [-2,2]")
        depths = []
        for opcode, operand in zip(self.opcode, self.operand, strict=True):
            if type(opcode) is not int or type(operand) is not int or not 0 <= operand <= 255:
                raise ValueError("操作码/操作数必须为整数且满足 uint8 范围")
            if opcode == 0:
                if not 0 <= operand < 12:
                    raise ValueError("未知 feature ID")
                depths.append(0)
            elif opcode == 1:
                if not 0 <= operand < len(self.constant_bits):
                    raise ValueError("常数下标越界")
                depths.append(0)
            elif opcode in OPERATORS:
                arity = OPERATORS[opcode][0]
                if operand != 0 or len(depths) < arity:
                    raise ValueError("函数操作数非零或栈下溢")
                depth = 1 + max(depths[-arity:])
                del depths[-arity:]
                depths.append(depth)
                if depth > 5:
                    raise ValueError("树深超过 5")
            else:
                raise ValueError("未知 opcode")
            if len(depths) > 6:
                raise ValueError("栈峰值超过 6")
        if len(depths) != 1:
            raise ValueError("执行后栈必须恰好一个值")

    @property
    def constants(self) -> tuple[float, ...]:
        return tuple(struct.unpack("<f", struct.pack("<I", bits))[0] for bits in self.constant_bits)

    def to_dict(self) -> dict:
        return {
            "program_id": self.program_id,
            "ir_version": self.ir_version,
            "numeric_spec_id": self.numeric_spec_id,
            "feature_spec_id": self.feature_spec_id,
            "opcode": list(self.opcode),
            "operand": list(self.operand),
            "constant_bits": list(self.constant_bits),
        }

    @classmethod
    def from_dict(cls, value: dict) -> Program:
        expected = {
            "ir_version",
            "numeric_spec_id",
            "feature_spec_id",
            "opcode",
            "operand",
            "constant_bits",
        }
        if set(value) not in (expected, expected | {"program_id"}):
            raise ValueError("IR 字段缺失或未知")
        return cls(
            **{
                name: tuple(item) if name in ("opcode", "operand", "constant_bits") else item
                for name, item in value.items()
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def identifier(self) -> str:
        return self.program_id

    @property
    def key(self) -> tuple:
        """算法内的精确 IR 元组：仅用于排序/验证候选去重，不缓存训练 fitness。"""
        return (
            self.ir_version,
            self.numeric_spec_id,
            self.feature_spec_id,
            self.opcode,
            self.operand,
            self.constant_bits,
        )


def export_tree(tree, *, feature_spec_id: int | None = None) -> Program:
    """DEAP 前缀树按原左右顺序导出；不代数化简或重新编号。"""
    if not 1 <= len(tree) <= 63:
        raise ValueError("树大小超限")
    version = getattr(tree, "feature_spec_id", 1) if feature_spec_id is None else feature_spec_id
    if hasattr(tree, "feature_spec_id") and tree.feature_spec_id != version:
        raise ValueError("不能在导出时静默修改个体的特征语义版本")
    feature_ids = {name: index for index, name in enumerate(feature_names(version))}
    opcode, operand, constants = [], [], []

    def visit(index: int, depth: int) -> int:
        if depth > 5 or index >= len(tree):
            raise ValueError("树深超限或前缀不完整")
        node = tree[index]
        if node.arity:
            if node.name not in FUNCTIONS or node.arity != FUNCTIONS[node.name][1]:
                raise ValueError("未知 primitive 或元数")
            next_index = index + 1
            for _ in range(node.arity):
                next_index = visit(next_index, depth + 1)
            opcode.append(FUNCTIONS[node.name][0])
            operand.append(0)
            return next_index
        if isinstance(node.value, str):
            if node.value not in feature_ids:
                raise ValueError("未知 symbolic terminal")
            opcode.append(0)
            operand.append(feature_ids[node.value])
        else:
            constant = float(node.value)
            if not math.isfinite(constant) or not -2 <= constant <= 2:
                raise ValueError("树中的常数超出 ERC 范围")
            bits = struct.unpack("<I", struct.pack("<f", constant))[0]
            if bits not in constants:
                constants.append(bits)
            opcode.append(1)
            operand.append(constants.index(bits))
        return index + 1

    if visit(0, 0) != len(tree):
        raise ValueError("树有多余节点")
    if not getattr(tree, "program_id", None):
        tree.program_id = f"program-{next(_program_numbers):06d}"
    return Program(
        tuple(opcode),
        tuple(operand),
        tuple(constants),
        feature_spec_id=version,
        program_id=tree.program_id,
    )


def evaluate(program: Program, features: np.ndarray) -> np.ndarray:
    """向量化诊断 oracle；线上不从 Python 调用此函数。"""
    if features.dtype != np.float32 or features.ndim < 1 or features.shape[0] != 12:
        raise ValueError("特征必须为首维 12 的 FP32 数组")
    if not np.all(np.isfinite(features)) or np.any((features < 0) | (features > 1)):
        raise ValueError("feature spec v1 的终端必须有限且在 [0,1]")
    stack = []
    for opcode, operand in zip(program.opcode, program.operand, strict=True):
        if opcode == 0:
            stack.append(features[operand])
        elif opcode == 1:
            stack.append(np.full(features.shape[1:], program.constants[operand], dtype=np.float32))
        else:
            arity, function = OPERATORS[opcode]
            arguments = stack[-arity:]
            del stack[-arity:]
            stack.append(function(*arguments))
    return np.asarray(stack[0], dtype=np.float32)


def choose_actions(scores: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] != 32 or masks.shape != (len(scores),):
        raise ValueError("需要 [colony,32] 分数与每 colony 的 mask")
    if masks.dtype != np.uint32 or np.any(masks == 0) or not np.all(np.isfinite(scores)):
        raise ValueError("mask 必须是非零 uint32 且分数有限")
    legal = (masks[:, None] >> np.arange(32, dtype=np.uint32)) & 1
    maximum = np.max(np.where(legal, scores, -np.inf), axis=1, keepdims=True)
    tied = legal.astype(bool) & ((maximum - scores) <= np.float32(1e-6))
    return np.argmax(tied, axis=1).astype(np.int32)
