"""Static/Rule的规范参数身份；不使用GP IR或标签承载基线规则。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class BaselinePolicy:
    policy_spec_id: int = 1
    kind: str = "static"
    mne_level: int = 0
    max_mne_level: int = 0
    region: int = 0
    restart_mode: str = "none"
    restart_period: int = 0
    restart_probability: float = 0.0
    stagnation_step: int = 0
    restart_stagnation: int = 0
    restart_cooldown: int = 0

    def __post_init__(self):
        for field in fields(self):
            if field.name in ("kind", "restart_mode", "restart_probability"):
                continue
            value = getattr(self, field.name)
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError("基线版本、档位与次数参数必须为uint32整数")
        if (
            self.policy_spec_id != 1
            or type(self.kind) is not str
            or type(self.restart_mode) is not str
            or self.kind not in ("static", "rule")
            or not 0 <= self.mne_level <= self.max_mne_level <= 3
            or self.region > 3
            or type(self.restart_probability) not in (int, float)
            or not math.isfinite(self.restart_probability)
            or not 0 <= self.restart_probability <= 1
        ):
            raise ValueError("基线类别、版本、MNE、区域或概率无效")
        if self.kind == "static":
            if (
                self.max_mne_level != self.mne_level
                or self.stagnation_step
                or self.restart_stagnation
                or self.restart_cooldown
            ):
                raise ValueError("静态控制不能附带反馈规则")
            valid = (
                (
                    self.restart_mode == "none"
                    and not self.restart_period
                    and not self.restart_probability
                )
                or (
                    self.restart_mode == "periodic"
                    and self.restart_period > 0
                    and not self.restart_probability
                )
                or (self.restart_mode == "bernoulli" and not self.restart_period)
            )
            if not valid:
                raise ValueError("静态重启模式与周期/概率不符")
        elif (
            self.restart_mode != "none"
            or self.restart_period
            or self.restart_probability
            or ((self.max_mne_level == self.mne_level) != (self.stagnation_step == 0))
            or ((self.restart_stagnation == 0) != (self.restart_cooldown == 0))
        ):
            raise ValueError("规则控制需要规范的停滞升级/重启冷却参数")
        object.__setattr__(self, "restart_probability", float(self.restart_probability) + 0.0)

    def validate_mask(self, mask: int) -> None:
        required = sum(
            1 << (self.region * 4 + level)
            for level in range(self.mne_level, self.max_mne_level + 1)
        )
        if type(mask) is not int or not 0 < mask <= 0xFFFFFFFF or mask & required != required:
            raise ValueError("实验mask必须保留基线可能选择的保持动作")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> BaselinePolicy:
        if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
            raise ValueError("基线配置必须精确包含规格的全部字段")
        return cls(**value)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()
