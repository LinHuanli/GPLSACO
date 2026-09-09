"""E2动作限制的规范身份；固定规则读取每个变体自己的已完成状态。"""

from __future__ import annotations

from dataclasses import dataclass

from gp_faco.baseline_policy import BaselinePolicy


@dataclass(frozen=True)
class FactorialPolicy:
    variant: str
    baseline_policy: BaselinePolicy
    factorial_spec_id: int = 1

    def __post_init__(self):
        if (
            type(self.factorial_spec_id) is not int
            or self.factorial_spec_id != 1
            or type(self.variant) is not str
            or self.variant not in ("M00", "M10", "M01", "M11")
            or type(self.baseline_policy) is not BaselinePolicy
        ):
            raise ValueError("析因配置需要已验证的M00规则与合法变体/版本")

    def validate_mask(self, mask):
        if type(mask) is not int or not 0 < mask <= 0xFFFFFFFF or not mask & 0xFFFF:
            raise ValueError("析因mask必须保留保持动作")
        if self.variant in ("M00", "M01"):
            self.baseline_policy.validate_mask(mask)

    def to_dict(self):
        return {
            "factorial_spec_id": self.factorial_spec_id,
            "variant": self.variant,
            "baseline_policy": self.baseline_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {
            "factorial_spec_id",
            "variant",
            "baseline_policy",
        }:
            raise ValueError("析因配置必须精确包含规格全部字段")
        return cls(
            value["variant"],
            BaselinePolicy.from_dict(value["baseline_policy"]),
            value["factorial_spec_id"],
        )

    @property
    def identifier(self):
        return f"{self.variant}-{self.baseline_policy.identifier}"
