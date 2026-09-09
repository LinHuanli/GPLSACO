"""联合单树与条件三树；同一个体只有一个外部终局fitness。"""

from dataclasses import dataclass

import numpy as np

from gp_faco.program_ir import Program, choose_actions, evaluate

ROLES = ("restart", "region", "mne")
ROLE_FEATURES = ((0, 1, 2, 3, 4, 6, 7), (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11), tuple(range(12)))


@dataclass(frozen=True)
class Controller:
    representation: str
    trees: tuple[Program, ...]
    controller_id: str = "controller"

    def __post_init__(self):
        object.__setattr__(self, "trees", tuple(self.trees))
        count = {"joint_single": 1, "conditional_three": 3}.get(self.representation)
        if count is None or len(self.trees) != count or sum(len(p.opcode) for p in self.trees) > 63:
            raise ValueError("控制器表示、树数量或总节点限制无效")
        if any(p.feature_spec_id != 2 for p in self.trees):
            raise ValueError("新控制器只接受评价次数特征v2")
        if count == 3:
            for program, allowed in zip(self.trees, ROLE_FEATURES, strict=True):
                if any(
                    op == 0 and arg not in allowed
                    for op, arg in zip(program.opcode, program.operand, strict=True)
                ):
                    raise ValueError("角色树包含尚未决定的动作特征")

    @property
    def key(self):
        # 精确IR直接比较；编号不参与结构判重，不生成摘要。
        return self.representation, tuple(p.key for p in self.trees)

    def to_dict(self):
        return {
            "controller_version": 1,
            "representation": self.representation,
            "controller_id": self.controller_id,
            "trees": [p.to_dict() for p in self.trees],
        }

    @classmethod
    def from_dict(cls, value):
        if "opcode" in value:
            program = Program.from_dict(value)
            return cls("joint_single", (program,), program.identifier)
        if value.get("controller_version") != 1:
            raise ValueError("未知控制器版本")
        return cls(
            value["representation"],
            tuple(Program.from_dict(p) for p in value["trees"]),
            value["controller_id"],
        )


def score_controller(controller, features, masks, *, evaluator=evaluate):
    """CPU oracle：三次比较都读取同一动作前状态，最后仅提交一个完整动作。"""
    features = np.asarray(features, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.uint32)
    if features.shape != (12, len(masks), 32) or np.any(masks == 0):
        raise ValueError("需要12×情境×32特征和非空合法动作mask")
    if controller.representation == "joint_single":
        scores = evaluator(controller.trees[0], features)
        return {"scores": scores, "actions": choose_actions(scores, masks)}
    prefix = np.zeros(len(masks), dtype=np.int32)
    rows = np.arange(len(masks))
    stages = []
    for program, width, stride in zip(controller.trees, (2, 4, 4), (16, 4, 1), strict=True):
        action = prefix[:, None] + np.arange(width)[None, :] * stride
        # 固定代表候选的特征值；被屏蔽的分支仍可给出分数，但绝不能被选中。
        values = evaluator(program, features)[rows[:, None], action]
        legal = ((masks[:, None] >> action.astype(np.uint32)) & np.uint32((1 << stride) - 1)) != 0
        maximum = np.max(np.where(legal, values, -np.inf), axis=1)
        chosen = np.argmax(legal & (maximum[:, None] - values <= np.float32(1e-6)), axis=1)
        stages.append({"scores": values, "legal": legal, "selected": chosen})
        prefix += chosen.astype(np.int32) * stride
    return {"stages": stages, "actions": prefix}


class DecisionContexts:
    def __init__(self, features, masks):
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.masks = np.ascontiguousarray(masks, dtype=np.uint32)
        self.scorer = score_controller
        if (
            self.features.shape != (12, 64, 32)
            or self.masks.shape != (64,)
            or np.any(self.masks == 0)
        ):
            raise ValueError("行为描述必须使用冻结的64个完整决策情境")

    def behavior(self, controller):
        return np.asarray(
            self.scorer(controller, self.features, self.masks)["actions"], dtype=np.uint8
        )

    def interventions(self, controller):
        original = self.behavior(controller)
        output = {}
        for feature in (1, 2, 3):
            changed = self.features.copy()
            changed[feature] = 1
            actions = np.asarray(self.scorer(controller, changed, self.masks)["actions"])
            output[str(feature)] = int(np.count_nonzero(actions != original))
        return output
