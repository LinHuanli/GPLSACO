"""共享繁殖修复：有界重试、精确结构判重和64情境行为容量控制。"""

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
from deap import gp

from gp_faco.controller import ROLE_FEATURES, Controller
from gp_faco.evolution import Individual, individual_from_program, isolated_deap_rng
from gp_faco.primitives import feature_names, make_primitive_set
from gp_faco.program_ir import export_tree


@dataclass(frozen=True)
class DiversitySettings:
    population: int = 128
    generations: int = 10
    elites: int = 4
    tournament: int = 4
    max_height: int = 5
    max_nodes: int = 63
    crossover: float = 0.80
    mutation: float = 0.15
    reproduction: float = 0.05
    nonterminal_probability: float = 0.90
    variation_attempts: int = 25
    immigrant_attempts: int = 128
    behavior_cap: int = 2

    def __post_init__(self):
        if not (
            2 <= self.population <= 128
            and 0 <= self.elites < self.population
            and 1 <= self.generations
            and 1 <= self.tournament
            and 1 <= self.max_height <= 5
            and 1 <= self.max_nodes <= 63
            and self.variation_attempts > 0
            and self.immigrant_attempts > 0
            and self.behavior_cap > 0
            and 0 <= self.nonterminal_probability <= 1
            and min(self.crossover, self.mutation, self.reproduction) >= 0
            and abs(self.crossover + self.mutation + self.reproduction - 1) < 1e-12
        ):
            raise ValueError("繁殖协议参数无效")


class Candidate:
    def __init__(self, trees, representation, identifier="candidate"):
        self.trees = trees
        self.representation = representation
        self.identifier = identifier
        self.fitness = None
        self.pc = None
        self._controller = None

    def controller(self):
        if self._controller is None or self._controller.controller_id != self.identifier:
            self._controller = Controller(
                self.representation, tuple(export_tree(t) for t in self.trees), self.identifier
            )
        return self._controller

    def expressions(self):
        return [str(t) for t in self.trees]


class DiverseEvolution:
    def __init__(self, representation, seed, contexts, settings=None):
        settings = settings or DiversitySettings()
        if representation not in ("joint_single", "conditional_three"):
            raise ValueError("未知个体表示")
        self.representation, self.seed, self.contexts, self.settings = (
            representation,
            seed,
            contexts,
            settings,
        )
        self.rng = random.Random(seed)
        self.psets = [
            make_primitive_set(feature_spec_id=2)
            for _ in range(1 if representation == "joint_single" else 3)
        ]
        if representation == "conditional_three":
            names = feature_names(2)
            for pset, allowed in zip(self.psets, ROLE_FEATURES, strict=True):
                pset.terminals[object] = [
                    t
                    for t in pset.terminals[object]
                    if getattr(t, "value", None) not in names or names.index(t.value) in allowed
                ]
        self.generation = 1
        self.population = []
        self.history = []
        self.counts = {}
        self.breeding_seconds = 0.0

    def _counter(self, operator, outcome):
        key = f"{operator}_{outcome}"
        self.counts[key] = self.counts.get(key, 0) + 1

    def _valid(self, candidate):
        return sum(map(len, candidate.trees)) <= self.settings.max_nodes and all(
            t.height <= self.settings.max_height for t in candidate.trees
        )

    def _random(self, position):
        # 每个目标高度×full/grow分层循环，三角色相位错开；grow必须达到该目标高度。
        trees = []
        for role, pset in enumerate(self.psets):
            target = min(2 + (position + role) % 3, self.settings.max_height)
            full = (position // 3 + role) % 2 == 0

            def grow(depth, forced, target=target, pset=pset):
                # 保留一条到目标高度的随机分支，其他分支从深度2起按grow规则终止。
                # 这样不以无界拒绝采样实现高度分层，也不会把抽到浅树误记为深树。
                if depth == target or (
                    depth >= 2 and not forced and random.random() < pset.terminalRatio
                ):
                    terminal = random.choice(pset.terminals[object])
                    return [terminal() if isinstance(terminal, gp.MetaEphemeral) else terminal]
                primitive = random.choice(pset.primitives[object])
                branch = random.randrange(primitive.arity) if forced else -1
                return [primitive] + [
                    node
                    for child in range(primitive.arity)
                    for node in grow(depth + 1, child == branch)
                ]

            expression = gp.genFull(pset, target, target) if full else grow(0, True)
            tree = Individual(expression, 2)
            if tree.height != target:
                return None
            trees.append(tree)
        candidate = Candidate(trees, self.representation)
        return candidate if self._valid(candidate) else None

    def _admissible(self, candidate, accepted):
        if candidate is None or not self._valid(candidate):
            self._counter("rejected", "limits")
            return False
        controller = candidate.controller()
        if any(controller.key == p.controller().key for p in accepted):
            self._counter("rejected", "identical_ir")
            return False
        if candidate.pc is None:
            candidate.pc = self.contexts.behavior(controller)
        if sum(np.array_equal(candidate.pc, p.pc) for p in accepted) >= self.settings.behavior_cap:
            self._counter("rejected", "behavior_capacity")
            return False
        return True

    def _immigrant(self, position, accepted):
        for _ in range(self.settings.immigrant_attempts):
            self._counter("immigrant", "attempted")
            candidate = self._random(position)
            if self._admissible(candidate, accepted):
                self._counter("immigrant", "accepted")
                return candidate
        raise RuntimeError(
            f"{self.representation}/seed{self.seed}/g{self.generation}/i{position}: "
            "随机移民128次内无法满足唯一IR和行为容量；停止本预实验，不放宽约束"
        )

    def _identifiers(self):
        for index, candidate in enumerate(self.population):
            candidate.identifier = (
                f"{self.representation}-s{self.seed}-g{self.generation:02d}-i{index:03d}"
            )
            for role, tree in enumerate(candidate.trees):
                tree.program_id = f"{candidate.identifier}-r{role}"
            candidate.fitness = None

    def initialize(self):
        if self.population:
            raise ValueError("种群已经初始化")
        start = time.perf_counter()
        with isolated_deap_rng(self.rng):
            for position in range(self.settings.population):
                self.population.append(self._immigrant(position, self.population))
        self._identifiers()
        self.breeding_seconds = time.perf_counter() - start

    def tournament(self):
        contestants = [random.choice(self.population) for _ in range(self.settings.tournament)]
        best = min(p.fitness for p in contestants)
        # 繁殖只比较fitness；同分随机，绝不隐含偏好更短的树。
        return random.choice([p for p in contestants if p.fitness == best])

    def node(self, tree):
        nonterminal = random.random() < self.settings.nonterminal_probability
        choices = [i for i, node in enumerate(tree) if bool(node.arity) == nonterminal]
        return random.choice(choices or list(range(len(tree))))

    def mutate(self, parent):
        child = copy.deepcopy(parent)
        role = random.randrange(len(child.trees))
        tree = child.trees[role]
        index = self.node(tree)
        low, high = (1, 3) if len(tree) == 1 else (0, 2)
        tree[tree.searchSubtree(index)] = gp.genFull(self.psets[role], low, high)
        child.pc = None
        child._controller = None
        return child

    def crossover(self, left, right):
        role = random.randrange(len(left.trees))
        child = copy.deepcopy(left)
        a, b = self.node(left.trees[role]), self.node(right.trees[role])
        child.trees[role][child.trees[role].searchSubtree(a)] = copy.deepcopy(
            right.trees[role][right.trees[role].searchSubtree(b)]
        )
        # 多树交叉同时重组规则组合：未选中的角色整树来自另一个父代。
        for other in range(len(child.trees)):
            if other != role:
                child.trees[other] = copy.deepcopy(right.trees[other])
        child.pc = None
        child._controller = None
        return child

    def _variation(self):
        left = self.tournament()
        draw = random.random()
        operator = (
            "crossover"
            if draw < self.settings.crossover
            else (
                "mutation"
                if draw < self.settings.crossover + self.settings.mutation
                else "reproduction"
            )
        )
        self._counter(operator, "attempted")
        if operator == "crossover":
            right = None
            for _ in range(self.settings.variation_attempts):
                candidate = self.tournament()
                if not np.array_equal(left.pc, candidate.pc):
                    right = candidate
                    break
            if right is None:
                self._counter("crossover", "no_distinct_parent")
                operator = "mutation"
                self._counter(operator, "attempted")
                child = self.mutate(left)
            else:
                child = self.crossover(left, right)
        elif operator == "mutation":
            child = self.mutate(left)
        else:
            return copy.deepcopy(left), operator
        if not self._valid(child):
            self._counter(operator, "over_limit")
            return None, operator
        if child.controller().key == left.controller().key:
            self._counter(operator, "unchanged_ir")
            return None, operator
        self._counter(operator, "structure_changed")
        child.pc = self.contexts.behavior(child.controller())
        if not np.array_equal(child.pc, left.pc):
            self._counter(operator, "behavior_changed")
        return child, operator

    def assign(self, fitness):
        if len(fitness) != len(self.population) or any(
            p.fitness is not None for p in self.population
        ):
            raise ValueError("必须一次收齐同一面板全部个体的真实评价")
        if not np.all(np.isfinite(fitness)):
            raise ValueError("预实验评价失败；不将失败当作有效训练数据")
        for candidate, value in zip(self.population, fitness, strict=True):
            candidate.fitness = float(value)

    def ranking(self):
        if any(p.fitness is None for p in self.population):
            raise ValueError("种群尚未全部评价")
        # 报告和精英的稳定排序可以用大小；父代选择不使用此排序。
        return sorted(
            self.population, key=lambda p: (p.fitness, sum(map(len, p.trees)), p.controller().key)
        )

    def metrics(self):
        groups = []
        for p in self.population:
            match = next((g for g in groups if np.array_equal(g[0], p.pc)), None)
            if match is None:
                groups.append([p.pc, 1])
            else:
                match[1] += 1
        actions = np.concatenate([p.pc for p in self.population])
        return {
            "unique_ir": len(self.population),
            "unique_behavior": len(groups),
            "max_behavior_group": max(g[1] for g in groups),
            "variation": self.counts.copy(),
            "breeding_seconds": self.breeding_seconds,
            "heights": [[t.height for t in p.trees] for p in self.population],
            "nodes": [[len(t) for t in p.trees] for p in self.population],
            "context_action_counts": {
                "restart": np.bincount(actions // 16, minlength=2).tolist(),
                "region": np.bincount(actions % 16 // 4, minlength=4).tolist(),
                "mne": np.bincount(actions % 4, minlength=4).tolist(),
            },
        }

    def advance(self):
        if self.generation >= self.settings.generations:
            return False
        start = time.perf_counter()
        self.counts = {}
        accepted = [copy.deepcopy(p) for p in self.ranking()[: self.settings.elites]]
        with isolated_deap_rng(self.rng):
            for position in range(len(accepted), self.settings.population):
                for _ in range(self.settings.variation_attempts):
                    child, operator = self._variation()
                    if self._admissible(child, accepted):
                        self._counter(operator, "accepted")
                        accepted.append(child)
                        break
                else:
                    accepted.append(self._immigrant(position, accepted))
        self.generation += 1
        self.population = accepted
        self._identifiers()
        self.breeding_seconds = time.perf_counter() - start
        return True

    def state_dict(self):
        return {
            "version": 1,
            "representation": self.representation,
            "seed": self.seed,
            "settings": asdict(self.settings),
            "generation": self.generation,
            "rng": self.rng.getstate(),
            "history": self.history,
            "counts": self.counts,
            "breeding_seconds": self.breeding_seconds,
            "population": [
                {"controller": p.controller().to_dict(), "fitness": p.fitness, "pc": p.pc.tolist()}
                for p in self.population
            ],
        }

    @classmethod
    def from_state_dict(cls, state, contexts):
        run = cls(
            state["representation"], state["seed"], contexts, DiversitySettings(**state["settings"])
        )

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value

        run.rng.setstate(tuples(state["rng"]))
        run.generation, run.history, run.counts = (
            state["generation"],
            state["history"],
            state["counts"],
        )
        run.breeding_seconds = state["breeding_seconds"]
        for item in state["population"]:
            controller = Controller.from_dict(item["controller"])
            candidate = Candidate(
                [
                    individual_from_program(p, pset)
                    for p, pset in zip(controller.trees, run.psets, strict=True)
                ],
                run.representation,
                controller.controller_id,
            )
            candidate.fitness, candidate.pc = (
                item["fitness"],
                np.asarray(item["pc"], dtype=np.uint8),
            )
            run.population.append(candidate)
        return run
