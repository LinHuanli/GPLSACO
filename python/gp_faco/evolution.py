"""标准DEAP遗传算子与显式代际屏障；不调用Python在线策略。"""

from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import ClassVar

from deap import algorithms, base, gp

from gp_faco.primitives import FUNCTIONS, feature_names, make_primitive_set
from gp_faco.program_ir import Program, export_tree


class MinimizeFitness(base.Fitness):
    weights: ClassVar = (-1.0,)


class Individual(gp.PrimitiveTree):
    def __init__(self, content, feature_spec_id: int = 1):
        super().__init__(content)
        self.feature_spec_id = feature_spec_id
        self.fitness = MinimizeFitness()
        self.evaluated_panel = None


@dataclass(frozen=True)
class EvolutionSettings:
    population: int = 128
    generations: int = 50
    initial_depth_min: int = 1
    initial_depth_max: int = 3
    max_depth: int = 5
    max_nodes: int = 63
    tournament_size: int = 4
    elites: int = 4
    crossover_probability: float = 0.8
    mutation_probability: float = 0.2
    mutation_depth_min: int = 0
    mutation_depth_max: int = 2
    no_feedback: bool = False
    feature_spec_id: int = 1

    def __post_init__(self) -> None:
        integers = {
            name: value
            for name, value in asdict(self).items()
            if name not in ("crossover_probability", "mutation_probability", "no_feedback")
        }
        if any(type(v) is not int for v in integers.values()):
            raise ValueError("演化计数与树限制必须为整数")
        if not (
            2 <= self.population
            and 1 <= self.generations
            and 0 <= self.elites < self.population
            and 1 <= self.tournament_size
            and 0 <= self.initial_depth_min <= self.initial_depth_max <= self.max_depth <= 5
            and 0 <= self.mutation_depth_min <= self.mutation_depth_max <= self.max_depth
            and 1 <= self.max_nodes <= 63
            and type(self.no_feedback) is bool
            and self.feature_spec_id in (1, 2)
        ):
            raise ValueError("演化参数范围无效")
        for value in (self.crossover_probability, self.mutation_probability):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("交叉与变异概率必须在[0,1]")
        # 保证初始化无需丢弃后重抽；最大二叉树是所有生成树的上界。
        if 2 ** (self.initial_depth_max + 1) - 1 > self.max_nodes:
            raise ValueError("初始化深度可能生成超过节点限制的树")


@contextmanager
def isolated_deap_rng(rng: random.Random):
    """DEAP使用模块random；短暂切换后恢复，面板抽样用另一个Random对象。"""
    prior = random.getstate()
    random.setstate(rng.getstate())
    try:
        yield
    finally:
        rng.setstate(random.getstate())
        random.setstate(prior)


def ranking_key(individual: Individual) -> tuple:
    if not individual.fitness.valid:
        raise ValueError("不能选择尚未评价的个体")
    value = individual.fitness.values[0]
    if math.isnan(value) or value == -math.inf:
        raise ValueError("fitness只能是有限值或显式失败的+infinity")
    program = export_tree(individual)
    return value, len(individual), program.sha256


def individual_from_program(program: Program, pset) -> Individual:
    """从验证后的postfix重建前缀树；不eval字符串，不作代数化简。"""
    names = {opcode: name for name, (opcode, _, _) in FUNCTIONS.items()}
    if program.feature_spec_id != getattr(pset, "feature_spec_id", 1):
        raise ValueError("checkpoint程序与当前grammar特征版本不符")
    stack = []
    for op, operand in zip(program.opcode, program.operand, strict=True):
        if op == 0:
            name = feature_names(program.feature_spec_id)[operand]
            if name not in pset.mapping:
                raise ValueError("程序使用了当前grammar已删除的反馈终端")
            stack.append([copy.deepcopy(pset.mapping[name])])
        elif op == 1:
            stack.append([gp.Terminal(program.constants[operand], False, object)])
        else:
            primitive = copy.deepcopy(pset.mapping[names[op]])
            children = stack[-primitive.arity :]
            del stack[-primitive.arity :]
            stack.append([primitive] + [node for child in children for node in child])
    individual = Individual(stack[0], program.feature_spec_id)
    if export_tree(individual) != program:
        raise ValueError("checkpoint程序不是本项目规范化导出的IR")
    return individual


class Evolution:
    def __init__(self, settings: EvolutionSettings, seed: int):
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("演化seed必须为uint64")
        self.settings = settings
        self.rng = random.Random(seed)
        self.pset = make_primitive_set(settings.no_feedback, settings.feature_spec_id)
        self.generation = 0
        self.population: list[Individual] = []
        self.panel_id: str | None = None
        self.winners: list[dict] = []
        self.variation_counts = {"crossovers": 0, "mutations": 0, "fallbacks": 0}
        self.toolbox = base.Toolbox()
        # staticLimit会复制它收到的全部位置参数；辅助参数必须藏在方法中，不能预绑定给toolbox。
        self.toolbox.register("mate", self._mate_counted)
        self.toolbox.register("mutate", self._mutate_counted)
        # 一个联合谓词只作一次随机父代回退，避免堆叠两个limit造成额外随机选择。
        bounded = gp.staticLimit(self._over_limit, 0)
        self.toolbox.decorate("mate", bounded)
        self.toolbox.decorate("mutate", bounded)

    def _over_limit(self, individual) -> int:
        return int(
            individual.height > self.settings.max_depth or len(individual) > self.settings.max_nodes
        )

    def _mutate(self, individual):
        def expression(pset, type_):
            return gp.genFull(
                pset, self.settings.mutation_depth_min, self.settings.mutation_depth_max, type_
            )

        return gp.mutUniform(individual, expression, self.pset)

    def _counted_operator(self, operation, name, *individuals):
        self.variation_counts[name] += 1
        results = operation(*individuals)
        self.variation_counts["fallbacks"] += sum(self._over_limit(v) for v in results)
        return results

    def _mate_counted(self, left, right):
        return self._counted_operator(gp.cxOnePoint, "crossovers", left, right)

    def _mutate_counted(self, individual):
        return self._counted_operator(self._mutate, "mutations", individual)

    def initialize(self) -> None:
        if self.population:
            raise ValueError("种群已经初始化")
        with isolated_deap_rng(self.rng):
            self.population = [
                Individual(
                    gp.genHalfAndHalf(
                        self.pset, self.settings.initial_depth_min, self.settings.initial_depth_max
                    ),
                    self.settings.feature_spec_id,
                )
                for _ in range(self.settings.population)
            ]
        for individual in self.population:
            export_tree(individual)

    def begin_panel(self, panel_id: str) -> None:
        if not self.population or self.panel_id is not None or not panel_id:
            raise ValueError("代际状态不允许开始新面板")
        self.panel_id = panel_id
        # 不依赖varAnd的变化检测：精英、未变树和完全重复树全部失效。
        for individual in self.population:
            del individual.fitness.values
            individual.evaluated_panel = None

    def assign(self, index: int, value: float, panel_id: str, program_sha256: str) -> None:
        if panel_id != self.panel_id or self.panel_id is None:
            raise ValueError("返回分数属于其他代面板")
        if type(index) is not int or not 0 <= index < len(self.population):
            raise ValueError("个体位置越界")
        individual = self.population[index]
        if individual.fitness.valid or export_tree(individual).sha256 != program_sha256:
            raise ValueError("重复赋值或程序身份不符")
        if type(value) not in (int, float) or math.isnan(value) or value == -math.inf:
            raise ValueError("无效fitness")
        individual.fitness.values = (float(value),)
        individual.evaluated_panel = panel_id

    def finish_generation(self) -> dict:
        if self.panel_id is None or any(
            not individual.fitness.valid or individual.evaluated_panel != self.panel_id
            for individual in self.population
        ):
            raise ValueError("全部个体完成当前面板之前不能结束一代")
        if len(self.winners) != self.generation:
            raise ValueError("本代已经归档")
        winner = min(self.population, key=ranking_key)
        result = {
            "generation": self.generation,
            "panel_id": self.panel_id,
            "program": export_tree(winner).to_dict(),
            "expression": str(winner),
            "fitness": winner.fitness.values[0]
            if math.isfinite(winner.fitness.values[0])
            else None,
        }
        self.winners.append(result)
        return result

    def advance(self) -> bool:
        if len(self.winners) != self.generation + 1:
            raise ValueError("必须先评价并保存本代冠军")
        if self.generation + 1 == self.settings.generations:
            return False  # 最后一代保留已评价种群，不创建悬空未评后代。
        order = sorted(self.population, key=ranking_key)
        elites = [copy.deepcopy(v) for v in order[: self.settings.elites]]
        with isolated_deap_rng(self.rng):
            parents = [
                min(
                    [random.choice(self.population) for _ in range(self.settings.tournament_size)],
                    key=ranking_key,
                )
                for _ in range(self.settings.population - self.settings.elites)
            ]
            offspring = algorithms.varAnd(
                parents,
                self.toolbox,
                self.settings.crossover_probability,
                self.settings.mutation_probability,
            )
        self.population = elites + offspring
        self.generation += 1
        self.panel_id = None
        for individual in self.population:
            del individual.fitness.values
            individual.evaluated_panel = None
            if self._over_limit(individual):
                raise RuntimeError("staticLimit未拦截超限后代")
            export_tree(individual)
        return True

    def shortlist(self) -> tuple[Program, ...]:
        if (
            self.generation + 1 != self.settings.generations
            or len(self.winners) != self.settings.generations
            or any(not v.fitness.valid for v in self.population)
        ):
            raise ValueError("只有已评价的最终代才能形成验证shortlist")
        programs = [Program.from_dict(v["program"]) for v in self.winners]
        programs.extend(export_tree(v) for v in self.population)
        return tuple(sorted({p.sha256: p for p in programs}.values(), key=lambda p: p.sha256))

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "settings": asdict(self.settings),
            "generation": self.generation,
            "panel_id": self.panel_id,
            "winners": copy.deepcopy(self.winners),
            "rng_state": self.rng.getstate(),
            "variation_counts": dict(self.variation_counts),
            "population": [
                {
                    "program": export_tree(v).to_dict(),
                    "valid": v.fitness.valid,
                    "fitness": v.fitness.values[0]
                    if v.fitness.valid and math.isfinite(v.fitness.values[0])
                    else None,
                    "evaluated_panel": v.evaluated_panel,
                }
                for v in self.population
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Evolution:
        expected = {
            "version",
            "settings",
            "generation",
            "panel_id",
            "winners",
            "rng_state",
            "variation_counts",
            "population",
        }
        if set(state) != expected or type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("未知演化checkpoint结构")
        restored = cls(EvolutionSettings(**state["settings"]), 0)
        generation = state["generation"]
        if type(generation) is not int or not 0 <= generation < restored.settings.generations:
            raise ValueError("checkpoint代数越界")
        restored.generation = generation
        panel = state["panel_id"]
        if panel is not None and (type(panel) is not str or not panel):
            raise ValueError("checkpoint面板身份无效")
        restored.panel_id = panel
        if len(state["population"]) != restored.settings.population:
            raise ValueError("checkpoint种群大小不符")
        for item in state["population"]:
            if (
                set(item) != {"program", "valid", "fitness", "evaluated_panel"}
                or type(item["valid"]) is not bool
            ):
                raise ValueError("checkpoint个体结构无效")
            individual = individual_from_program(Program.from_dict(item["program"]), restored.pset)
            if restored._over_limit(individual):
                raise ValueError("checkpoint个体超出当前树限制")
            if item["valid"]:
                if panel is None or item["evaluated_panel"] != panel:
                    raise ValueError("checkpoint个体分数不属于当前面板")
                value = item["fitness"]
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value)
                ):
                    raise ValueError("checkpoint fitness必须有限或显式null失败")
                individual.fitness.values = (math.inf if value is None else value,)
                individual.evaluated_panel = panel
            elif item["fitness"] is not None or item["evaluated_panel"] is not None:
                raise ValueError("未评个体含旧fitness或面板身份")
            restored.population.append(individual)
        winners = copy.deepcopy(state["winners"])
        if len(winners) not in (generation, generation + 1):
            raise ValueError("checkpoint冠军数量与代际不符")
        for index, winner in enumerate(winners):
            if (
                set(winner) != {"generation", "panel_id", "program", "expression", "fitness"}
                or type(winner["generation"]) is not int
                or winner["generation"] != index
                or type(winner["panel_id"]) is not str
                or not winner["panel_id"]
                or (
                    winner["fitness"] is not None
                    and (
                        type(winner["fitness"]) not in (int, float)
                        or not math.isfinite(winner["fitness"])
                    )
                )
            ):
                raise ValueError("checkpoint冠军来源无效")
            individual = individual_from_program(
                Program.from_dict(winner["program"]), restored.pset
            )
            if restored._over_limit(individual) or str(individual) != winner["expression"]:
                raise ValueError("checkpoint冠军超出限制或表达式与IR不同")
        if len(winners) == generation + 1 and (
            panel is None or any(not v.fitness.valid for v in restored.population)
        ):
            raise ValueError("未完成整代就保存了冠军")
        if len(winners) == generation + 1:
            best = min(restored.population, key=ranking_key)
            best_fitness = best.fitness.values[0]
            if (
                winners[-1]["panel_id"] != panel
                or Program.from_dict(winners[-1]["program"]) != export_tree(best)
                or winners[-1]["fitness"] != (best_fitness if math.isfinite(best_fitness) else None)
            ):
                raise ValueError("checkpoint当前冠军不符合完整种群的选择规则")
        restored.winners = winners
        if set(state["variation_counts"]) != set(restored.variation_counts) or any(
            type(value) is not int or value < 0 for value in state["variation_counts"].values()
        ):
            raise ValueError("checkpoint变异计数无效")
        restored.variation_counts = dict(state["variation_counts"])
        restored.rng.setstate(_tuples(state["rng_state"]))
        return restored


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
