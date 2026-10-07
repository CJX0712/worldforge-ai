"""默认规则计划器与步骤 DAG。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from worldforge.agents.builtin_tools import detect_arithmetic
from worldforge.errors import PlanError
from worldforge.types import Plan, Step, ToolSpec


class RuleBasedPlanner:
    """按确定性规则产出两步计划：检索 -> 生成，或算术直达。

    不需要 LLM，因此在无模型、无网络、无 Key 的环境下也能跑通全链路。
    """

    name = "rule"

    def plan(self, goal: str, tools: Sequence[ToolSpec], ctx: Mapping[str, Any]) -> Plan:
        available = {spec.name for spec in tools}
        steps: list[Step] = []
        if "calculator" in available and detect_arithmetic(goal):
            steps.append(Step(step_id="s1", kind="tool", payload={"tool": "calculator", "args": {"expression": detect_arithmetic(goal) or ""}}))
        else:
            if "search" in available:
                steps.append(Step(step_id="s1", kind="tool", payload={"tool": "search", "args": {"query": goal}}))
            steps.append(
                Step(
                    step_id="s2",
                    kind="generate",
                    payload={"question": goal},
                    depends_on=("s1",) if steps else (),
                )
            )
        return self._validate(Plan(goal=goal, steps=tuple(steps), meta={"planner": self.name}))

    def replan(self, plan: Plan, done: Sequence[Step], observation: str, ctx: Mapping[str, Any]) -> Plan:
        remaining = [step for step in plan.steps if step.step_id not in {d.step_id for d in done}]
        return self._validate(Plan(goal=plan.goal, steps=tuple(remaining), meta=dict(plan.meta)))

    @staticmethod
    def _validate(plan: Plan) -> Plan:
        topological_sort(plan.steps)
        return plan


def topological_sort(steps: Sequence[Step]) -> list[str]:
    """拓扑排序并检测环。"""

    ids = [step.step_id for step in steps]
    if len(set(ids)) != len(ids):
        raise PlanError("duplicate step_id in plan")
    incoming = {step.step_id: set(step.depends_on) for step in steps}
    for step_id, deps in incoming.items():
        unknown = deps - set(ids)
        if unknown:
            raise PlanError(f"step '{step_id}' depends on unknown step(s): {sorted(unknown)}")
    ready = sorted([sid for sid, deps in incoming.items() if not deps])
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for step_id in sorted(incoming):
            if current in incoming[step_id]:
                incoming[step_id].discard(current)
                if not incoming[step_id] and step_id not in order and step_id not in ready:
                    ready.append(step_id)
        ready.sort()
    if len(order) != len(ids):
        raise PlanError(f"cycle detected in plan: {sorted(set(ids) - set(order))}")
    return order


__all__ = ["RuleBasedPlanner", "topological_sort"]
__author__ = "晨星"
