"""计划器 Port 契约。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from worldforge.types import Plan, Step, ToolSpec


@runtime_checkable
class Planner(Protocol):
    """创建及修订不可变执行计划。"""

    def plan(self, goal: str, tools: Sequence[ToolSpec], ctx: Mapping[str, Any]) -> Plan:
        """为目标创建步骤有向无环图。"""
        ...

    def replan(
        self,
        plan: Plan,
        done: Sequence[Step],
        observation: str,
        ctx: Mapping[str, Any],
    ) -> Plan:
        """根据已完成步骤和观察创建新计划。"""
        ...


__all__ = ["Planner"]
__author__ = "晨星"
