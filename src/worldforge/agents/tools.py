"""工具注册表与确定性路由。

作者：晨星
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from worldforge.agents.builtin_tools import detect_arithmetic
from worldforge.core.tokenizer import normalize_text
from worldforge.types import ToolContext, ToolResult, ToolSpec


class ToolRegistry:
    """按工具名注册与调用。

    不变量
        I1  路由是确定性的：算术优先，其次显式指定工具名，最后按参数形状
        I2  工具名与参数键都按 normalize_text 归一，中文问句与英文工具名都能命中
    """

    def __init__(self, tools: Sequence[Any] = ()) -> None:
        self._tools: dict[str, Any] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Any) -> None:
        self._tools[normalize_text(tool.spec.name)] = tool

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in sorted(self._tools)]

    def get(self, name: str) -> Any | None:
        return self._tools.get(normalize_text(name))

    def route(self, utterance: str, explicit: str | None = None) -> tuple[str | None, dict[str, Any]]:
        """决定该调哪个工具。

        小模型的路由与算术不可靠，所以算术走确定性前置规则：正则检出算式就
        直接交给 calculator，绝不让模型重算。
        """

        if explicit:
            return normalize_text(explicit), {}
        expression = detect_arithmetic(utterance)
        if expression and "calculator" in self._tools:
            return "calculator", {"expression": expression}
        normalized = normalize_text(utterance)
        for name in sorted(self._tools):
            if name and name in normalized:
                return name, {}
        return None, {}

    def invoke(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(ok=False, output="", error=f"unknown tool '{name}'")
        try:
            return tool.invoke(args, ctx)
        except Exception as exc:  # noqa: BLE001 - 工具内部异常必须归一为结果对象
            return ToolResult(ok=False, output="", error=f"{type(exc).__name__}: {exc}")

    def parse_action(self, action: str, raw_args: str) -> dict[str, Any]:
        """把 ReAct 的 Action 行解析成参数字典。"""

        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"text": raw_args.strip()}
        return parsed if isinstance(parsed, dict) else {"text": raw_args.strip()}


__all__ = ["ToolRegistry"]
__author__ = "晨星"
