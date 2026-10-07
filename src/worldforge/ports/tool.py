"""工具 Port 契约。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from worldforge.types import ToolContext, ToolResult, ToolSpec


@runtime_checkable
class Tool(Protocol):
    """具有可描述参数模式的受控工具接口。"""

    @property
    def spec(self) -> ToolSpec:
        """返回稳定工具说明。"""
        ...

    def invoke(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        """执行工具并把内部失败归一为结果对象。"""
        ...


__all__ = ["Tool"]
__author__ = "晨星"
