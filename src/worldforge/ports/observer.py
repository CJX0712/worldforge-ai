"""可观测性 Port 契约。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Observer(Protocol):
    """追踪、事件和指标的统一接口。"""

    def span(self, name: str, **attrs: Any) -> Any:
        """返回记录墙钟延迟的上下文管理器。"""
        ...

    def emit(self, event: str, **fields: Any) -> None:
        """记录结构化事件。"""
        ...

    def metric(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        """记录一个数值指标。"""
        ...

    def trace_id(self) -> str:
        """返回当前追踪标识。"""
        ...

    def flush(self) -> dict[str, Any]:
        """导出并清空缓冲。"""
        ...


__all__ = ["Observer"]
__author__ = "晨星"
