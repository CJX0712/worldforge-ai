"""跨层共享的不可变数据类型。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

Role = str


@dataclass(frozen=True)
class Document:
    """待摄入的原始文档。"""

    doc_id: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """具有稳定标识的文档分块。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    """检索或重排产生的候选结果。"""

    chunk_id: str
    score: float
    source: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """语言模型对话消息。"""

    role: Role
    content: str
    name: str | None = None


@dataclass(frozen=True)
class Completion:
    """语言模型完成结果。"""

    text: str
    model_id: str
    finish_reason: str
    usage: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """已取回并准备渲染的证据。"""

    chunk_id: str
    doc_id: str
    text: str
    score: float


@dataclass(frozen=True)
class ToolSpec:
    """工具名称、说明及参数模式。"""

    name: str
    description: str
    args_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """工具调用的受控结果。"""

    ok: bool
    output: str
    error: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    """计划中的一个不可变步骤。"""

    step_id: str
    kind: str
    payload: Mapping[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """由依赖关系连接的一组计划步骤。"""

    goal: str
    steps: tuple[Step, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolContext:
    """工具执行所需的显式上下文。"""

    trace_id: str
    seed: int
    observer: Any
    budget: Mapping[str, int] = field(default_factory=dict)


__all__ = [
    "Role",
    "Document",
    "Chunk",
    "Hit",
    "Message",
    "Completion",
    "Evidence",
    "ToolSpec",
    "ToolResult",
    "Step",
    "Plan",
    "ToolContext",
]
__author__ = "晨星"
