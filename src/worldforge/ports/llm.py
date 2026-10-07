"""语言模型 Port 契约。

作者：晨星
"""

from __future__ import annotations

from typing import Iterator, Protocol, Sequence, runtime_checkable

from worldforge.types import Completion, Message


@runtime_checkable
class LLM(Protocol):
    """同步语言模型的结构化接口。"""

    @property
    def model_id(self) -> str:
        """返回不含墙钟信息的稳定模型标识。"""
        ...

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        """完成单轮提示。"""
        ...

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        """完成多轮对话。"""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Iterator[str]:
        """逐块产出文本增量。"""
        ...


__all__ = ["LLM"]
__author__ = "晨星"
