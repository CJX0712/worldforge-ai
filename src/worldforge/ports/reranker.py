"""候选重排 Port 契约。

作者：晨星
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from worldforge.types import Hit


@runtime_checkable
class Reranker(Protocol):
    """对检索候选执行稳定重排。"""

    def rerank(self, query: str, candidates: Sequence[Hit], top_k: int) -> list[Hit]:
        """返回最多 top_k 条重排结果。"""
        ...


__all__ = ["Reranker"]
__author__ = "晨星"
