"""文档分块 Port 契约。

作者：晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from worldforge.types import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    """把文档转换为稳定连续分块。"""

    def chunk(self, doc: Document) -> list[Chunk]:
        """返回序号连续且标识稳定的分块。"""
        ...


__all__ = ["Chunker"]
__author__ = "晨星"
