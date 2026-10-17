"""稀疏词法索引 Port 契约。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from worldforge.types import Chunk, Hit


@runtime_checkable
class LexicalIndex(Protocol):
    """文档分块的稀疏索引接口。"""

    def index(self, chunks: Sequence[Chunk]) -> int:
        """建立或增量更新索引并返回新增条数。"""
        ...

    def drop_document(self, doc_id: str) -> int:
        """移除指定文档的全部倒排记录。"""
        ...

    def search(self, query: str, top_k: int) -> list[Hit]:
        """返回稳定排序的稀疏检索结果。"""
        ...

    def stats(self) -> dict[str, float]:
        """返回文档数、平均长度和词汇量。"""
        ...


__all__ = ["LexicalIndex"]
__author__ = "晨星"
