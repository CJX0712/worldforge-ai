"""向量存储 Port 契约。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from worldforge.types import Hit


@runtime_checkable
class VectorStore(Protocol):
    """向量索引的写入、检索与持久化接口。"""

    @property
    def dim(self) -> int:
        """返回索引向量维度。"""
        ...

    @property
    def fingerprint(self) -> str:
        """返回模型标识与维度组成的索引指纹。"""
        ...

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metas: Sequence[Mapping[str, Any]],
    ) -> int:
        """按标识覆盖写入并返回写入条数。"""
        ...

    def delete(self, ids: Sequence[str]) -> int:
        """删除指定分块并返回实际删除条数。"""
        ...

    def drop_document(self, doc_id: str) -> int:
        """删除文档的全部向量。"""
        ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[Hit]:
        """执行余弦检索并返回稳定排序结果。"""
        ...

    def count(self) -> int:
        """返回索引中的向量数量。"""
        ...

    def clear(self) -> None:
        """清空索引及指纹。"""
        ...

    def save(self, path: str) -> None:
        """保存索引。"""
        ...

    def load(self, path: str) -> None:
        """加载索引并校验指纹。"""
        ...


__all__ = ["VectorStore"]
__author__ = "晨星"
