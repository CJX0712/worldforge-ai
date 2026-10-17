"""文档存储 Port 契约。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from worldforge.types import Chunk, Document


@runtime_checkable
class DocStore(Protocol):
    """原始文档及分块的存取接口。"""

    def put_document(self, doc: Document) -> None:
        """覆盖写入文档元数据。"""
        ...

    def get_document(self, doc_id: str) -> Document | None:
        """读取文档，不存在时返回空值。"""
        ...

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[str]:
        """按字典序分页列出文档标识。"""
        ...

    def put_chunks(self, chunks: Sequence[Chunk]) -> int:
        """覆盖写入分块并返回条数。"""
        ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """按请求顺序返回存在的分块。"""
        ...

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        """按序号升序返回文档分块。"""
        ...

    def drop_document(self, doc_id: str) -> int:
        """删除文档及其全部分块。"""
        ...

    def count_chunks(self) -> int:
        """返回分块总数。"""
        ...


__all__ = ["DocStore"]
__author__ = "晨星"
