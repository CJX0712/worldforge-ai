"""默认文档存储：进程内字典实现。

作者：晨星
"""

from __future__ import annotations

from typing import Sequence

from worldforge.types import Chunk, Document


class MemoryDocStore:
    """进程内 DocStore。

    选它作默认是因为评估需要完全确定性：sqlite 的写入顺序与页布局会让
    `chunks_of` 的返回顺序依赖插入历史，跨运行不可复现。
    """

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}

    def put_document(self, doc: Document) -> None:
        self._docs[doc.doc_id] = doc

    def get_document(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[str]:
        return sorted(self._docs)[offset : offset + limit]

    def put_chunks(self, chunks: Sequence[Chunk]) -> int:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        return len(chunks)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        found = [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]
        # 字典序而非请求顺序：外部传入的顺序不可控，按 chunk_id 排序才能跨运行复现
        return sorted(found, key=lambda c: c.chunk_id)

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        owned = [c for c in self._chunks.values() if c.doc_id == doc_id]
        return sorted(owned, key=lambda c: c.ordinal)

    def drop_document(self, doc_id: str) -> int:
        self._docs.pop(doc_id, None)
        orphans = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in orphans:
            del self._chunks[cid]
        return len(orphans)

    def count_chunks(self) -> int:
        return len(self._chunks)


__all__ = ["MemoryDocStore"]
__author__ = "晨星"
