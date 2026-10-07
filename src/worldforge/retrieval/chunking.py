"""摄入编排：文档 -> 分块 -> 索引。

作者：晨星
"""

from __future__ import annotations

from typing import Sequence

from worldforge.errors import IngestError
from worldforge.types import Chunk, Document


class Ingestor:
    """把文档写入 DocStore、VectorStore 与 LexicalIndex。

    不变量
        I1  重复摄入同一 doc_id 是**替换**而非 upsert：先 drop 再写，
            否则旧修订的剩余分块会变成检索得到却已不存在的孤儿
        I2  三个后端写入必须同时成功或同时不写（失败即回滚文档侧）
    """

    def __init__(self, docstore: object, embedder: object, vectorstore: object, lexical: object, max_chunks: int = 5000) -> None:
        self._docs = docstore
        self._embedder = embedder
        self._vectors = vectorstore
        self._lexical = lexical
        self._max_chunks = max_chunks

    def ingest(self, doc: Document, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            raise IngestError(f"document '{doc.doc_id}' produced zero chunks")
        if len(chunks) > self._max_chunks:
            raise IngestError(f"document '{doc.doc_id}' produced {len(chunks)} chunks, above cap {self._max_chunks}")

        self._docs.drop_document(doc.doc_id)
        self._vectors.drop_document(doc.doc_id)
        self._lexical.drop_document(doc.doc_id)

        try:
            self._docs.put_document(doc)
            self._docs.put_chunks(chunks)
            vectors = self._embedder.embed([c.text for c in chunks])  # type: ignore[attr-defined]
            self._vectors.upsert(
                [c.chunk_id for c in chunks],
                vectors,
                [{"doc_id": c.doc_id, "ordinal": c.ordinal, "text": c.text, **dict(c.meta)} for c in chunks],
            )
            self._lexical.index(chunks)
        except Exception:
            # 半写入状态比写入失败更难排查：显式回滚到"文档不存在"
            self._docs.drop_document(doc.doc_id)
            self._vectors.drop_document(doc.doc_id)
            self._lexical.drop_document(doc.doc_id)
            raise
        return len(chunks)

    def drop(self, doc_id: str) -> int:
        removed = self._vectors.drop_document(doc_id)
        self._lexical.drop_document(doc_id)
        self._docs.drop_document(doc_id)
        return removed


__all__ = ["Ingestor"]
__author__ = "晨星"
