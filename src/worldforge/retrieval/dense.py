"""稠密召回编排。

作者：晨星
"""

from __future__ import annotations

from worldforge.types import Hit


class DenseRetriever:
    """向量余弦召回。"""

    def __init__(self, embedder: object, vectorstore: object) -> None:
        self._embedder = embedder
        self._store = vectorstore

    def search(self, query: str, top_k: int) -> list[Hit]:
        vector = self._embedder.embed_one(query)  # type: ignore[attr-defined]
        hits = list(self._store.search(vector, top_k))  # type: ignore[attr-defined]
        return [
            Hit(chunk_id=hit.chunk_id, score=hit.score, source="dense", meta=dict(hit.meta))
            for hit in hits
        ]


__all__ = ["DenseRetriever"]
__author__ = "晨星"
