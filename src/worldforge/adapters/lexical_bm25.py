"""Robertson IDF 的 BM25 倒排索引（默认后端）。

作者：晨星
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from worldforge.core.bm25 import robertson_idf
from worldforge.core.sortutil import stable_sort
from worldforge.core.tokenizer import tokenize
from worldforge.types import Chunk, Hit


class BM25Index:
    """增量倒排索引，未命中任何词时返回空列表。"""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._tokens: dict[str, list[str]] = {}
        self._frequencies: dict[str, Counter[str]] = {}
        self._doc_ids: dict[str, str] = {}

    def index(self, chunks: Sequence[Chunk]) -> int:
        added = 0
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._tokens[chunk.chunk_id] = tokens
            self._frequencies[chunk.chunk_id] = Counter(tokens)
            self._doc_ids[chunk.chunk_id] = chunk.doc_id
            added += 1
        return added

    def drop_document(self, doc_id: str) -> int:
        targets = [chunk_id for chunk_id, owner in self._doc_ids.items() if owner == doc_id]
        for chunk_id in targets:
            self._tokens.pop(chunk_id, None)
            self._frequencies.pop(chunk_id, None)
            self._doc_ids.pop(chunk_id, None)
        return len(targets)

    def search(self, query: str, top_k: int) -> list[Hit]:
        if top_k <= 0:
            return []
        query_tokens = [token for token in tokenize(query) if token]
        if not query_tokens or not self._tokens:
            return []
        total = len(self._tokens)
        average_length = math.fsum(len(tokens) for tokens in self._tokens.values()) / total
        document_frequency = {
            token: sum(1 for counter in self._frequencies.values() if counter.get(token, 0) > 0)
            for token in set(query_tokens)
        }
        hits: list[Hit] = []
        for chunk_id, counter in self._frequencies.items():
            length = len(self._tokens.get(chunk_id, ()))
            length_ratio = length / average_length if average_length else 0.0
            terms: list[float] = []
            matched = 0
            for token in query_tokens:
                frequency = counter.get(token, 0)
                if frequency == 0:
                    continue
                matched += 1
                idf = robertson_idf(total, document_frequency[token])
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                terms.append(idf * frequency * (self.k1 + 1.0) / denominator)
            if matched == 0:
                continue
            score = float(math.fsum(terms))
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    score=score,
                    source="sparse",
                    meta={"doc_id": self._doc_ids.get(chunk_id, ""), "matched": matched},
                )
            )
        ordered = stable_sort(hits)
        return ordered[:top_k]

    def stats(self) -> dict[str, float]:
        total = len(self._tokens)
        average_length = math.fsum(len(tokens) for tokens in self._tokens.values()) / total if total else 0.0
        vocabulary = {token for tokens in self._tokens.values() for token in tokens}
        return {"docs": float(total), "avgdl": float(average_length), "vocab": float(len(vocabulary))}


__all__ = ["BM25Index"]
__author__ = "晨星"
