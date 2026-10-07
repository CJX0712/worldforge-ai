"""IDF 加权词法重排（默认后端）。

作者：晨星
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from worldforge.core.bm25 import robertson_idf
from worldforge.core.sortutil import stable_sort
from worldforge.core.tokenizer import tokenize
from worldforge.types import Hit

_SMOOTHING = 0.5


class LexicalReranker:
    """按查询词 IDF 加权覆盖率加位置先验重排候选。"""

    name = "lexical"

    def __init__(self, positional_prior: float = 0.05) -> None:
        self._positional_prior = float(positional_prior)

    def rerank(self, query: str, candidates: Sequence[Hit], top_k: int) -> list[Hit]:
        if not candidates:
            return []
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return stable_sort(list(candidates))[:top_k]

        texts = [str(hit.meta.get("text", "")) for hit in candidates]
        token_lists = [set(tokenize(text)) for text in texts]
        document_frequency = {
            token: sum(1 for tokens in token_lists if token in tokens) for token in query_tokens
        }
        total = len(candidates)
        idf = {token: robertson_idf(total, document_frequency[token]) for token in query_tokens}

        reranked: list[Hit] = []
        for position, hit in enumerate(candidates):
            tokens = token_lists[position]
            if not tokens:
                score = 0.0
            else:
                weighted_hit = math.fsum(idf[token] for token in query_tokens if token in tokens)
                weighted_total = math.fsum(idf.values())
                coverage = weighted_hit / (weighted_total + _SMOOTHING)
                position_prior = 1.0 / (1.0 + self._positional_prior * position)
                score = coverage * position_prior
            reranked.append(
                Hit(chunk_id=hit.chunk_id, score=float(score), source="rerank", meta=dict(hit.meta))
            )
        ordered = stable_sort(reranked)
        return ordered[:top_k]


__all__ = ["LexicalReranker"]
__author__ = "晨星"
