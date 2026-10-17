"""Robertson IDF 形式的零依赖 BM25 评分。

作者：晨星
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def robertson_idf(total_documents: int, document_frequency: int) -> float:
    """计算恒为正的 Robertson IDF。"""

    if total_documents < 0:
        raise ValueError("total_documents 不能为负数")
    if not 0 <= document_frequency <= total_documents:
        raise ValueError("document_frequency 必须位于合法范围")
    if total_documents == 0:
        return 0.0
    ratio = (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
    return math.log1p(ratio)


def bm25_scores(
    documents: Sequence[Sequence[str]],
    query_tokens: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """按输入顺序返回每篇已分词文档的 BM25 分数。"""

    if k1 <= 0:
        raise ValueError("k1 必须大于 0")
    if not 0.0 <= b <= 1.0:
        raise ValueError("b 必须位于 0 到 1")
    corpus = [tuple(document) for document in documents]
    if not corpus:
        return []
    if not query_tokens:
        return [0.0] * len(corpus)

    frequencies = [Counter(document) for document in corpus]
    lengths = [len(document) for document in corpus]
    average_length = math.fsum(lengths) / len(lengths)
    document_frequency = {
        token: sum(1 for frequency in frequencies if frequency.get(token, 0) > 0)
        for token in set(query_tokens)
    }

    scores: list[float] = []
    for frequency, length in zip(frequencies, lengths, strict=False):
        score_terms: list[float] = []
        length_ratio = length / average_length if average_length else 0.0
        for token in query_tokens:
            term_frequency = frequency.get(token, 0)
            if term_frequency == 0:
                continue
            idf = robertson_idf(len(corpus), document_frequency[token])
            denominator = term_frequency + k1 * (1.0 - b + b * length_ratio)
            score_terms.append(idf * term_frequency * (k1 + 1.0) / denominator)
        scores.append(float(math.fsum(score_terms)))
    return scores


class BM25:
    """保存分词语料并提供可重复使用的 BM25 评分。"""

    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._documents = tuple(tuple(document) for document in documents)
        self.k1 = float(k1)
        self.b = float(b)
        bm25_scores(self._documents, (), k1=self.k1, b=self.b)

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        """为查询词元返回全部文档的分数。"""

        return bm25_scores(self._documents, query_tokens, k1=self.k1, b=self.b)

    def score(self, query_tokens: Sequence[str], document_index: int) -> float:
        """返回指定文档的分数。"""

        return self.scores(query_tokens)[document_index]


__all__ = ["BM25", "robertson_idf", "bm25_scores"]
__author__ = "晨星"
