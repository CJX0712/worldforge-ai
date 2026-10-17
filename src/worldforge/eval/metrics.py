"""检索与生成质量指标。

作者：晨星
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from worldforge.core.text import strip_citations
from worldforge.core.tokenizer import tokenize


def doc_hit_rate(retrieved: Sequence[str], gold_docs: Sequence[str]) -> float:
    """文档级命中率，作为检索头条指标。

    块级召回在没有人工标注段落时会被过宽的真值系统性低估：把整篇文档的所有
    分块标为相关，其中多数从未包含答案。所以头条看文档级，块级只作诊断。
    """

    if not gold_docs:
        return 0.0
    return round(len(set(retrieved) & set(gold_docs)) / len(set(gold_docs)), 6)


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """块级召回（诊断项）。"""

    if not gold:
        return 0.0
    hit = len(set(retrieved[:k]) & set(gold))
    return round(hit / len(set(gold)), 6)


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """首个正确命中的倒数排名。"""

    gold_set = set(gold)
    for rank, item in enumerate(retrieved, start=1):
        if item in gold_set:
            return round(1.0 / rank, 6)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """归一化折损累计增益（二值相关性）。"""

    gold_set = set(gold)
    if not gold_set:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], start=1) if item in gold_set)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return round(dcg / ideal, 6) if ideal else 0.0


def _tokens(text: str) -> set[str]:
    return set(tokenize(strip_citations(text)))


def lexical_faithfulness(answer: str, evidence: Sequence[str]) -> float:
    """答案词元被证据覆盖率（不依赖任何模型的忠实度基线）。"""

    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for item in evidence:
        evidence_tokens |= _tokens(item)
    if not evidence_tokens:
        return 0.0
    return round(len(answer_tokens & evidence_tokens) / len(answer_tokens), 6)


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """对多次运行取均值，结果键排序保证跨运行可比。"""

    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
        for key in keys
    }


__all__ = [
    "doc_hit_rate",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "lexical_faithfulness",
    "aggregate",
]
__author__ = "晨星"
