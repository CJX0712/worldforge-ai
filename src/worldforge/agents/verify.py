"""接地与忠实度核验。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from worldforge.core.text import strip_citations
from worldforge.core.tokenizer import tokenize


def _overlap(answer: str, evidence: str) -> float:
    """词集合余弦：对长度不敏感，长问句短答句也能给出有意义分数。"""

    left = set(tokenize(answer))
    right = set(tokenize(evidence))
    if not left or not right:
        return 0.0
    return len(left & right) / (len(left) ** 0.5 * len(right) ** 0.5)


def grounding_score(answer: str, evidences: Sequence[str]) -> float:
    """答案被证据支撑的比例。

    核验前必须剥离引用标记：`[id#n]` 与 `(依据: ...)` 是元数据不是断言，
    不剥离会把自家输出当成"无支撑断言"，指标凭空掉一截。
    """

    cleaned = strip_citations(answer)
    if not cleaned.strip():
        return 0.0
    if not evidences:
        return 0.0
    best = max(_overlap(cleaned, strip_citations(evidence)) for evidence in evidences)
    return round(min(1.0, best), 6)


def faithfulness(answer: str, evidences: Sequence[str]) -> float:
    """答案句子平均有多少比例能在证据里找到对应片段。"""

    from worldforge.adapters.llm_mock import split_sentences

    cleaned = strip_citations(answer)
    sentences = split_sentences(cleaned)
    if not sentences:
        return 0.0
    corpus = [strip_citations(e) for e in evidences]
    if not corpus:
        return 0.0
    scores = [max(_overlap(sentence, evidence) for evidence in corpus) for sentence in sentences]
    return round(sum(scores) / len(scores), 6)


__all__ = ["grounding_score", "faithfulness"]
__author__ = "晨星"
