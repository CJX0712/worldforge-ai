"""证据取回与渲染。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from worldforge.core.text import render_evidence, single_line
from worldforge.types import Evidence, Hit

CONTEXT_OPEN = "<kb-context>"
CONTEXT_CLOSE = "</kb-context>"


def to_evidence(hits: Sequence[Hit]) -> list[Evidence]:
    """把命中转换为证据对象，缺正文时退化为空串而不是丢弃该命中。"""

    return [
        Evidence(
            chunk_id=hit.chunk_id,
            doc_id=str(hit.meta.get("doc_id", "")),
            text=str(hit.meta.get("text", "")),
            score=float(hit.score),
        )
        for hit in hits
    ]


def render_for_prompt(evidences: Sequence[Evidence]) -> str:
    """渲染成上下文块。

    每个证据压成恰好一行：多行渲染会被消费方按行解析，导致只有首行进入模型，
    上下文静默丢掉约 90%。
    """

    body = render_evidence(evidences)
    return f"{CONTEXT_OPEN}\n{body}\n{CONTEXT_CLOSE}"


def dedupe_evidence(evidences: Sequence[Evidence], limit: int = 5) -> list[Evidence]:
    """按正文去重，避免同一段落被多路召回重复注入。"""

    seen: set[str] = set()
    out: list[Evidence] = []
    for evidence in evidences:
        fingerprint = single_line(evidence.text)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(evidence)
        if len(out) >= limit:
            break
    return out


__all__ = ["CONTEXT_OPEN", "CONTEXT_CLOSE", "to_evidence", "render_for_prompt", "dedupe_evidence"]
__author__ = "晨星"
