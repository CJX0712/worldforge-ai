"""多路召回融合。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from worldforge.core.rrf import rrf_scores
from worldforge.core.sortutil import stable_sort
from worldforge.types import Hit

# 1e-11 量级的分差经 RRF 的 rank 翻转会放大成 2.64e-4 的分数跳变（约 2.6e7 倍），
# 稳定排序救不了——根因在 rank 不在顺序。融合前先量化到 6 位小数。
_QUANTIZE = 6


def fuse(rankings: Sequence[Sequence[Hit]], method: str = "rrf", rrf_k: int = 60) -> list[Hit]:
    """把多路候选融合成单一排序。"""

    if method == "rrf":
        return _rrf_fuse(rankings, rrf_k)
    if method == "max":
        return _max_fuse(rankings)
    raise ValueError(f"unknown fusion method: {method}")


def _rrf_fuse(rankings: Sequence[Sequence[Hit]], k: int) -> list[Hit]:
    quantised: list[list[tuple[str, float]]] = []
    for ranking in rankings:
        seen: dict[str, float] = {}
        for hit in ranking:
            key = hit.chunk_id
            value = round(hit.score, _QUANTIZE)
            if key not in seen or value > seen[key]:
                seen[key] = value
        quantised.append(sorted(seen.items()))

    order: list[list[str]] = []
    for items in quantised:
        order.append([cid for cid, _ in sorted(items, key=lambda item: -item[1])])
    scores = rrf_scores(order, k=k)

    merged: dict[str, Hit] = {}
    for ranking in rankings:
        for hit in ranking:
            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = Hit(
                    chunk_id=hit.chunk_id,
                    score=round(scores.get(hit.chunk_id, 0.0), 12),
                    source="fused",
                    meta=dict(hit.meta),
                )
            else:
                # 同一分块被多路召回：补齐后到路的元数据，保留最先出现的
                for key, value in hit.meta.items():
                    existing.meta.setdefault(key, value)
    return stable_sort(list(merged.values()))


def _max_fuse(rankings: Sequence[Sequence[Hit]]) -> list[Hit]:
    best: dict[str, Hit] = {}
    for ranking in rankings:
        for hit in ranking:
            score = round(hit.score, _QUANTIZE)
            current = best.get(hit.chunk_id)
            if current is None or score > round(current.score, _QUANTIZE):
                best[hit.chunk_id] = Hit(
                    chunk_id=hit.chunk_id, score=score, source="fused", meta=dict(hit.meta)
                )
    return stable_sort(list(best.values()))


__all__ = ["fuse"]
__author__ = "晨星"
