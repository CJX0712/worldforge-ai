"""倒数排名融合算法。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from worldforge.core.sortutil import stable_sort
from worldforge.types import Hit


def rrf_scores(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """按排名从一开始累计 RRF 分数。"""

    if k <= 0:
        raise ValueError("k 必须大于 0")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    k: int = 60,
    top_k: int | None = None,
) -> list[Hit]:
    """融合多路 Hit 排名并按统一稳定顺序返回。"""

    if top_k is not None and top_k < 0:
        raise ValueError("top_k 不能为负数")
    ranked_ids = [[hit.chunk_id for hit in ranking] for ranking in rankings]
    scores = rrf_scores(ranked_ids, k=k)
    first_hit: dict[str, Hit] = {}
    for ranking in rankings:
        for hit in ranking:
            first_hit.setdefault(hit.chunk_id, hit)
    fused = [
        Hit(chunk_id=chunk_id, score=score, source="fused", meta=first_hit[chunk_id].meta)
        for chunk_id, score in scores.items()
    ]
    ordered = stable_sort(fused)
    return ordered if top_k is None else ordered[:top_k]


rrf_fuse = reciprocal_rank_fusion

__all__ = ["rrf_scores", "reciprocal_rank_fusion", "rrf_fuse"]
__author__ = "晨星"
