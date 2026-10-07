"""混合检索：双路召回 -> 融合 -> 重排。

作者：晨星
"""

from __future__ import annotations

from worldforge.retrieval.dense import DenseRetriever
from worldforge.retrieval.fusion import fuse
from worldforge.retrieval.sparse import SparseRetriever
from worldforge.types import Hit


class HybridRetriever:
    """稀疏 + 稠密双路召回，RRF 融合后交给重排器。

    不变量
        I1  候选 meta 必须带 text，重排器要靠它算覆盖率
        I2  hybrid=False 时只走稀疏分支，但 source 仍标记为真实来源
    """

    def __init__(
        self,
        sparse: SparseRetriever,
        dense: DenseRetriever,
        reranker: object,
        *,
        hybrid: bool = True,
        fusion: str = "rrf",
        rrf_k: int = 60,
        top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> None:
        self._sparse = sparse
        self._dense = dense
        self._reranker = reranker
        self._hybrid = hybrid
        self._fusion = fusion
        self._rrf_k = rrf_k
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        self.last_sparse: list[Hit] = []
        self.last_dense: list[Hit] = []

    def search(self, query: str) -> list[Hit]:
        sparse_hits = self._sparse.search(query, self._top_k)
        self.last_sparse = sparse_hits
        rankings: list[list[Hit]] = [sparse_hits]
        self.last_dense = []
        if self._hybrid:
            self.last_dense = self._dense.search(query, self._top_k)
            rankings.append(self.last_dense)

        fused = fuse(rankings, method=self._fusion, rrf_k=self._rrf_k)
        candidates = self._hydrate(fused)
        return list(self._reranker.rerank(query, candidates, self._rerank_top_k))  # type: ignore[attr-defined]

    def _hydrate(self, hits: list[Hit]) -> list[Hit]:
        """补齐重排所需的正文，并剔除已不存在的分块。"""

        missing = [hit.chunk_id for hit in hits if not hit.meta.get("text")]
        texts: dict[str, str] = {}
        if missing:
            store = getattr(self._sparse._lexical, "_docstore", None)
            if store is not None:  # pragma: no cover - 仅在装配了 docstore 时生效
                for chunk in store.get_chunks(missing):  # type: ignore[attr-defined]
                    texts[chunk.chunk_id] = chunk.text
        return [
            Hit(
                chunk_id=hit.chunk_id,
                score=hit.score,
                source=hit.source,
                meta={**hit.meta, "text": hit.meta.get("text") or texts.get(hit.chunk_id, "")},
            )
            for hit in hits
        ]


__all__ = ["HybridRetriever"]
__author__ = "晨星"
