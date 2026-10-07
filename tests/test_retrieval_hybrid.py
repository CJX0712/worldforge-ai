"""混合检索、双语扩展与融合稳定性。"""

from __future__ import annotations

from worldforge.config import Config
from worldforge.core.sortutil import stable_sort
from worldforge.pipeline import build_pipeline
from worldforge.retrieval.fusion import fuse
from worldforge.types import Hit

DOCS = (
    ("kb-en", "Retrieval augmented generation combines information retrieval with a large language model."),
    ("kb-zh", "检索增强生成把信息检索与大语言模型结合起来，先召回文档再生成答案。"),
    ("kb-bm25", "BM25 uses inverse document frequency to measure term distinctiveness and normalizes by document length."),
)


def _seed(pipeline):
    for doc_id, text in DOCS:
        pipeline.ingest(doc_id, text)


def test_hybrid_fusion_returns_target_first():
    pipeline = build_pipeline(Config())
    _seed(pipeline)
    hits = pipeline.retrieve("BM25 逆文档频率 区分度", top_k=3)
    assert hits[0].meta["doc_id"] == "kb-bm25"
    assert hits[0].score >= hits[-1].score


def test_explain_exposes_both_branches():
    pipeline = build_pipeline(Config())
    _seed(pipeline)
    pipeline.retrieve("BM25", top_k=3)
    retriever = pipeline.parts["retriever"]
    assert retriever.last_sparse, "稀疏分支没有产出候选"
    assert retriever.last_dense, "稠密分支没有产出候选"


def test_bilingual_expansion_only_touches_sparse_branch():
    pipeline = build_pipeline(Config(bilingual=True))
    _seed(pipeline)
    pipeline.retrieve("retrieval augmented generation 检索增强", top_k=3)
    sparse = pipeline.parts["retriever"].last_sparse
    dense = pipeline.parts["retriever"].last_dense
    assert sparse and dense
    # 稀疏查询被扩展，稠密分支仍使用原查询文本
    assert pipeline.parts["retriever"]._sparse.last_expanded  # noqa: SLF001
    assert len(dense) > 0


def test_rrf_quantisation_absorbs_tiny_score_gap():
    def make(score: float) -> Hit:
        return Hit(chunk_id="a", score=score, source="dense", meta={})

    exact = fuse([[make(1.0)]], method="rrf")
    nudged = fuse([[make(1.0 + 1e-11)]], method="rrf")
    assert exact[0].score == nudged[0].score


def test_fusion_keeps_best_meta_and_marks_source():
    a = Hit(chunk_id="a", score=0.9, source="sparse", meta={"doc_id": "d1"})
    b = Hit(chunk_id="a", score=0.4, source="dense", meta={"doc_id": "d1", "text": "body"})
    fused = fuse([[a], [b]])
    assert len(fused) == 1
    assert fused[0].source == "fused"
    assert fused[0].meta["text"] == "body"


def test_stable_sort_breaks_ties_by_id():
    hits = [Hit(chunk_id="c", score=1.0, source="s"), Hit(chunk_id="a", score=1.0, source="s")]
    assert [h.chunk_id for h in stable_sort(hits)] == ["a", "c"]


def test_max_fusion_takes_the_highest_score():
    hits = fuse([[Hit(chunk_id="a", score=0.2, source="s")], [Hit(chunk_id="a", score=0.8, source="d")]], method="max")
    assert hits[0].score == 0.8
