"""Robertson IDF 与 BM25 评分测试。

作者：晨星
"""

from __future__ import annotations

from worldforge.core.bm25 import BM25, bm25_scores, robertson_idf
from worldforge.core.rrf import reciprocal_rank_fusion, rrf_scores
from worldforge.types import Hit


def test_robertson_idf_is_positive_for_half_and_all_documents() -> None:
    assert robertson_idf(4, 2) > 0.0
    assert robertson_idf(4, 4) > 0.0


def test_target_in_one_of_three_documents_scores_strictly_positive() -> None:
    corpus = [
        ["rare", "rare", "alpha"],
        ["beta", "gamma"],
        ["delta", "epsilon"],
    ]
    scores = bm25_scores(corpus, ["rare"])
    assert scores[0] > 0.0
    assert scores[1:] == [0.0, 0.0]
    assert BM25(corpus).scores(["rare"]) == scores


def test_empty_inputs_and_parameter_validation() -> None:
    assert bm25_scores([], ["term"]) == []
    assert bm25_scores([["term"]], []) == [0.0]


def test_rrf_fuses_ranks_and_breaks_ties_by_chunk_id() -> None:
    rankings = [
        [Hit("b", 9.0, "dense"), Hit("a", 8.0, "dense")],
        [Hit("a", 7.0, "sparse"), Hit("b", 6.0, "sparse")],
    ]
    scores = rrf_scores([["b", "a"], ["a", "b"]])
    assert scores["a"] == scores["b"]
    fused = reciprocal_rank_fusion(rankings)
    assert [hit.chunk_id for hit in fused] == ["a", "b"]
    assert all(hit.source == "fused" for hit in fused)


__author__ = "晨星"
