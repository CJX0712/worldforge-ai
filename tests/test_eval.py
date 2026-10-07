"""评估指标、门限与套件确定性。"""

from __future__ import annotations

import pytest

from worldforge.errors import EvalGateError
from worldforge.eval.metrics import aggregate, doc_hit_rate, ndcg_at_k, recall_at_k, reciprocal_rank
from worldforge.eval.runner import evaluate, run_retrieval
from worldforge.eval.scorecard import FLOORS, check, load_baseline, thresholds
from worldforge.pipeline import build_pipeline


def test_doc_hit_rate_uses_document_level_truth():
    assert doc_hit_rate(["d1", "d2"], ["d1"]) == 1.0
    assert doc_hit_rate(["d3"], ["d1"]) == 0.0
    assert doc_hit_rate(["d1"], []) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_recall_and_ndcg():
    assert recall_at_k(["a", "b"], ["a", "z"], 2) == 0.5
    assert ndcg_at_k(["a", "b"], ["a"], 2) == pytest.approx(1.0)


def test_aggregate_is_key_sorted():
    assert list(aggregate([{"b": 1.0, "a": 2.0}])) == ["a", "b"]


def test_threshold_never_drops_below_floor():
    assert thresholds({"doc_hit_rate": 0.9}, {"metrics": {"doc_hit_rate": 0.2}})["doc_hit_rate"] == FLOORS["doc_hit_rate"]


def test_threshold_tightens_against_baseline():
    got = thresholds({"doc_hit_rate": 0.9}, {"metrics": {"doc_hit_rate": 1.0}})["doc_hit_rate"]
    assert got == pytest.approx(0.95)


def test_check_raises_when_below_floor():
    with pytest.raises(EvalGateError):
        check({"doc_hit_rate": 0.1, "reciprocal_rank": 0.1})


def test_check_passes_on_current_corpus():
    report = evaluate(build_pipeline, enforce=True)
    assert report["gate"]["passed"] is True


def test_agent_metrics_excluded_from_retrieval_aggregation():
    report = evaluate(build_pipeline, enforce=False)
    assert "tool_route_hit" in report["agent_metrics"]
    assert "tool_route_hit" not in report["retrieval_metrics"]


def test_suite_is_deterministic_across_runs():
    first = aggregate(run_retrieval(build_pipeline))
    second = aggregate(run_retrieval(build_pipeline))
    assert first == second


def test_baseline_file_is_valid_json():
    baseline = load_baseline()
    assert "metrics" in baseline and "runs" in baseline
