"""评估层：指标、门限与套件执行。"""

from worldforge.eval.metrics import (
    aggregate,
    doc_hit_rate,
    lexical_faithfulness,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from worldforge.eval.runner import Corpus, evaluate, run_agent, run_rag, run_retrieval
from worldforge.eval.scorecard import FLOORS, check, load_baseline, thresholds

__all__ = [
    "doc_hit_rate",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "lexical_faithfulness",
    "aggregate",
    "check",
    "thresholds",
    "load_baseline",
    "FLOORS",
    "evaluate",
    "run_retrieval",
    "run_rag",
    "run_agent",
    "Corpus",
]
__author__ = "晨星"
