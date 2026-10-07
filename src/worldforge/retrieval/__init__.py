"""检索层：分块编排、双路召回、融合与证据渲染。"""

from worldforge.retrieval.chunking import Ingestor
from worldforge.retrieval.dense import DenseRetriever
from worldforge.retrieval.evidence import dedupe_evidence, render_for_prompt, to_evidence
from worldforge.retrieval.fusion import fuse
from worldforge.retrieval.hybrid import HybridRetriever
from worldforge.retrieval.sparse import SparseRetriever

__all__ = [
    "Ingestor",
    "SparseRetriever",
    "DenseRetriever",
    "fuse",
    "HybridRetriever",
    "to_evidence",
    "render_for_prompt",
    "dedupe_evidence",
]
__author__ = "晨星"
