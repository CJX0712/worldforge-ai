"""黄金集与套件执行。

强制约束：每个 case 都在**全新管道**上跑。复用运行期已写入数据的单例管道会让
doc_id 重复、先前用例的写入污染后续检索，指标忽高忽低且不可复现。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldforge.eval.metrics import (
    aggregate,
    doc_hit_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from worldforge.eval.scorecard import check, load_baseline

DATASET_DIR = Path(__file__).with_name("datasets")

RETRIEVAL_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "r1",
        "query": "BM25 的相关性排序函数是谁提出的",
        "gold_docs": ["kb-bm25"],
        "gold_chunks": ["kb-bm25#0"],
    },
    {
        "id": "r2",
        "query": "Transformer 用什么机制建模长程依赖",
        "gold_docs": ["kb-transformer"],
        "gold_chunks": ["kb-transformer#0"],
    },
    {
        "id": "r3",
        "query": "检索增强生成如何缓解模型幻觉",
        "gold_docs": ["kb-rag"],
        "gold_chunks": ["kb-rag#0"],
    },
)

AGENT_CASES: tuple[dict[str, Any], ...] = (
    {"id": "a1", "utterance": "帮我算 12*(3+4) 等于多少", "tool": "calculator", "expect": "84"},
    {"id": "a2", "utterance": "12*(3+4) 等于多少", "tool": "calculator", "expect": "84"},
    {"id": "a3", "utterance": "echo 你好世界", "tool": "echo", "expect": "你好世界"},
)

RAG_CASES: tuple[dict[str, Any], ...] = (
    {"id": "g1", "question": "BM25 的相关性排序函数是谁提出的", "must_include": "Robertson"},
)


@dataclass(frozen=True)
class Corpus:
    """评估语料：刻意写成非均匀长度，避免分块边界总落在同一位置。"""

    documents: tuple[tuple[str, str], ...] = field(
        default=(
            (
                "kb-transformer",
                "# Transformer\n"
                "Transformer 是一种完全基于自注意力的神经网络架构，2017 年由 Vaswani 等人提出。\n"
                "它用多头注意力同时建模序列中任意两个位置之间的依赖关系。\n"
                "由于不再依赖循环结构，长程依赖可以在单层内被直接建模，训练也能完全并行。\n"
                "位置编码补足了自注意力本身不感知顺序的缺陷，常用的有正弦位置编码与可学习位置编码。",
            ),
            (
                "kb-bm25",
                "# BM25\n"
                "BM25 是一种经典的相关性排序函数，由 Robertson 提出并命名，常被称为 Okapi BM25。\n"
                "它用逆文档频率衡量词项区分度，用文档长度归一化抑制长文档天然占便宜的偏置。\n"
                "BM25 的关键优势是在小语料上依然稳健，不像语义模型那样需要大规模训练数据。\n"
                "在 RRF 融合中，BM25 常作为稀疏分支与稠密向量检索互补。",
            ),
            (
                "kb-rag",
                "# 检索增强生成\n"
                "检索增强生成把信息检索与大语言模型结合成一个完整链路。\n"
                "它先按问题召回相关文档，再让模型只依据这些文档生成答案，从而缓解模型幻觉。\n"
                "证据被压成单行注入提示，能避免消费方按行解析时只取到首行、丢掉绝大部分上下文。\n"
                "检索质量决定上限：召回不到的内容，再强的生成模型也无从作答。",
            ),
        )
    )


def load_jsonl(name: str) -> list[dict[str, Any]]:
    """读取黄金集；文件缺失时返回空列表而不是崩溃。"""

    path = DATASET_DIR / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_retrieval(build: Callable[[], Any], cases: Sequence[Mapping[str, Any]] = RETRIEVAL_CASES) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for case in cases:
        pipeline = build()
        corpus = Corpus()
        for doc_id, text in corpus.documents:
            pipeline.ingest(doc_id, text)
        hits = pipeline.retrieve(str(case["query"]), top_k=5)
        chunk_ids = [h.chunk_id for h in hits]
        doc_ids = [str(h.meta.get("doc_id", "")) for h in hits]
        rows.append(
            {
                "doc_hit_rate": doc_hit_rate(doc_ids, list(case["gold_docs"])),
                "reciprocal_rank": reciprocal_rank(chunk_ids, list(case["gold_chunks"])),
                "recall_at_5": recall_at_k(chunk_ids, list(case["gold_chunks"]), 5),
                "ndcg_at_5": ndcg_at_k(chunk_ids, list(case["gold_chunks"]), 5),
            }
        )
    return rows


def run_rag(build: Callable[[], Any], cases: Sequence[Mapping[str, Any]] = RAG_CASES) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for case in cases:
        pipeline = build()
        for doc_id, text in Corpus().documents:
            pipeline.ingest(doc_id, text)
        result = pipeline.answer(str(case["question"]))
        answer = result["answer"]
        required = str(case["must_include"])
        rows.append(
            {
                "answer_hit": 1.0 if required in answer else 0.0,
                "faithfulness": float(result["faithfulness"]),
                "grounding": float(result["grounding"]),
            }
        )
    return rows


def run_agent(build: Callable[[], Any], cases: Sequence[Mapping[str, Any]] = AGENT_CASES) -> list[dict[str, float]]:
    """工具路由类用例只进 agent_metrics，不进检索聚合。"""

    rows: list[dict[str, float]] = []
    for case in cases:
        pipeline = build()
        for doc_id, text in Corpus().documents:
            pipeline.ingest(doc_id, text)
        run = pipeline.act(str(case["utterance"]))
        rows.append(
            {
                "tool_route_hit": 1.0 if str(case["expect"]) in run["answer"] else 0.0,
                "tool_calls": float(run["tool_calls"]),
                "converged": 1.0 if run["converged"] else 0.0,
            }
        )
    return rows


def evaluate(build: Callable[[], Any], enforce: bool = True) -> dict[str, Any]:
    """跑全套并输出 scorecard。"""

    retrieval = aggregate(run_retrieval(build))
    rag = aggregate(run_rag(build))
    agent = aggregate(run_agent(build))
    metrics = {**retrieval, **rag}
    report: dict[str, Any] = {
        "retrieval_metrics": retrieval,
        "rag_metrics": rag,
        "agent_metrics": agent,
        "note": "agent_metrics 不参与检索聚合：这些用例设计上不检索",
    }
    if enforce:
        report["gate"] = check(metrics, load_baseline())
    else:
        report["gate"] = {"passed": None, "note": "enforce=False，仅观测不判定"}
    return report


__all__ = [
    "Corpus",
    "RETRIEVAL_CASES",
    "RAG_CASES",
    "AGENT_CASES",
    "load_jsonl",
    "run_retrieval",
    "run_rag",
    "run_agent",
    "evaluate",
]
__author__ = "晨星"
