"""吞吐与延迟基准，结果写 perf.json。

延迟单独成文件，绝不混进 verify 的确定性断言：同一份代码每次墙钟都不同。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from worldforge.eval.runner import Corpus  # noqa: E402
from worldforge.pipeline import build_pipeline  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main() -> int:
    iterations = 200
    pipeline = build_pipeline()
    for doc_id, text in Corpus().documents:
        pipeline.ingest(doc_id, text)
    queries = ["BM25 是谁提出的", "Transformer 如何建模长程依赖", "检索增强生成怎么缓解幻觉"]

    for query in queries:  # 预热，避免把首次构建索引的成本算进延迟
        pipeline.retrieve(query, top_k=5)

    latencies: list[float] = []
    for index in range(iterations):
        query = queries[index % len(queries)]
        started = time.perf_counter()
        pipeline.retrieve(query, top_k=5)
        latencies.append((time.perf_counter() - started) * 1000.0)

    answer_started = time.perf_counter()
    for _ in range(20):
        pipeline.answer("BM25 是谁提出的")
    answer_ms = (time.perf_counter() - answer_started) * 1000.0 / 20

    report = {
        "iterations": iterations,
        "retrieve_p50_ms": round(percentile(latencies, 0.50), 4),
        "retrieve_p95_ms": round(percentile(latencies, 0.95), 4),
        "retrieve_mean_ms": round(statistics.fmean(latencies), 4),
        "retrieve_qps": round(1000.0 / statistics.fmean(latencies), 2),
        "answer_mean_ms": round(answer_ms, 4),
        "note": "CPU-only 零依赖栈的延迟；启用 extras 后端后数值不可比",
    }
    (ROOT / "perf.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
__author__ = "晨星"
