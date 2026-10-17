"""生成评估基线：跑 N 次取均值写入 eval/baseline.json。

基线只用于收紧门限，不用于放宽下限：threshold = max(floor, baseline * 0.95)，
floor 永远生效。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from worldforge.eval.metrics import aggregate  # noqa: E402
from worldforge.eval.runner import run_agent, run_rag, run_retrieval  # noqa: E402
from worldforge.eval.scorecard import BASELINE_PATH  # noqa: E402
from worldforge.pipeline import build_pipeline  # noqa: E402


def collect() -> dict[str, float]:
    return {
        **aggregate(run_retrieval(build_pipeline)),
        **aggregate(run_rag(build_pipeline)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="record evaluation baseline")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=BASELINE_PATH)
    args = parser.parse_args()

    samples: list[dict[str, float]] = []
    for _ in range(max(1, args.runs)):
        samples.append(collect())

    metrics = {
        key: round(statistics.fmean(float(sample.get(key, 0.0)) for sample in samples), 6)
        for key in sorted({key for sample in samples for key in sample})
    }
    payload = {"runs": len(samples), "metrics": metrics, "agent": aggregate(run_agent(build_pipeline))}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
__author__ = "晨星"
