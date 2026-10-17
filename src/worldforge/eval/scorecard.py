"""门限判定与基线读写。

门限策略（ADR-0003）：threshold = max(floor, baseline * 0.95)。
floor 必须是会失败的硬下限，不能写 0.0，否则门禁形同虚设；未回填基线时
threshold 直接取 floor，不留空窗期。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldforge.errors import EvalGateError

FLOORS: dict[str, float] = {
    "doc_hit_rate": 0.60,
    "reciprocal_rank": 0.50,
    "recall_at_5": 0.30,
    "faithfulness": 0.40,
}

BASELINE_PATH = Path(__file__).with_name("baseline.json")


def load_baseline(path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path) if path else BASELINE_PATH
    if not target.exists():
        return {"runs": 0, "metrics": {}}
    return json.loads(target.read_text(encoding="utf-8"))


def thresholds(metrics: Mapping[str, float], baseline: Mapping[str, Any] | None = None) -> dict[str, float]:
    """计算每个指标的门限。"""

    base = (baseline or {}).get("metrics", {})
    out: dict[str, float] = {}
    for key in sorted(set(FLOORS) | set(metrics)):
        floor = FLOORS.get(key)
        measured = base.get(key)
        if floor is None:
            continue
        out[key] = max(floor, round(measured * 0.95, 6)) if isinstance(measured, (int, float)) else floor
    return out


def check(metrics: Mapping[str, float], baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """逐项判定，未达门限抛 EvalGateError。"""

    limits = thresholds(metrics, baseline)
    failures: list[dict[str, Any]] = []
    for key in sorted(limits):
        value = float(metrics.get(key, 0.0))
        if value + 1e-9 < limits[key]:
            failures.append({"metric": key, "value": round(value, 6), "threshold": limits[key]})
    if failures:
        raise EvalGateError(f"eval gate failed: {failures}")
    return {"passed": True, "thresholds": limits, "metrics": {k: metrics[k] for k in sorted(metrics)}}


__all__ = ["FLOORS", "BASELINE_PATH", "load_baseline", "thresholds", "check"]
__author__ = "晨星"
