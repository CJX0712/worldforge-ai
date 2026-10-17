"""追踪、指标与导出。

作者：晨星
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from worldforge.core.seeding import deterministic_view

_EXCLUDED_LATENCY = frozenset({"latency_ms", "elapsed_ms", "wall_ms"})


def strip_latency(payload: Any) -> Any:
    """剔除墙钟延迟字段，得到可参与确定性断言的视图。

    延迟必须走独立的 perf 通道：把它混进断言载荷会让同一份代码每次都"失败"。
    """

    return _strip(payload)


def _strip(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _strip(v) for k, v in value.items() if k not in _EXCLUDED_LATENCY}
    if isinstance(value, (list, tuple)):
        return [_strip(v) for v in value]
    return value


class Metrics:
    """计数、量表与分位统计。"""

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, list[float]] = {}

    def incr(self, name: str, value: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0.0) + value

    def observe(self, name: str, value: float) -> None:
        self._gauges.setdefault(name, []).append(value)

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {"counters": dict(sorted(self._counters.items()))}
        for name in sorted(self._gauges):
            samples = sorted(self._gauges[name])
            count = len(samples)
            out[name] = {
                "count": count,
                "p50": samples[count // 2],
                "p95": samples[min(count - 1, int(count * 0.95))],
                "max": samples[-1],
            }
        return out


class Tracer:
    """带上下文的追踪器；嵌套 span 用 contextvar 传递父级。"""

    def __init__(self, observer: Any) -> None:
        self._observer = observer

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        with self._observer.span(name, **attrs) as record:
            yield record

    def emit(self, event: str, **fields: Any) -> None:
        self._observer.emit(event, **fields)

    def metric(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        self._observer.metric(name, value, tags)

    def flush(self) -> dict[str, Any]:
        return strip_latency(self._observer.flush())


def export_json(payload: Any) -> str:
    """导出确定性 JSON（剔除延迟、键排序、浮点定点）。"""

    return json.dumps(deterministic_view(strip_latency(payload)), ensure_ascii=False, sort_keys=True)


def export_prometheus(metrics: Mapping[str, Any]) -> str:
    """导出 Prometheus 文本格式。"""

    lines: list[str] = []
    for name in sorted(metrics.get("counters", {})):
        lines.append(f"worldforge_counter{{{name}}} {metrics['counters'][name]}")
    for name in sorted(metrics):
        if name == "counters" or not isinstance(metrics[name], Mapping):
            continue
        for quantile in ("p50", "p95", "max"):
            lines.append(f"worldforge_latency_{quantile}{{stage=\"{name}\"}} {metrics[name][quantile]}")
    return "\n".join(lines) + "\n"


__all__ = ["Metrics", "Tracer", "strip_latency", "export_json", "export_prometheus"]
__author__ = "晨星"
