"""默认可观测性实现：环形缓冲 Observer。

作者：晨星
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


class MemoryObserver:
    """进程内追踪与指标缓冲。

    墙钟延迟只在这里产生（`span` 内部），其余层不得读取时间做业务判断。
    """

    def __init__(self, buffer_size: int = 1000) -> None:
        self._buffer_size = max(1, buffer_size)
        self._spans: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._metrics: list[dict[str, Any]] = []
        self._trace = "trace-local"

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        record: dict[str, Any] = {"name": name, "trace_id": self._trace, **attrs}
        start = time.perf_counter()
        try:
            yield record
        finally:
            record["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 6)
            self._spans.append(record)
            self._evict(self._spans)

    def emit(self, event: str, **fields: Any) -> None:
        self._events.append({"event": event, "trace_id": self._trace, **fields})
        self._evict(self._events)

    def metric(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        self._metrics.append(
            {"name": name, "value": float(value), "tags": dict(tags or {}), "trace_id": self._trace}
        )
        self._evict(self._metrics)

    def trace_id(self) -> str:
        return self._trace

    def flush(self) -> dict[str, Any]:
        """导出并清空缓冲。

        这里返回**原始**数据（含墙钟延迟）：延迟是导出的合法内容。把它剔除
        是确定性视图的职责，由 observability.strip_latency / deterministic_view
        在对外边界上完成，分层做才不会两处都丢信息或都不丢。
        """

        payload = {
            "spans": [dict(item) for item in self._spans],
            "events": [dict(item) for item in self._events],
            "metrics": [dict(item) for item in self._metrics],
        }
        self._spans.clear()
        self._events.clear()
        self._metrics.clear()
        return payload

    def _evict(self, bucket: list[Any]) -> None:
        if len(bucket) > self._buffer_size:
            del bucket[: len(bucket) - self._buffer_size]


__all__ = ["MemoryObserver"]
__author__ = "晨星"
