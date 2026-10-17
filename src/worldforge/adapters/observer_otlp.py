"""OTLP 可观测性后端（可选 extras: observe-otlp）。

作者：晨星
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from worldforge.adapters.observer_memory import MemoryObserver
from worldforge.core.seeding import deterministic_view


class OtlpObserver(MemoryObserver):
    """继承内存 Observer，并在 flush 时把快照 POST 到 OTLP collector。

    继承而不是组合：span/metric 的记录语义完全一致，远程上报只发生在 flush，
    放在热路径上会把检索延迟和上报延迟耦合起来。
    """

    def __init__(self, endpoint: str = "", buffer_size: int = 1000, timeout: float = 5.0) -> None:
        super().__init__(buffer_size=buffer_size)
        self._endpoint = endpoint
        self._timeout = timeout
        self.exported = 0

    def flush(self) -> dict[str, Any]:
        payload = super().flush()
        if self._endpoint:
            self._export(payload)
        return payload

    def _export(self, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                self.exported += response.status
        except (urllib.error.URLError, TimeoutError):
            # 观测链路失败绝不能影响业务链路：丢数据好过让请求失败
            pass

    def exportable(self) -> dict[str, Any]:
        return deterministic_view({"spans": self._spans, "metrics": self._metrics})


__all__ = ["OtlpObserver"]
__author__ = "晨星"
