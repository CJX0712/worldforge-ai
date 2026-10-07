"""跨运行比较所需的固定种子与确定性视图。

作者：晨星
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, Callable

_EXCLUDE_KEYS = frozenset(
    {
        "ts",
        "timestamp",
        "started_at",
        "ended_at",
        "created",
        "created_at",
        "latency_ms",
        "duration_ms",
        "wall_ms",
        "elapsed_ms",
        "pid",
        "hostname",
        "thread_id",
        "uptime_s",
    }
)


def seed_everything(seed: int) -> int:
    """固定标准库随机源与子进程可继承的哈希种子。"""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed 必须是整数")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def deterministic_view(obj: Any, ndims: int = 6) -> Any:
    """剔除墙钟字段并转换为可稳定比较的纯数据结构。"""

    if ndims < 0:
        raise ValueError("ndims 不能为负数")
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return repr(obj)
        rounded = round(obj, ndims)
        return 0.0 if rounded == 0.0 else rounded
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return deterministic_view(
            {field.name: getattr(obj, field.name) for field in fields(obj)},
            ndims,
        )
    if hasattr(obj, "_asdict"):
        return deterministic_view(obj._asdict(), ndims)
    if isinstance(obj, Mapping):
        pairs = sorted(
            ((key, value) for key, value in obj.items() if str(key) not in _EXCLUDE_KEYS),
            key=lambda pair: str(pair[0]),
        )
        return {str(key): deterministic_view(value, ndims) for key, value in pairs}
    if isinstance(obj, (list, tuple)):
        return [deterministic_view(value, ndims) for value in obj]
    if isinstance(obj, (set, frozenset)):
        values = [deterministic_view(value, ndims) for value in obj]
        return sorted(values, key=_canonical_bytes)
    return repr(obj)


def assert_deterministic(
    fn: Callable[..., Any],
    *args: Any,
    runs: int = 2,
    ndims: int = 6,
    **kwargs: Any,
) -> Any:
    """重复执行函数并断言规范化输出逐字节相同。"""

    if runs < 2:
        raise ValueError("runs 必须至少为 2")
    first_result = fn(*args, **kwargs)
    expected = _canonical_bytes(deterministic_view(first_result, ndims))
    for run_index in range(1, runs):
        result = fn(*args, **kwargs)
        actual = _canonical_bytes(deterministic_view(result, ndims))
        if actual != expected:
            raise AssertionError(f"第 {run_index + 1} 次运行结果不确定")
    return first_result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = ["seed_everything", "deterministic_view", "assert_deterministic"]
__author__ = "晨星"
