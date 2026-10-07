"""对检索结果执行统一的确定性排序。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def stable_sort(
    items: Iterable[T],
    *,
    key: Callable[[T], float] | None = None,
    tie_key: Callable[[T], str] | None = None,
    score_getter: Callable[[T], float] | None = None,
    id_getter: Callable[[T], str] | None = None,
) -> list[T]:
    """统一按分数降序、稳定标识升序排序。

    key / tie_key 是 ADAPTERS.md 示例使用的别名，与 score_getter / id_getter 等价。
    """

    get_score = key or score_getter or _default_score
    get_id = tie_key or id_getter or _default_id
    return sorted(items, key=lambda item: (-float(get_score(item)), str(get_id(item))))


def _default_score(item: Any) -> float:
    if hasattr(item, "score"):
        return float(item.score)
    if isinstance(item, dict) and "score" in item:
        return float(item["score"])
    raise ValueError("排序项缺少 score")


def _default_id(item: Any) -> str:
    for name in ("chunk_id", "doc_id", "id"):
        if hasattr(item, name):
            return str(getattr(item, name))
        if isinstance(item, dict) and name in item:
            return str(item[name])
    raise ValueError("排序项缺少稳定标识")


__all__ = ["stable_sort"]
__author__ = "晨星"
