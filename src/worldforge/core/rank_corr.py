"""Pearson 与 Spearman 秩相关系数。

作者：晨星
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """计算 Pearson 系数；任一输入为常量时返回零。"""

    _validate_pair(left, right)
    if len(left) < 2:
        return 0.0
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    left_mean = math.fsum(left_values) / len(left_values)
    right_mean = math.fsum(right_values) / len(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    left_ss = math.fsum(value * value for value in left_centered)
    right_ss = math.fsum(value * value for value in right_centered)
    if left_ss == 0.0 or right_ss == 0.0:
        return 0.0
    numerator = math.fsum(a * b for a, b in zip(left_centered, right_centered))
    correlation = numerator / math.sqrt(left_ss * right_ss)
    return max(-1.0, min(1.0, float(correlation)))


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """以平均秩处理并列值并计算 Spearman 系数。"""

    _validate_pair(left, right)
    if len(left) < 2 or _is_constant(left) or _is_constant(right):
        return 0.0
    return pearson(_average_ranks(left), _average_ranks(right))


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: (float(pair[1]), pair[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def _is_constant(values: Sequence[float]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def _validate_pair(left: Sequence[float], right: Sequence[float]) -> None:
    if len(left) != len(right):
        raise ValueError("两个序列长度必须相同")


pearson_correlation = pearson
spearman_correlation = spearman

__all__ = ["pearson", "spearman", "pearson_correlation", "spearman_correlation"]
__author__ = "晨星"
