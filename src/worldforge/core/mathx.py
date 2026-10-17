"""零依赖向量数学原语。

作者：晨星
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全相除；分母为零时返回显式默认值。"""

    if denominator == 0:
        return float(default)
    return float(numerator / denominator)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """计算等长向量点积。"""

    if len(left) != len(right):
        raise ValueError("向量长度必须相同")
    return float(math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=False)))


def l2_norm(vector: Sequence[float]) -> float:
    """计算向量的 L2 范数。"""

    return math.sqrt(math.fsum(float(value) * float(value) for value in vector))


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """返回 L2 归一化副本；零向量保持为零向量。"""

    values = [float(value) for value in vector]
    norm = l2_norm(values)
    if norm == 0.0:
        return [0.0] * len(values)
    return [value / norm for value in values]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算余弦相似度；任一零向量返回零。"""

    if len(left) != len(right):
        raise ValueError("向量长度必须相同")
    denominator = l2_norm(left) * l2_norm(right)
    return safe_div(dot(left, right), denominator)


__all__ = ["safe_div", "dot", "l2_norm", "l2_normalize", "cosine_similarity"]
__author__ = "晨星"
