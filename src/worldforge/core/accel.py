"""可选 NumPy 向量加速及纯 Python 等价回退。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from worldforge.core import mathx

try:
    import numpy as _np
except ImportError:
    _np = None


def numpy_available() -> bool:
    """返回当前进程是否可使用 NumPy 加速。"""

    return _np is not None


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """计算点积，NumPy 不可用时自动使用标准库实现。"""

    if len(left) != len(right):
        raise ValueError("向量长度必须相同")
    if _np is None:
        return mathx.dot(left, right)
    left_array = _np.asarray(left, dtype=_np.float64)
    right_array = _np.asarray(right, dtype=_np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1:
        raise ValueError("输入必须是一维向量")
    return float(_np.dot(left_array, right_array))


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """返回归一化向量，保证零向量语义与纯 Python 路径一致。"""

    if _np is None:
        return mathx.l2_normalize(vector)
    array = _np.asarray(vector, dtype=_np.float64)
    if array.ndim != 1:
        raise ValueError("输入必须是一维向量")
    norm = float(_np.linalg.norm(array))
    if norm == 0.0:
        return [0.0] * int(array.size)
    return [float(value) for value in array / norm]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算余弦相似度，任一零向量返回零。"""

    if len(left) != len(right):
        raise ValueError("向量长度必须相同")
    if _np is None:
        return mathx.cosine_similarity(left, right)
    left_array = _np.asarray(left, dtype=_np.float64)
    right_array = _np.asarray(right, dtype=_np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1:
        raise ValueError("输入必须是一维向量")
    denominator = float(_np.linalg.norm(left_array) * _np.linalg.norm(right_array))
    if denominator == 0.0:
        return 0.0
    return float(_np.dot(left_array, right_array) / denominator)


def cosine_scores(query: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    """计算一个查询向量与矩阵各行的余弦相似度。"""

    if len(matrix) == 0:
        return []
    if _np is None:
        return [mathx.cosine_similarity(query, row) for row in matrix]
    query_array = _np.asarray(query, dtype=_np.float64)
    matrix_array = _np.asarray(matrix, dtype=_np.float64)
    if query_array.ndim != 1 or matrix_array.ndim != 2:
        raise ValueError("查询必须是一维向量且矩阵必须是二维")
    if matrix_array.shape[1] != query_array.shape[0]:
        raise ValueError("向量长度必须相同")
    query_norm = float(_np.linalg.norm(query_array))
    row_norms = _np.linalg.norm(matrix_array, axis=1)
    if query_norm == 0.0:
        return [0.0] * int(matrix_array.shape[0])
    products = matrix_array @ query_array
    return [
        0.0 if float(norm) == 0.0 else float(product / (norm * query_norm))
        for product, norm in zip(products, row_norms)
    ]


__all__ = ["numpy_available", "dot", "l2_normalize", "cosine_similarity", "cosine_scores"]
__author__ = "晨星"
