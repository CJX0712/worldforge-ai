"""NumPy 加速与纯 Python 数值一致性测试。

作者：晨星
"""

from __future__ import annotations

import random

import pytest

np = pytest.importorskip("numpy")

from worldforge.core import accel  # noqa: E402
from worldforge.core.mathx import cosine_similarity as python_cosine  # noqa: E402
from worldforge.core.mathx import dot as python_dot  # noqa: E402
from worldforge.core.mathx import l2_normalize as python_normalize  # noqa: E402


def test_accelerated_path_matches_python_for_random_vectors() -> None:
    assert accel.numpy_available()
    generator = random.Random(1337)
    for _ in range(200):
        left = [generator.uniform(-10.0, 10.0) for _ in range(32)]
        right = [generator.uniform(-10.0, 10.0) for _ in range(32)]
        assert accel.dot(left, right) == pytest.approx(python_dot(left, right), abs=1e-12)
        assert accel.cosine_similarity(left, right) == pytest.approx(python_cosine(left, right), abs=1e-12)
        assert accel.l2_normalize(left) == pytest.approx(python_normalize(left), abs=1e-12)


def test_zero_vector_parity() -> None:
    zeros = [0.0] * 8
    assert accel.l2_normalize(zeros) == python_normalize(zeros)
    assert accel.cosine_similarity(zeros, zeros) == python_cosine(zeros, zeros)


__author__ = "晨星"
