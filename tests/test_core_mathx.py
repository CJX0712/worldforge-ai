"""向量数学原语测试。

作者：晨星
"""

from __future__ import annotations

import math

import pytest

from worldforge.core.hash_embed import hash_embed
from worldforge.core.mathx import cosine_similarity, dot, l2_norm, l2_normalize, safe_div
from worldforge.core.sortutil import stable_sort
from worldforge.types import Hit


def test_dot_norm_and_normalization() -> None:
    vector = [3.0, 4.0]
    assert dot(vector, [1.0, 2.0]) == 11.0
    assert l2_norm(vector) == 5.0
    assert l2_normalize(vector) == pytest.approx([0.6, 0.8], abs=1e-12)


def test_zero_vector_and_safe_division() -> None:
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert safe_div(1.0, 0.0, default=-1.0) == -1.0


def test_cosine_and_dimension_validation() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 1.0]), 1.0 / math.sqrt(2.0))
    with pytest.raises(ValueError):
        dot([1.0], [1.0, 2.0])


def test_hash_embedding_and_stable_sort_are_deterministic() -> None:
    first = hash_embed("人工智能 AI", dim=32)
    assert first == hash_embed("人工智能 AI", dim=32)
    assert l2_norm(first) == pytest.approx(1.0, abs=1e-12)
    hits = [Hit("b", 1.0, "dense"), Hit("a", 1.0, "dense"), Hit("c", 2.0, "dense")]
    assert [hit.chunk_id for hit in stable_sort(hits)] == ["c", "a", "b"]


__author__ = "晨星"
