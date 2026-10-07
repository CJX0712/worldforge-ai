"""秩相关系数测试。

作者：晨星
"""

from __future__ import annotations

import pytest

from worldforge.core.rank_corr import pearson, spearman


def test_pearson_and_spearman_direction() -> None:
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_any_constant_input_returns_zero() -> None:
    assert pearson([0.5, 0.5, 0.5], [1, 2, 3]) == 0.0
    assert spearman([1, 2, 3], [7, 7, 7]) == 0.0
    assert pearson([2, 2, 2], [3, 3, 3]) == 0.0
    assert spearman([2, 2, 2], [3, 3, 3]) == 0.0


def test_spearman_uses_average_ranks_for_ties() -> None:
    assert spearman([1, 1, 2], [1, 1, 3]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pearson([1], [1, 2])


__author__ = "晨星"
