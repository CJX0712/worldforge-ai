"""固定种子与确定性视图测试。

作者：晨星
"""

from __future__ import annotations

import random

import pytest

from worldforge.core.seeding import assert_deterministic, deterministic_view, seed_everything


def test_seed_everything_repeats_random_sequence() -> None:
    seed_everything(42)
    first = [random.random() for _ in range(3)]
    seed_everything(42)
    assert [random.random() for _ in range(3)] == first


def test_deterministic_view_removes_wall_clock_fields() -> None:
    left = {"result": 1.23456789, "wall_ms": 10.0, "nested": {"timestamp": "a", "value": -0.0}}
    right = {"result": 1.23456780, "wall_ms": 99.0, "nested": {"timestamp": "b", "value": 0.0}}
    assert deterministic_view(left) == deterministic_view(right)


def test_assert_deterministic_runs_three_times() -> None:
    calls = 0

    def stable() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": {3, 1, 2}, "duration_ms": calls}

    result = assert_deterministic(stable, runs=3)
    assert result["value"] == {1, 2, 3}
    assert calls == 3


def test_assert_deterministic_rejects_different_output() -> None:
    counter = iter((1, 2, 3))
    with pytest.raises(AssertionError):
        assert_deterministic(lambda: next(counter), runs=3)


__author__ = "晨星"
