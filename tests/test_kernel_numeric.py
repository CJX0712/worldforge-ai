"""numeric.py 不变量测试（模块 1，I1 - I6）。

容差全部从 numeric 模块集中导入，测试里不硬编码魔数。
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from worldforge.core.kernels.numeric import (
    ATOL_STRICT,
    EPS,
    LOG_MAX,
    LOG_TINY,
    clamp,
    log_softmax,
    logsumexp,
    safe_exp,
    safe_log,
    softmax,
)

SEED_CASES = [
    [1.0, 2.0, 3.0, 4.0],
    [0.0, 0.0, 0.0],
    [-5.0, 0.0, 5.0],
    [1e3, 1e3 + 1.0, 1e3 + 2.0],
    [-1e3, 0.0, 1e3],
    [0.1, -0.1, 0.05, -0.05, 0.0],
]

# I4 的用例排除跨度达 2e3 的输入：那里最小的 softmax 分量下溢为 0，log(0) 无定义。
I4_CASES = SEED_CASES[:4] + [SEED_CASES[5]]


def test_I1_softmax_sums_to_one():
    """I1 行和为一：abs(fsum(softmax(x)) - 1.0) < 1e-12。"""
    for x in SEED_CASES:
        p = softmax(x)
        assert abs(math.fsum(p) - 1.0) < ATOL_STRICT
    assert softmax([]) == []


def test_I2_shift_invariance():
    """I2 平移不变：max abs(softmax(x + c) - softmax(x)) < 1e-12，c = 1e3 / -1e3。"""
    for x in SEED_CASES:
        base = softmax(x)
        for c in (1e3, -1e3):
            shifted = softmax([v + c for v in x])
            err = max(abs(a - b) for a, b in zip(shifted, base))
            assert err < ATOL_STRICT, (x, c, err)


def test_I2b_large_shift_ulp_limit():
    """大常数平移：c = 1e6 时容差放宽到 1e-10。

    float64 在 1e6 量级的 ulp 已是 1.16e-10，x_i + c 本身就无法精确表示，
    平移不变性只能到 ulp 量级；对整数值输入实测为逐位相等（err == 0.0）。
    """
    for x in SEED_CASES:
        base = softmax(x)
        shifted = softmax([v + 1e6 for v in x])
        err = max(abs(a - b) for a, b in zip(shifted, base))
        assert err < 1e-10, (x, err)
    for x in ([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0], [-5.0, 0.0, 5.0]):
        assert softmax(x) == softmax([v + 1e6 for v in x])


def test_I3_logsumexp_bounds():
    """I3 LSE 夹逼：max(x) <= lse(x) <= max(x) + log(n)。"""
    for x in SEED_CASES:
        lse = logsumexp(x)
        m = max(x)
        assert m - ATOL_STRICT <= lse <= m + math.log(len(x)) + ATOL_STRICT
    assert logsumexp([]) == float("-inf")
    assert logsumexp([float("-inf")] * 3) == float("-inf")
    assert logsumexp([1.0, float("inf")]) == float("inf")


def test_I4_log_softmax_consistency():
    """I4 对数一致：max abs(log(softmax(x)_i) - log_softmax(x)_i) < 1e-12。

    只在 softmax 未下溢到 0 的量级上断言：下溢后 log(0) 无定义，
    这正是 log_softmax 存在的理由（它永远有定义）。
    """
    for x in I4_CASES:
        p = softmax(x)
        ls = log_softmax(x)
        err = max(abs(math.log(a) - b) for a, b in zip(p, ls))
        assert err < ATOL_STRICT, (x, err)


def test_I5_overflow_immunity():
    """I5 溢出免疫：x = [1000]*4 时 lse == 1000 + log(4)，softmax == [0.25]*4，全程无 inf。"""
    x = [1000.0] * 4
    lse = logsumexp(x)
    assert abs(lse - (1000.0 + math.log(4.0))) < ATOL_STRICT
    p = softmax(x)
    for v in p:
        assert abs(v - 0.25) < ATOL_STRICT
        assert not math.isinf(v) and not math.isnan(v)
    # 巨大的负向偏移也不能产生 nan
    p2 = softmax([-1e6, -1e6 + 1.0])
    assert all(not math.isnan(v) for v in p2)


def test_I6_strict_monotonicity():
    """I6 严格保序：x_i > x_j 推出 softmax(x)_i > softmax(x)_j（严格，无等号退化）。"""
    x = [1.0, 1.5, 2.0, 0.5, 2.5]
    p = softmax(x)
    for i in range(len(x)):
        for j in range(len(x)):
            if x[i] > x[j]:
                assert p[i] > p[j], (i, j, p[i], p[j])


def test_I5b_all_masked_row_degenerates_to_uniform():
    """全 -inf 行退化为均匀分布，绝不为 nan。"""
    p = softmax([float("-inf")] * 5)
    assert all(abs(v - 0.2) < ATOL_STRICT for v in p)
    ls = log_softmax([float("-inf")] * 5)
    assert all(abs(v + math.log(5.0)) < ATOL_STRICT for v in ls)


def test_safe_exp_and_safe_log_clamps():
    """safe_exp / safe_log 的钳位边界与不产生异常。"""
    assert safe_exp(LOG_TINY - 1.0) == 0.0
    assert safe_exp(LOG_MAX + 1.0) == float("inf")
    assert safe_exp(float("-inf")) == 0.0
    assert abs(safe_exp(0.0) - 1.0) < EPS
    assert safe_log(0.0) == LOG_TINY
    assert safe_log(-1.0) == LOG_TINY
    assert abs(safe_log(math.e) - 1.0) < ATOL_STRICT
    # 直接 math.exp(1000) 会抛 OverflowError，safe_exp 不会
    assert safe_exp(1000.0) == float("inf")


def test_clamp_interval():
    """clamp 的输出恒落在闭区间内。"""
    assert clamp(5.0, 0.0, 1.0) == 1.0
    assert clamp(-5.0, 0.0, 1.0) == 0.0
    assert clamp(0.5, 0.0, 1.0) == 0.5
