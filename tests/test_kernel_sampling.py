"""sampling.py 不变量测试（模块 4，I1 - I13）。"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

import pytest

from worldforge.core.kernels.numeric import ATOL_STRICT, softmax
from worldforge.core.kernels.prob import entropy, normalize
from worldforge.core.kernels.sampling import (
    PCG32,
    InvalidNucleusProbError,
    TemperatureZeroError,
    apply_temperature,
    beam_search,
    sample_categorical,
    top_k_filter,
    top_k_indices,
    top_p_filter,
)

MASTER_SEED = 20250101
N_SAMPLES = 100000


def _logits(seed, n=32):
    rng = PCG32(seed)
    return [rng.gauss() * 2.0 for _ in range(n)]


def test_I1_filters_keep_normalization():
    """I1 每次过滤后概率和仍为 1（1e-12）。"""
    for s in range(20):
        z = _logits(MASTER_SEED + s)
        for filtered in (top_k_filter(z, 5), top_p_filter(z, 0.9),
                         top_p_filter(top_k_filter(z, 5), 0.9)):
            p = softmax(filtered)
            assert abs(math.fsum(p) - 1.0) < ATOL_STRICT


def test_I2_top_k_cardinality():
    """I2 top-k 基数：无并列时正好 k 个位置非 -inf，且就是最大的 k 个。"""
    for s in range(20):
        z = _logits(MASTER_SEED + s)
        for k in (1, 3, 7):
            f = top_k_filter(z, k)
            alive = [i for i, v in enumerate(f) if v != float("-inf")]
            assert len(alive) == k
            order = sorted(range(len(z)), key=lambda i: (-z[i], i))
            assert alive == sorted(order[:k])


def test_I3_top_p_minimality():
    """I3 top-p 极小性：保留前缀累积概率 >= p - 1e-9，去掉最后一个则 < p。"""
    for s in range(20):
        z = _logits(MASTER_SEED + s)
        for p in (0.5, 0.9, 0.95, 1.0):
            f = top_p_filter(z, p)
            probs = softmax(z)
            order = sorted(range(len(z)), key=lambda i: (-z[i], i))
            keep = [i for i in order if f[i] != float("-inf")]
            assert keep == order[:len(keep)]
            cum = math.fsum(probs[i] for i in keep)
            assert cum >= p - 1e-9
            if len(keep) > 1:
                assert cum - probs[keep[-1]] < p


def test_I4_temperature_monotonic_entropy():
    """I4 T1 < T2 推出 H(softmax(z/T1)) <= H(softmax(z/T2)) + 1e-9。"""
    for s in range(10):
        z = _logits(MASTER_SEED + s)
        ts = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]
        hs = [entropy(softmax(apply_temperature(z, t))) for t in ts]
        for a, b in zip(hs, hs[1:], strict=False):
            assert a <= b + 1e-9, (ts, hs)


def test_I5_temperature_limit_is_argmax():
    """I5 T = 1e-8：max(softmax(z/T)) > 1 - 1e-6，且 argmax 位置与 argmax(z) 一致。"""
    for s in range(10):
        z = _logits(MASTER_SEED + s)
        p = softmax(apply_temperature(z, 1e-8))
        assert max(p) > 1.0 - 1e-6
        assert max(range(len(z)), key=lambda i: (p[i], -i)) == \
            max(range(len(z)), key=lambda i: (z[i], -i))


def test_I6_temperature_one_identity():
    """I6 T = 1 恒等：softmax(z/1.0) 与 softmax(z) 逐位相等。"""
    for s in range(10):
        z = _logits(MASTER_SEED + s)
        assert softmax(apply_temperature(z, 1.0)) == softmax(z)


def test_I7_beam_scores_descending():
    """I7 beam 分数降序。"""
    def step(ids):
        t = len(ids)
        return [-1.0 - 0.05 * abs(v - (t % 4)) for v in range(4)]

    beams = beam_search(step, B=3, max_len=4, eos_id=99)
    assert len(beams) <= 3
    for a, b in zip(beams, beams[1:], strict=False):
        assert a[0] >= b[0]


def test_I8_beam_width_one_is_greedy():
    """I8 B = 1 时输出序列 == 逐步 argmax 的序列（同 tie-break）。"""
    def step(ids):
        t = len(ids)
        return [-1.0 - 0.05 * abs(v - (t % 4)) for v in range(4)]

    max_len = 5
    beams = beam_search(step, B=1, max_len=max_len, eos_id=99)
    greedy = []
    for _t in range(max_len):
        lp = step(greedy)
        greedy.append(max(range(4), key=lambda v: (lp[v], -v)))
    assert beams[0][1] == greedy


def test_I9_rng_reproducible():
    """I9 同 seed 两次实例化的 next_u32 序列逐位相等。"""
    a = PCG32(MASTER_SEED)
    b = PCG32(MASTER_SEED)
    sa = [a.next_u32() for _ in range(1000)]
    sb = [b.next_u32() for _ in range(1000)]
    assert sa == sb
    assert PCG32(MASTER_SEED + 1).next_u32() != sa[0]
    assert PCG32(MASTER_SEED, stream=1).next_u32() != sa[0]


def test_I10_rng_uniformity():
    """I10 n = 100000：abs(mean - 0.5) < 5e-3、abs(var - 1/12) < 5e-3、卡方检验。"""
    rng = PCG32(MASTER_SEED)
    xs = [rng.next_float() for _ in range(N_SAMPLES)]
    assert all(0.0 <= v < 1.0 for v in xs)
    mean = math.fsum(xs) / N_SAMPLES
    var = math.fsum((v - mean) ** 2 for v in xs) / N_SAMPLES
    assert abs(mean - 0.5) < 5e-3, mean
    assert abs(var - 1.0 / 12.0) < 5e-3, var
    bins = [0] * 100
    for v in xs:
        bins[min(int(v * 100), 99)] += 1
    expect = N_SAMPLES / 100.0
    chi2 = math.fsum((b - expect) ** 2 / expect for b in bins)
    assert chi2 < 150.0, chi2


def test_I11_rng_no_short_period():
    """I11 无短周期：1e5 次输出中重复值个数符合泊松（期望约 1.16），远小于上限。"""
    rng = PCG32(MASTER_SEED)
    seen = set()
    dup = 0
    for _ in range(N_SAMPLES):
        v = rng.next_u32()
        if v in seen:
            dup += 1
        else:
            seen.add(v)
    assert dup <= 10, dup


def test_I12_sample_index_in_range():
    """I12 采样索引恒满足 0 <= idx < V。"""
    rng = PCG32(MASTER_SEED)
    probs = normalize([rng.next_float() + 0.01 for _ in range(16)])
    for _ in range(2000):
        idx = sample_categorical(probs, rng, T=0.7, k=5, p=0.9)
        assert 0 <= idx < 16


def test_I13_gaussian_moments():
    """I13 PCG32.gauss 取 1e5 样本：均值 0、标准差 1、偏度近 0（1e-2）。"""
    rng = PCG32(MASTER_SEED)
    xs = [rng.gauss() for _ in range(N_SAMPLES)]
    n = float(N_SAMPLES)
    mean = math.fsum(xs) / n
    var = math.fsum((v - mean) ** 2 for v in xs) / n
    std = math.sqrt(var)
    assert abs(mean) < 0.01, mean
    assert abs(std - 1.0) < 0.01, std
    m3 = math.fsum((v - mean) ** 3 for v in xs) / n
    assert abs(m3 / std ** 3) < 0.05


def test_top_k_indices_deterministic_tie_break():
    """并列时索引小者优先，两次调用逐位相同。"""
    v = [1.0, 2.0, 2.0, 1.0]
    assert top_k_indices(v, 2) == [1, 2]
    assert top_k_indices(v, 2) == top_k_indices(v, 2)
    assert top_k_indices(v, 8) == [1, 2, 0, 3]


def test_error_paths():
    """T <= 0 与非法 nucleus 概率必须显式抛错。"""
    with pytest.raises(TemperatureZeroError):
        apply_temperature([1.0, 2.0], 0.0)
    with pytest.raises(InvalidNucleusProbError):
        top_p_filter([1.0, 2.0], 0.0)
