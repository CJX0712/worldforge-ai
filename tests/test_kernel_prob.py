"""prob.py 不变量测试（模块 3，I1 - I13）。"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

import pytest

from worldforge.core.kernels.numeric import ATOL_STRICT
from worldforge.core.kernels.prob import (
    AllZeroProbabilityError,
    conditional_entropy,
    cross_entropy,
    entropy,
    js_divergence,
    kl_divergence,
    mutual_information,
    normalize,
    wasserstein_1d,
)
from worldforge.core.kernels.sampling import PCG32

KL_TOL = 1e-15
CONV_TOL = 1e-10


def _rand_p(n, seed):
    rng = PCG32(seed)
    return normalize([rng.next_float() + 0.05 for _ in range(n)])


def test_I1_kl_nonnegative():
    """I1 KL(p||q) >= -1e-15（200 组随机分布）。"""
    for s in range(200):
        p = _rand_p(6, 1000 + s)
        q = _rand_p(6, 2000 + s)
        assert kl_divergence(p, q) >= -KL_TOL


def test_I2_kl_self_is_zero():
    """I2 KL(p||p) == 0（1e-12）。"""
    for s in range(20):
        p = _rand_p(7, 3000 + s)
        assert abs(kl_divergence(p, p)) < ATOL_STRICT


def test_I3_kl_asymmetry_reference_value():
    """I3 KL 不对称，且 p=[0.9,0.1] 对 q=[0.5,0.5] 的参考值 0.3681。"""
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    forward = kl_divergence(p, q)
    backward = kl_divergence(q, p)
    assert abs(forward - 0.3681) < 1e-4
    assert abs(forward - backward) > 1e-3


def test_I4_js_bounded():
    """I4 0 <= JS <= ln2（base=e）；base=2 时上界为 1。"""
    for s in range(200):
        p = _rand_p(5, 4000 + s)
        q = _rand_p(5, 5000 + s)
        v = js_divergence(p, q)
        assert v >= -KL_TOL
        assert v <= math.log(2.0) + ATOL_STRICT
        v2 = js_divergence(p, q, base=2.0)
        assert v2 <= 1.0 + ATOL_STRICT


def test_I5_js_symmetric():
    """I5 JS 对称：abs(JS(p,q) - JS(q,p)) < 1e-12。"""
    for s in range(50):
        p = _rand_p(6, 6000 + s)
        q = _rand_p(6, 7000 + s)
        assert abs(js_divergence(p, q) - js_divergence(q, p)) < ATOL_STRICT


def test_I6_entropy_upper_bound():
    """I6 H(p) <= log(n)；均匀分布取等号。"""
    for s in range(50):
        p = _rand_p(9, 8000 + s)
        assert entropy(p) <= math.log(9) + ATOL_STRICT
        assert entropy(p) >= -ATOL_STRICT
    for n in (2, 5, 16):
        assert abs(entropy([1.0] * n) - math.log(n)) < ATOL_STRICT


def test_I7_gibbs_inequality():
    """I7 H(p,q) >= H(p)；q == p 时相等。"""
    for s in range(50):
        p = _rand_p(6, 9000 + s)
        q = _rand_p(6, 11000 + s)
        assert cross_entropy(p, q) - entropy(p) >= -ATOL_STRICT
        assert abs(cross_entropy(p, p) - entropy(p)) < ATOL_STRICT


def test_I8_mutual_information_nonnegative():
    """I8 I(X;Y) >= -1e-15。"""
    rng = PCG32(12000)
    for s in range(30):
        joint = [[rng.next_float() + 0.01 for _ in range(4)] for _ in range(3)]
        assert mutual_information(joint) >= -KL_TOL


def test_I9_independence_gives_zero_mi():
    """I9 joint = outer(px, py) 时 abs(I) < 1e-12。"""
    px = _rand_p(4, 13000)
    py = _rand_p(5, 14000)
    joint = [[a * b for b in py] for a in px]
    assert abs(mutual_information(joint)) < ATOL_STRICT


def test_I10_self_information_equals_entropy():
    """I10 I(X;X) == H(X)（joint = diag(px)）。"""
    px = _rand_p(5, 15000)
    joint = [[px[i] if i == j else 0.0 for j in range(5)] for i in range(5)]
    assert abs(mutual_information(joint) - entropy(px)) < ATOL_STRICT


def test_I11_wasserstein_two_point_masses():
    """I11 W1(delta_a, delta_b) == abs(a - b)（1e-12）。"""
    for a, b in ((0.0, 3.0), (-2.5, 1.5), (7.0, -1.0)):
        assert abs(wasserstein_1d([a], [1.0], [b], [1.0]) - abs(a - b)) < ATOL_STRICT


def test_I12_wasserstein_is_a_metric():
    """I12 W1 度量性质：非负、自反为 0、对称、三角不等式。"""
    rng = PCG32(16000)
    triples = []
    for _ in range(20):
        def make(seed_shift):
            r = PCG32(17000 + seed_shift)
            xs = [r.gauss() * 2.0 for _ in range(6)]
            ws = [r.next_float() + 0.1 for _ in range(6)]
            return xs, ws
        triples.append((make(0), make(1), make(2)))
    for (xp, wp), (xq, wq), (xr, wr) in triples:
        w_pq = wasserstein_1d(xp, wp, xq, wq)
        w_qp = wasserstein_1d(xq, wq, xp, wp)
        w_pr = wasserstein_1d(xp, wp, xr, wr)
        w_qr = wasserstein_1d(xq, wq, xr, wr)
        assert w_pq >= 0.0
        assert abs(wasserstein_1d(xp, wp, xp, wp)) < ATOL_STRICT
        assert abs(w_pq - w_qp) < ATOL_STRICT
        assert w_pr <= w_pq + w_qr + ATOL_STRICT


def test_I13_mi_equals_entropy_minus_conditional():
    """I13 I(X;Y) == H(X) - H(X|Y)（1e-10，独立实现交叉验证）。"""
    rng = PCG32(18000)
    for _ in range(20):
        joint = [[rng.next_float() + 0.01 for _ in range(4)] for _ in range(3)]
        n = 3
        m = 4
        flat = [v for row in joint for v in row]
        jp = normalize(flat)
        px = [math.fsum(jp[i * m + j] for j in range(m)) for i in range(n)]
        lhs = mutual_information(joint)
        rhs = entropy(px) - conditional_entropy(joint)
        assert abs(lhs - rhs) < CONV_TOL


def test_normalize_rejects_all_zero():
    """全零输入抛 AllZeroProbabilityError，不静默返回均匀分布。"""
    with pytest.raises(AllZeroProbabilityError):
        normalize([0.0, 0.0, 0.0])
