"""attention.py 不变量测试（模块 5，I1 - I11）。

核心是 I1：分块实现与朴素实现逐块大小等价。alpha 折算一旦漏掉，
这里的残差会从 1e-16 跳到 1e-2 量级。
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from worldforge.core.kernels.attention import attention_naive, attention_tiled
from worldforge.core.kernels.numeric import ATOL_STRICT
from worldforge.core.kernels.sampling import PCG32

BLOCK_CASES = [(1, 1), (2, 3), (3, 7), (7, 13), (16, 16), (64, 64), (40, 40), (1, 40)]
EQ_TOL = 1e-9
LSE_TOL = 1e-10


def _case(seed, Nq=9, Nk=17, d=5, dv=4):
    rng = PCG32(seed)
    Q = [[rng.gauss() for _ in range(d)] for _ in range(Nq)]
    K = [[rng.gauss() for _ in range(d)] for _ in range(Nk)]
    V = [[rng.gauss() for _ in range(dv)] for _ in range(Nk)]
    return Q, K, V


def _max_diff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b, strict=False) for x, y in zip(ra, rb, strict=False))


def test_I1_tiled_equals_naive():
    """I1 分块 vs 朴素等价（核心）：逐块大小 max abs 差 < 1e-9。"""
    Q, K, V = _case(20250101)
    On, _ = attention_naive(Q, K, V)
    for Br, Bc in BLOCK_CASES:
        Ot, _ = attention_tiled(Q, K, V, Br=Br, Bc=Bc)
        err = _max_diff(Ot, On)
        assert err < EQ_TOL, (Br, Bc, err)


def test_I2_lse_equivalence():
    """I2 LSE 等价（排除全 mask 行）：max abs 差 < 1e-10。"""
    Q, K, V = _case(20250102)
    _, Ln = attention_naive(Q, K, V)
    for Br, Bc in BLOCK_CASES:
        _, Lt = attention_tiled(Q, K, V, Br=Br, Bc=Bc)
        err = max(abs(a - b) for a, b in zip(Lt, Ln, strict=False))
        assert err < LSE_TOL, (Br, Bc, err)


def test_I3_output_in_convex_hull():
    """I3 凸组合：O[i][c] 落在 V 第 c 列的 min / max 之间（1e-9）。"""
    Q, K, V = _case(20250103)
    Ot, _ = attention_tiled(Q, K, V, Br=3, Bc=5)
    dv = len(V[0])
    for c in range(dv):
        lo = min(V[j][c] for j in range(len(V)))
        hi = max(V[j][c] for j in range(len(V)))
        for i in range(len(Q)):
            assert lo - EQ_TOL <= Ot[i][c] <= hi + EQ_TOL


def test_I4_equal_scores_give_uniform_weights():
    """I4 K 全 0（score 全 0）时 O[i] == mean_j V[j]（1e-10）。"""
    Q, _, V = _case(20250104)
    K = [[0.0] * len(Q[0]) for _ in range(len(V))]
    Ot, _ = attention_tiled(Q, K, V, Br=2, Bc=3)
    Nk = len(V)
    dv = len(V[0])
    mean = [math.fsum(V[j][c] for j in range(Nk)) / Nk for c in range(dv)]
    for i in range(len(Q)):
        for c in range(dv):
            assert abs(Ot[i][c] - mean[c]) < 1e-10


def test_I5_causal_equivalence_with_prefix_recompute():
    """I5 因果等价：O_causal[i] 等于只用前 i+1 个 key 重算的结果（1e-9）。"""
    Q, K, V = _case(20250105, Nq=8, Nk=8)
    Oc, _ = attention_tiled(Q, K, V, causal=True, Br=3, Bc=3)
    for i in range(len(Q)):
        m = i + 1
        oi, _ = attention_naive([Q[i]], K[:m], V[:m])
        for c in range(len(V[0])):
            assert abs(Oc[i][c] - oi[0][c]) < EQ_TOL


def test_I6_causal_upper_triangle_irrelevance():
    """I6 因果上三角无关性：扰动 K[j] (j > i) 后 O_causal[i] 不变（1e-12）。"""
    Q, K, V = _case(20250106, Nq=8, Nk=8)
    base, _ = attention_tiled(Q, K, V, causal=True, Br=3, Bc=3)
    K2 = [row[:] for row in K]
    for j in range(2, len(K2)):
        for t in range(len(K2[0])):
            K2[j][t] += 7.5
    after, _ = attention_tiled(Q, K2, V, causal=True, Br=3, Bc=3)
    for i in range(2):
        for c in range(len(V[0])):
            assert abs(base[i][c] - after[i][c]) < ATOL_STRICT


def test_I7_block_size_invariance():
    """I7 块大小无关性：不同 (Br, Bc) 之间彼此一致（含不整除的块大小）。"""
    Q, K, V = _case(20250107, Nq=13, Nk=31)
    outs = [attention_tiled(Q, K, V, Br=br, Bc=bc)[0] for br, bc in BLOCK_CASES]
    first = outs[0]
    for o in outs[1:]:
        assert _max_diff(o, first) < EQ_TOL


def test_I8_fully_masked_row_is_safe():
    """I8 全 mask 行：O[i] == 0、LSE[i] == -inf，绝不出现 nan。"""
    Q, K, V = _case(20250108, Nq=6, Nk=10)
    mask = [[True] * len(K) for _ in range(len(Q))]
    mask[2] = [False] * len(K)
    mask[5] = [False] * len(K)
    On, Ln = attention_naive(Q, K, V, mask=mask)
    Ot, Lt = attention_tiled(Q, K, V, mask=mask, Br=3, Bc=4)
    for i in (2, 5):
        assert all(v == 0.0 for v in On[i])
        assert all(v == 0.0 for v in Ot[i])
        assert Ln[i] == float("-inf")
        assert Lt[i] == float("-inf")
    for i in (0, 1, 3, 4):
        assert all(not math.isnan(v) for v in Ot[i])


def test_I9_scale_consistency():
    """I9 手动传 scale 与默认 1/sqrt(d) 在 d 相同时结果一致（1e-15）。"""
    Q, K, V = _case(20250109)
    d = len(Q[0])
    a, _ = attention_tiled(Q, K, V, scale=1.0 / math.sqrt(d), Br=4, Bc=4)
    b, _ = attention_tiled(Q, K, V, Br=4, Bc=4)
    assert _max_diff(a, b) < 1e-15


def test_I10_energy_conservation():
    """I10 能量守恒：任意固定 w 下 sum_c O[i][c] w_c 在 tiled / naive 间一致（1e-9）。"""
    Q, K, V = _case(20250110)
    rng = PCG32(999)
    dv = len(V[0])
    w = [rng.gauss() for _ in range(dv)]
    On, _ = attention_naive(Q, K, V)
    Ot, _ = attention_tiled(Q, K, V, Br=3, Bc=7)
    for i in range(len(Q)):
        en = math.fsum(On[i][c] * w[c] for c in range(dv))
        et = math.fsum(Ot[i][c] * w[c] for c in range(dv))
        assert abs(en - et) < EQ_TOL


def test_I11_batch_additivity():
    """I11 批处理可加性：Q 按行拆两半分别计算，与整体计算逐行一致（1e-9）。"""
    Q, K, V = _case(20250111, Nq=12, Nk=19)
    whole, _ = attention_tiled(Q, K, V, Br=5, Bc=5)
    head, _ = attention_tiled(Q[:5], K, V, Br=5, Bc=5)
    tail, _ = attention_tiled(Q[5:], K, V, Br=5, Bc=5)
    for i in range(5):
        for c in range(len(V[0])):
            assert abs(whole[i][c] - head[i][c]) < EQ_TOL
    for i in range(len(Q) - 5):
        for c in range(len(V[0])):
            assert abs(whole[i + 5][c] - tail[i][c]) < EQ_TOL


def test_masked_tiled_matches_naive():
    """带任意 mask 时（含不整除块大小）分块与朴素等价。"""
    Q, K, V = _case(20250112, Nq=11, Nk=23)
    rng = PCG32(777)
    mask = [[rng.next_float() > 0.35 for _ in range(len(K))] for _ in range(len(Q))]
    for row in mask:
        row[0] = True
    On, Ln = attention_naive(Q, K, V, mask=mask)
    for Br, Bc in ((1, 1), (3, 7), (4, 8), (16, 16), (64, 64)):
        Ot, Lt = attention_tiled(Q, K, V, mask=mask, Br=Br, Bc=Bc)
        assert _max_diff(Ot, On) < EQ_TOL, (Br, Bc)
        assert max(abs(a - b) for a, b in zip(Lt, Ln, strict=False)) < LSE_TOL, (Br, Bc)


def test_causal_tiled_matches_naive():
    """因果掩码下分块与朴素等价，且 LSE 也一致。"""
    Q, K, V = _case(20250113, Nq=15, Nk=15)
    On, Ln = attention_naive(Q, K, V, causal=True)
    for Br, Bc in ((1, 1), (2, 5), (7, 13), (16, 16)):
        Ot, Lt = attention_tiled(Q, K, V, causal=True, Br=Br, Bc=Bc)
        assert _max_diff(Ot, On) < EQ_TOL, (Br, Bc)
        assert max(abs(a - b) for a, b in zip(Lt, Ln, strict=False)) < LSE_TOL, (Br, Bc)


def test_invalid_block_size_rejected():
    """非正块大小必须显式报错，而不是静默产生空循环。"""
    Q, K, V = _case(20250114)
    try:
        attention_tiled(Q, K, V, Bc=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Bc = 0 should raise ValueError")
