"""moe.py 不变量测试（模块 6，I1 - I13）。

重点：I8 / I9 用的是**可证**的 sum_i f_i^2 下界与 E*max(P) 上界。
规格书里已实测确认 `E * sum_i f_i P_i >= 1` 不成立（20000 组随机配置最小 0.976），
因此这里绝不出现 `>= 1` 的断言。
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from worldforge.core.kernels.moe import load_balance_loss, moe_forward, route_topk, router_z_loss
from worldforge.core.kernels.numeric import ATOL_STRICT, softmax
from worldforge.core.kernels.sampling import PCG32

ALPHA = 0.01


def _routing_batch(E, N, seed):
    """用真实 argmax（top-1）路由产生一组 (freq_counts, P)。"""
    rng = PCG32(seed)
    counts = [0] * E
    psum = [0.0] * E
    for _ in range(N):
        h = [rng.gauss() * 1.5 for _ in range(E)]
        p = softmax(h)
        top = max(range(E), key=lambda i: (h[i], -i))
        counts[top] += 1
        for i in range(E):
            psum[i] += p[i]
    return counts, [v / N for v in psum]


def test_I1_gates_sum_to_one():
    """I1 每个 token 的门控权重和为 1（1e-12）。"""
    rng = PCG32(20250101)
    for _ in range(50):
        h = [rng.gauss() * 3.0 for _ in range(8)]
        for k in (1, 2, 3, 8):
            _, gates = route_topk(h, k)
            assert abs(math.fsum(gates) - 1.0) < ATOL_STRICT


def test_I2_topk_cardinality_and_order():
    """I2 len(idx) == k、互异、且按 h 降序。"""
    h = [1.0, 5.0, 3.0, 5.0, 2.0]
    idx, _ = route_topk(h, 3)
    assert len(idx) == 3
    assert len(set(idx)) == 3
    assert [h[i] for i in idx] == sorted((h[i] for i in idx), reverse=True)


def test_I3_gate_order_monotonic():
    """I3 权重排序单调：h_a > h_b 且都在 S 内，则 g_a > g_b。"""
    rng = PCG32(20250102)
    for _ in range(50):
        h = [rng.gauss() * 2.0 for _ in range(8)]
        idx, gates = route_topk(h, 4)
        for a in range(len(idx)):
            for b in range(len(idx)):
                if h[idx[a]] > h[idx[b]]:
                    assert gates[a] > gates[b]


def test_I4_deterministic_tie_break():
    """I4 确定性并列：logit 相等时索引小者优先。"""
    h = [2.0, 2.0, 2.0, 1.0]
    idx, gates = route_topk(h, 2)
    assert idx == [0, 1]
    assert route_topk(h, 2) == (idx, gates)
    assert abs(gates[0] - 0.5) < ATOL_STRICT


def test_I5_count_conservation():
    """I5 计数守恒：sum(freq) + dropped == B * k（整数精确）。"""
    rng = PCG32(20250103)
    E, B, d = 4, 24, 5
    x = [[rng.gauss() for _ in range(d)] for _ in range(B)]
    Wg = [[rng.gauss() for _ in range(d)] for _ in range(E)]
    experts = [(lambda i: (lambda v: [c * (1.0 + i) for c in v]))(i) for i in range(E)]
    for k in (1, 2, 3):
        for cf in (0.25, 0.5, 1.0, 2.0):
            _, _, st = moe_forward(x, experts, Wg, k=k, capacity_factor=cf)
            assert math.fsum(st["freq"]) + st["dropped"] == B * k


def test_I6_capacity_constraint():
    """I6 max(freq) <= capacity（整数精确）。"""
    rng = PCG32(20250104)
    E, B, d = 5, 40, 3
    x = [[rng.gauss() for _ in range(d)] for _ in range(B)]
    Wg = [[rng.gauss() for _ in range(d)] for _ in range(E)]
    experts = [(lambda v: v) for _ in range(E)]
    for cf in (0.3, 0.7, 1.25, 3.0):
        _, _, st = moe_forward(x, experts, Wg, k=2, capacity_factor=cf)
        assert max(st["freq"]) <= st["capacity"]
        assert st["capacity"] >= 1


def test_I7_perfect_balance_gives_one():
    """I7 人为构造的均匀 f、P 下 loss / alpha == 1（1e-12）。

    注意：这只对构造出来的均衡点成立，不是不等式下界。
    """
    for E in (2, 4, 8, 16):
        f = [10] * E
        P = [1.0 / E] * E
        loss = load_balance_loss(f, P, ALPHA)
        assert abs(loss / ALPHA - 1.0) < ATOL_STRICT, (E, loss / ALPHA)


def test_I8_provable_lower_bound():
    """I8 E*sum f P >= sum f_i^2（1e-9），且 sum f_i^2 >= 1/E（真实 argmax 路由）。"""
    worst_slack = float("inf")
    for seed in range(40):
        for E in (2, 3, 5, 8):
            counts, P = _routing_batch(E, 60, 20250105 + seed * 17 + E)
            total = math.fsum(counts)
            f = [c / total for c in counts]
            lhs = E * math.fsum(f[i] * P[i] for i in range(E))
            sq = math.fsum(v * v for v in f)
            worst_slack = min(worst_slack, lhs - sq)
            assert lhs - sq >= -1e-9, (E, lhs, sq)
            assert sq >= 1.0 / E - 1e-12
    assert worst_slack > -1e-9


def test_I9_upper_bound():
    """I9 E*sum f P <= E*max(P) 且 <= E（1e-12）。"""
    for seed in range(30):
        for E in (2, 4, 6):
            counts, P = _routing_batch(E, 50, 20250106 + seed * 13 + E)
            total = math.fsum(counts)
            f = [c / total for c in counts]
            lhs = E * math.fsum(f[i] * P[i] for i in range(E))
            assert lhs <= E * max(P) + 1e-12
            assert lhs <= E + 1e-12


def test_I10_f_and_P_normalized():
    """I10 abs(sum f - 1) 与 abs(sum P - 1) 均 < 1e-12。"""
    counts, P = _routing_batch(6, 80, 20250107)
    total = math.fsum(counts)
    f = [c / total for c in counts]
    assert abs(math.fsum(f) - 1.0) < ATOL_STRICT
    assert abs(math.fsum(P) - 1.0) < ATOL_STRICT


def test_I11_identity_experts_reproduce_input():
    """I11 experts 全为恒等映射且无丢弃时 y[b] == x[b]（1e-12）。"""
    rng = PCG32(20250108)
    E, B, d = 4, 20, 6
    x = [[rng.gauss() for _ in range(d)] for _ in range(B)]
    Wg = [[rng.gauss() for _ in range(d)] for _ in range(E)]
    y, _, st = moe_forward(x, [(lambda v: v) for _ in range(E)], Wg,
                           k=2, capacity_factor=4.0)
    assert st["dropped"] == 0
    for b in range(B):
        for c in range(d):
            assert abs(y[b][c] - x[b][c]) < ATOL_STRICT


def test_I12_sufficient_capacity_has_no_drops():
    """I12 容量充足时 dropped == 0；容量不足时按 I5 精确记账。"""
    E, B = 4, 32
    x = [[1.0 if j == (b % E) else 0.0 for j in range(E)] for b in range(B)]
    Wg = [[10.0 if i == j else 0.0 for j in range(E)] for i in range(E)]
    experts = [(lambda v: [c * 2.0 for c in v]) for _ in range(E)]
    _, _, st = moe_forward(x, experts, Wg, k=1, capacity_factor=1.0)
    assert st["dropped"] == 0
    assert st["freq"] == [8] * E
    _, _, st2 = moe_forward(x, experts, Wg, k=1, capacity_factor=0.5)
    assert st2["capacity"] == 4
    assert st2["dropped"] == B - 4 * E


def test_I13_probabilities_are_valid():
    """I13 0 <= P_i <= 1 且 f_i >= 0。"""
    counts, P = _routing_batch(5, 40, 20250109)
    total = math.fsum(counts)
    f = [c / total for c in counts]
    assert all(0.0 <= v <= 1.0 for v in P)
    assert all(v >= 0.0 for v in f)


def test_no_training_means_zero_aux_loss():
    """training=False 时辅助损失为 0。"""
    rng = PCG32(20250110)
    E, B, d = 3, 12, 4
    x = [[rng.gauss() for _ in range(d)] for _ in range(B)]
    Wg = [[rng.gauss() for _ in range(d)] for _ in range(E)]
    _, aux, _ = moe_forward(x, [(lambda v: v) for _ in range(E)], Wg, training=False)
    assert aux == 0.0


def test_router_z_loss_nonnegative():
    """router z-loss 非负，且全 -inf 输入返回 0 而不是 nan。"""
    assert router_z_loss([1.0, 2.0, 3.0]) >= 0.0
    assert router_z_loss([float("-inf")] * 4) == 0.0
