"""rl.py 不变量测试（模块 8，I1 - I14）。"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

import pytest

from worldforge.core.kernels.numeric import ATOL_STRICT
from worldforge.core.kernels.rl import (
    compute_gae,
    normalize_advantages,
    policy_entropy,
    ppo_clip_loss,
    value_clip_loss,
)
from worldforge.core.kernels.sampling import PCG32

GAMMA = 0.99
EPS = 0.2


def _rollout(T, seed, done_at=None):
    rng = PCG32(seed)
    r = [rng.gauss() + 1.0 for _ in range(T)]
    v = [rng.gauss() for _ in range(T + 1)]
    dones = [False] * T
    if done_at is not None:
        dones[done_at] = True
    return r, v, dones


def test_I1_lambda_zero_is_td_residual():
    """I1 lambda = 0 时 A_t == delta_t（1e-12）。"""
    r, v, dones = _rollout(12, 20250101)
    adv, _ = compute_gae(r, v, dones, gamma=GAMMA, lam=0.0)
    for t in range(12):
        delta = r[t] + GAMMA * v[t + 1] * (0.0 if dones[t] else 1.0) - v[t]
        assert abs(adv[t] - delta) < ATOL_STRICT


def test_I2_lambda_one_is_mc_return_minus_baseline():
    """I2 lambda = 1 时 A_t + V_t == R_t（无 done 截断的闭式 MC 回报，1e-10）。"""
    r, v, dones = _rollout(12, 20250102)
    adv, _ = compute_gae(r, v, dones, gamma=GAMMA, lam=1.0)
    T = len(r)
    for t in range(T):
        ret = math.fsum(GAMMA ** ell * r[t + ell] for ell in range(T - t)) + GAMMA ** (T - t) * v[T]
        assert abs((adv[t] + v[t]) - ret) < 1e-10


def test_I3_advantages_and_returns_self_consistent():
    """I3 对任意 lambda：A_t + V_t == returns_t（1e-12）。"""
    for lam in (0.0, 0.3, 0.8, 0.95, 1.0):
        r, v, dones = _rollout(10, 20250103)
        adv, ret = compute_gae(r, v, dones, gamma=GAMMA, lam=lam)
        for t in range(10):
            assert abs((adv[t] + v[t]) - ret[t]) < ATOL_STRICT


def test_I4_episode_boundary_truncation():
    """I4 dones[t0] = True 时，A_{t < t0} 与在 t0 处切开独立计算的 A 逐位一致（1e-12）。"""
    r, v, dones = _rollout(10, 20250104, done_at=6)
    adv_full, _ = compute_gae(r, v, dones, gamma=GAMMA, lam=0.95)
    adv_head, _ = compute_gae(r[:7], v[:8], dones[:7], gamma=GAMMA, lam=0.95)
    for t in range(7):
        assert abs(adv_full[t] - adv_head[t]) < ATOL_STRICT
    # 边界之后不得把回报漏过来：A_6 只含 r_6
    assert abs(adv_full[6] + v[6] - r[6]) < ATOL_STRICT


def test_I5_terminal_no_bootstrap():
    """I5 dones[T-1] = True 时 A_{T-1} + V_{T-1} == r_{T-1}（1e-12）。"""
    r, v, _ = _rollout(9, 20250105)
    dones = [False] * 9
    dones[8] = True
    adv, _ = compute_gae(r, v, dones, gamma=GAMMA, lam=0.95)
    assert abs(adv[8] + v[8] - r[8]) < ATOL_STRICT


def test_I6_normalization_moments():
    """I6 归一化后有效步上 abs(mean) < 1e-12、abs(std - 1) < 1e-12。"""
    rng = PCG32(20250106)
    adv = [rng.gauss() * 3.0 + 1.0 for _ in range(30)]
    out = normalize_advantages(adv)
    mean = math.fsum(out) / len(out)
    var = math.fsum((v - mean) ** 2 for v in out) / len(out)
    assert abs(mean) < ATOL_STRICT
    assert abs(math.sqrt(var) - 1.0) < ATOL_STRICT
    # padding 步不参与统计，也不被改写
    padded = adv + [0.0] * 10
    mask = [True] * 30 + [False] * 10
    out2 = normalize_advantages(padded, mask=mask)
    assert out2[30:] == [0.0] * 10
    m2 = math.fsum(out2[:30]) / 30
    assert abs(m2) < ATOL_STRICT


def test_I7_ratio_one_gives_advantage():
    """I7 logp == logp_old 时 L^CLIP == A（1e-12）。"""
    rng = PCG32(20250107)
    logp = [rng.gauss() for _ in range(20)]
    adv = [rng.gauss() * 2.0 for _ in range(20)]
    loss, ratios, clip_frac = ppo_clip_loss(logp, logp, adv)
    for ell, a in zip(loss, adv, strict=False):
        assert abs(ell - a) < ATOL_STRICT
    for r in ratios:
        assert abs(r - 1.0) < ATOL_STRICT
    assert clip_frac == 0.0


def test_I8_clip_saturation_values():
    """I8 A > 0 且 r > 1+eps 时 L == (1+eps)A；A < 0 且 r < 1-eps 时 L == (1-eps)A。"""
    logp_old = [0.0] * 4
    logp = [math.log(2.0), math.log(0.5), math.log(2.0), math.log(0.5)]
    adv = [1.0, -1.0, -1.0, 1.0]
    loss, ratios, _ = ppo_clip_loss(logp, logp_old, adv, eps=EPS)
    assert abs(loss[0] - (1.0 + EPS) * 1.0) < ATOL_STRICT
    assert abs(loss[1] - (1.0 - EPS) * (-1.0)) < ATOL_STRICT
    # A > 0 且 r 变小：取未裁剪值 r*A（更小，悲观）
    assert abs(loss[3] - 0.5 * 1.0) < ATOL_STRICT
    # A < 0 且 r 变大：r*A 更负，取它
    assert abs(loss[2] - 2.0 * (-1.0)) < ATOL_STRICT


def test_I9_pessimistic_lower_bound():
    """I9 L^CLIP <= r * A + 1e-12 恒成立（符号写反会立刻违反）。"""
    rng = PCG32(20250108)
    for _ in range(200):
        logp = [rng.gauss() * 1.5 for _ in range(8)]
        logp_old = [rng.gauss() * 1.5 for _ in range(8)]
        adv = [rng.gauss() * 3.0 for _ in range(8)]
        loss, ratios, _ = ppo_clip_loss(logp, logp_old, adv, eps=EPS)
        for ell, r, a in zip(loss, ratios, adv, strict=False):
            assert ell <= r * a + 1e-12


def test_I10_ratio_identity():
    """I10 r == exp(logp - logp_old)，且与概率域比值一致（1e-12）。"""
    rng = PCG32(20250109)
    logp = [rng.gauss() for _ in range(20)]
    logp_old = [rng.gauss() for _ in range(20)]
    _, ratios, _ = ppo_clip_loss(logp, logp_old, [0.0] * 20)
    for r, a, b in zip(ratios, logp, logp_old, strict=False):
        assert abs(r - math.exp(a - b)) < 1e-12
        assert abs(r - (math.exp(a) / math.exp(b))) < 1e-12


def test_I11_value_clip_no_op_within_eps():
    """I11 abs(v - v_old) <= eps 时 L^VF == (v - v_targ)^2（1e-12）。"""
    rng = PCG32(20250110)
    v_old = [rng.gauss() for _ in range(20)]
    v = [a + EPS * 0.5 for a in v_old]
    v_targ = [rng.gauss() for _ in range(20)]
    loss = value_clip_loss(v, v_old, v_targ, eps=EPS)
    for ell, a, t in zip(loss, v, v_targ, strict=False):
        assert abs(ell - (a - t) ** 2) < ATOL_STRICT


def test_I12_value_clip_pessimistic():
    """I12 L^VF >= (v - v_targ)^2 - 1e-12 恒成立。"""
    rng = PCG32(20250111)
    for _ in range(200):
        v_old = [rng.gauss() for _ in range(8)]
        v = [a + rng.gauss() for a in v_old]
        v_targ = [rng.gauss() for _ in range(8)]
        loss = value_clip_loss(v, v_old, v_targ, eps=EPS)
        for ell, a, t in zip(loss, v, v_targ, strict=False):
            assert ell >= (a - t) ** 2 - 1e-12


def test_I13_entropy_range():
    """I13 0 <= S <= log(|A|)；均匀分布时 abs(S - log(|A|)) < 1e-12。"""
    rng = PCG32(20250112)
    for n in (2, 5, 11):
        for _ in range(20):
            logits = [rng.gauss() * 2.0 for _ in range(n)]
            s = policy_entropy(logits)
            assert -1e-12 <= s <= math.log(n) + 1e-12
        assert abs(policy_entropy([0.0] * n) - math.log(n)) < ATOL_STRICT


def test_I14_clip_fraction_is_a_valid_fraction():
    """I14 clip_frac 落在 [0, 1]，且 logp == logp_old 时为 0（监控量）。"""
    rng = PCG32(20250113)
    logp_old = [0.0] * 50
    logp = [rng.gauss() for _ in range(50)]
    _, _, frac = ppo_clip_loss(logp, logp_old, [0.0] * 50, eps=EPS)
    assert 0.0 <= frac <= 1.0
    _, _, frac0 = ppo_clip_loss(logp_old, logp_old, [0.0] * 50, eps=EPS)
    assert frac0 == 0.0


def test_values_length_validation():
    """values 长度既不是 T 也不是 T+1 时必须显式报错。"""
    with pytest.raises(ValueError):
        compute_gae([1.0, 2.0], [0.0, 0.0, 0.0, 0.0], [False, False])
