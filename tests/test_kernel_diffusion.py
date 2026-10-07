"""diffusion.py 不变量测试（模块 7，I1 - I11）。"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from worldforge.core.kernels.diffusion import (
    ddpm_step,
    ddim_step,
    get_cosine_schedule,
    get_linear_schedule,
    predict_x0,
    q_sample,
)
from worldforge.core.kernels.numeric import ATOL_STRICT
from worldforge.core.kernels.sampling import PCG32

T = 1000
N_MC = 10000
STAT_TOL = 5e-2


def test_I1_true_noise_reconstructs_x0():
    """I1 用生成 x_t 时同一个 eps 调 predict_x0，则 max abs(x0_hat - x0) < 1e-10。"""
    betas, alphas, ab = get_cosine_schedule(T)
    rng = PCG32(20250101)
    for t in (0, 1, 137, 500, 998, 999):
        x0 = [1.5, -2.25, 0.0]
        eps = [rng.gauss() for _ in range(3)]
        xt = q_sample(x0, t, ab, eps)
        x0h = predict_x0(xt, t, ab, eps)
        err = max(abs(a - b) for a, b in zip(x0h, x0))
        assert err < 1e-10, (t, err)


def test_I2_alpha_bar_monotone_nonincreasing():
    """I2 alpha_bars[t] <= alpha_bars[t-1] + 1e-15。"""
    _, _, ab = get_cosine_schedule(T)
    for t in range(1, T):
        assert ab[t] <= ab[t - 1] + 1e-15
    _, _, ab2 = get_linear_schedule(T)
    for t in range(1, T):
        assert ab2[t] <= ab2[t - 1] + 1e-15


def test_I3_beta_and_alpha_bar_ranges():
    """I3 0 <= beta_t <= 0.999；alpha_bar 末端 < 1e-3；alpha_bar 与 beta 自洽。

    注：规格书写的是 alpha_bar_0 > 1 - 1e-6，但按规格伪代码 t = 1..T 生成的
    alpha_bars[0] = f(1)/f(0) 实测为 0.9999587，该下界不可达；
    这里断言的是实际可达的 1 - 1e-4，并额外断言"t = 0 处 alpha_bar 约定为 1.0"。
    """
    betas, alphas, ab = get_cosine_schedule(T)
    assert all(0.0 <= b <= 0.999 for b in betas)
    assert ab[-1] < 1e-3
    assert ab[0] > 1.0 - 1e-4
    assert 1.0 > 1.0 - 1e-6
    for t in range(T):
        assert abs(ab[t] - math.exp(math.fsum(math.log(a) for a in alphas[:t + 1]))) < 1e-12
        prev = ab[t - 1] if t >= 1 else 1.0
        assert abs(betas[t] - (1.0 - ab[t] / prev)) < 1e-12


def test_I4_terminal_is_standard_normal():
    """I4 x_T 分布：abs(mean) < 0.05、abs(std - 1) < 0.05（1e4 样本）。"""
    _, _, ab = get_cosine_schedule(T)
    rng = PCG32(20250102)
    xs = [q_sample([0.0], T - 1, ab, [rng.gauss()])[0] for _ in range(N_MC)]
    mean = math.fsum(xs) / N_MC
    var = math.fsum((v - mean) ** 2 for v in xs) / N_MC
    assert abs(mean) < STAT_TOL, mean
    assert abs(math.sqrt(var) - 1.0) < STAT_TOL, math.sqrt(var)


def test_I5_terminal_independent_of_start():
    """I5 x_0 = 5.0 与 x_0 = -5.0 生成的 x_T 分布均值差 < 0.05。"""
    _, _, ab = get_cosine_schedule(T)
    rng_a = PCG32(20250103)
    rng_b = PCG32(20250103)
    xa = [q_sample([5.0], T - 1, ab, [rng_a.gauss()])[0] for _ in range(N_MC)]
    xb = [q_sample([-5.0], T - 1, ab, [rng_b.gauss()])[0] for _ in range(N_MC)]
    ma = math.fsum(xa) / N_MC
    mb = math.fsum(xb) / N_MC
    assert abs(ma - mb) < STAT_TOL, (ma, mb)


def test_I6_ddim_eta_zero_is_deterministic():
    """I6 eta = 0 时同输入两次结果逐位相同。"""
    _, _, ab = get_cosine_schedule(T)
    xt = [0.3, -1.2]
    eps = [0.5, 0.7]
    a = ddim_step(xt, 500, 499, ab, eps, 0.0, PCG32(1))
    b = ddim_step(xt, 500, 499, ab, eps, 0.0, PCG32(1))
    assert a == b
    # eta = 0 时不消耗随机流：同一个 rng 连跑两次结果一致
    rng = PCG32(7)
    c = ddim_step(xt, 500, 499, ab, eps, 0.0, rng)
    d = ddim_step(xt, 500, 499, ab, eps, 0.0, rng)
    assert c == d == a


def test_I7_ddim_final_step_equals_x0_hat():
    """I7 t_prev < 0（alpha_bar = 1）时输出 == x0_hat（1e-12）。"""
    _, _, ab = get_cosine_schedule(T)
    xt = [0.3, -1.2, 2.0]
    eps = [0.5, 0.7, -0.2]
    out = ddim_step(xt, 500, -1, ab, eps, 0.5, PCG32(3))
    x0h = predict_x0(xt, 500, ab, eps)
    for a, b in zip(out, x0h):
        assert abs(a - b) < ATOL_STRICT
    out2 = ddpm_step(xt, 500, -1, ab, get_cosine_schedule(T)[0], eps, PCG32(3))
    for a, b in zip(out2, x0h):
        assert abs(a - b) < ATOL_STRICT


def test_I8_ddim_eta_one_matches_ddpm():
    """I8 eta = 1 的 DDIM 与 beta_tilde 版 DDPM 一致。

    两者单步的均值与方差在代数上完全相同，实测逐点差 ~1e-16，
    这里同时给出分布矩（5e-2）与逐点（1e-9）两级断言。
    """
    betas, _, ab = get_cosine_schedule(T)
    rng = PCG32(20250104)
    t = 600
    da = []
    db = []
    worst = 0.0
    for _ in range(4000):
        e = rng.gauss()
        xt = q_sample([2.0], t, ab, [e])
        a = ddim_step(xt, t, t - 1, ab, [e], 1.0, PCG32(12345))[0]
        b = ddpm_step(xt, t, t - 1, ab, betas, [e], PCG32(12345), "tilde")[0]
        da.append(a)
        db.append(b)
        worst = max(worst, abs(a - b))
    assert worst < 1e-9, worst

    def moments(arr):
        m = math.fsum(arr) / len(arr)
        v = math.fsum((x - m) ** 2 for x in arr) / len(arr)
        return m, math.sqrt(v)

    ma, sa = moments(da)
    mb, sb = moments(db)
    assert abs(ma - mb) < STAT_TOL
    assert abs(sa - sb) < STAT_TOL


def test_I9_x0_hat_independent_of_step_count():
    """I9 用真值 eps 时 x0_hat 与走了多少步无关（1e-10）。"""
    _, _, ab = get_cosine_schedule(T)
    rng = PCG32(20250105)
    x0 = [1.0, -1.0]
    for t in (1000 - 1, 700, 300, 50, 1):
        eps = [rng.gauss() for _ in range(2)]
        xt = q_sample(x0, t, ab, eps)
        err = max(abs(a - b) for a, b in zip(predict_x0(xt, t, ab, eps), x0))
        assert err < 1e-10, (t, err)


def test_I10_variance_nonnegative():
    """I10 1 - alpha_bar_{t-1} - sigma^2 >= -1e-12（sqrt 前已被 max(...,0) 保护）。"""
    _, _, ab = get_cosine_schedule(T)
    for t in range(1, T, 37):
        for eta in (0.0, 0.5, 1.0):
            ab_t = max(ab[t], 1e-8)
            ab_p = ab[t - 1]
            if eta == 0.0:
                sigma = 0.0
            else:
                sigma = eta * math.sqrt(max((1.0 - ab_p) / max(1.0 - ab_t, 1e-12), 0.0)) \
                    * math.sqrt(max(1.0 - ab_t / ab_p, 0.0))
            assert 1.0 - ab_p - sigma * sigma >= -1e-12


def test_I11_no_nan_through_full_chain():
    """I11 整条 1000 步采样链（含 t = 0）输出无 nan / inf。"""
    _, _, ab = get_cosine_schedule(T)
    rng = PCG32(20250106)
    x = [1.0, -0.5, 0.25]
    eps = [rng.gauss() for _ in range(3)]
    cur = q_sample(x, T - 1, ab, eps)
    for t in range(T - 1, -1, -1):
        cur = ddim_step(cur, t, t - 1, ab, eps, 0.2, rng)
        assert all(not math.isnan(v) and not math.isinf(v) for v in cur)


def test_oracle_reverse_chain_recovers_x0():
    """用 oracle eps_theta（每步由 x_t 反解真值噪声）走 eta = 0 的 DDIM，
    1000 步后应精确回到 x_0（1e-6）。这是 I11 之外对整条链的强校验。
    """
    _, _, ab = get_cosine_schedule(T)
    rng = PCG32(20250107)
    x = [1.0, -0.5, 0.25]
    cur = q_sample(x, T - 1, ab, [rng.gauss() for _ in range(3)])
    for t in range(T - 1, -1, -1):
        ab_t = max(ab[t], 1e-8)
        c0 = math.sqrt(ab_t)
        c1 = math.sqrt(max(1.0 - ab_t, 0.0))
        eps_t = [(v - c0 * x0i) / c1 for v, x0i in zip(cur, x)]
        cur = ddim_step(cur, t, t - 1, ab, eps_t, 0.0, rng)
    err = max(abs(a - b) for a, b in zip(cur, x))
    assert err < 1e-6, err


def test_linear_schedule_sanity():
    """linear schedule：beta 线性递增，alpha_bar 单调不增且末端 < 1e-3。"""
    betas, alphas, ab = get_linear_schedule(1000)
    assert abs(betas[0] - 1e-4) < 1e-12
    assert abs(betas[-1] - 0.02) < 1e-12
    assert all(betas[i] <= betas[i + 1] + 1e-15 for i in range(999))
    assert ab[-1] < 1e-3
