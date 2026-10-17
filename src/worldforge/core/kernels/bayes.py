"""共轭更新与变分推断（模块 10）。

做什么
    单变量高斯共轭（标量 / 批次）、Beta-Binomial 共轭、自实现 digamma、
    单变量高斯（未知均值 mu 与未知精度 tau）的 CAVI 坐标上升与 ELBO。

参数化约定
    Gamma 一律用 **rate**：tau ~ Gamma(a, rate b)，E[tau] = a / b，
    E[log tau] = psi(a) - log(b)。全模块不允许出现 scale 形式。

数学核心
    高斯共轭（已知观测方差 s2）：
        1/v_post = 1/v + 1/s2
        m_post   = v_post (m/v + x/s2) = (m s2 + x v) / (v + s2)
    CAVI（均值场 q(mu) q(tau)）：
        1/s_N^2 = 1/s0^2 + n E[tau]
        m_N     = s_N^2 (m0/s0^2 + n E[tau] xbar)
        a_N     = a0 + n/2
        b_N     = b0 + 0.5 sum_i [ (x_i - m_N)^2 + s_N^2 ]      <-- 必须加 s_N^2
    漏掉 + s_N^2 会让 ELBO 非单调，这是 CAVI 实现里最常见也最致命的错误。
    ELBO = E[log p(mu)] + E[log p(tau)] + sum_i E[log p(x_i|mu,tau)]
           + H[q(mu)] + H[q(tau)]
      E[log p(mu)]        = -0.5 log(2 pi s0^2) - 0.5 ((m_N-m0)^2 + s_N^2)/s0^2
      E[log p(tau)]       = a0 log b0 - lgamma(a0) + (a0-1) E[log tau] - b0 E[tau]
      E[log p(x_i|mu,tau)]= 0.5 E[log tau] - 0.5 log(2 pi) - 0.5 E[tau] ((x_i-m_N)^2 + s_N^2)
      H[q(mu)]            = 0.5 log(2 pi e s_N^2)
      H[q(tau)]           = a_N - log b_N + lgamma(a_N) + (1 - a_N) psi(a_N)

不变量
    I1  方差收缩：v_post <= v 且 v_post <= s2
    I2  后验均值夹逼：min(m, x) <= m_post <= max(m, x)
    I3  精度加权形式
    I4  批次 == 序贯
    I5  平坦先验（v = 1e8）退化为 MLE
    I7  Beta 计数精确：a_post + b_post == a + b + n（整数相等）
    I8  Beta 后验均值是先验均值与 MLE 的凸组合
    I9  Beta 后验均值夹逼
    I10 Beta 序贯 == 批次
    I12 ELBO 单调不减：min(ELBO_{t+1} - ELBO_t) >= -1e-9（核心）
    I13 ELBO 收敛
    I14 n = 0 且 q = 先验时 ELBO == 0
    I16 digamma 与 lgamma 中心差分一致（1e-6）
    I17 digamma 递推 psi(x+1) - psi(x) - 1/x == 0（1e-12）
    I18 迭代全程 s_N^2 > 0、a_N > 0、b_N > 0，无 nan

复杂度
    共轭更新 O(1)；digamma O(1)（递推步数有界）；CAVI O(iters * n)

常见坑
    - digamma 必须先递推把 x 抬到 >= 6 再用渐近式，x < 1 时直接用渐近式误差 1e-3。
    - 更新顺序：先算 E[tau]（用旧的 a_N/b_N）-> 更新 q(mu) -> 用新的 m_N, s_N^2 更新 q(tau)。
      顺序反了会震荡。
"""

import math

__all__ = [
    "gaussian_conjugate_update",
    "gaussian_conjugate_update_batch",
    "beta_binomial_update",
    "beta_posterior_mean",
    "digamma",
    "elbo_gaussian",
    "cavi_gaussian",
]


def gaussian_conjugate_update(m: float, v: float, x: float, s2: float):
    """单观测高斯共轭：先验 N(m, v)，观测 x ~ N(mu, s2)。

    返回 (m_post, v_post)。

    不变量
        I1 v_post <= v 且 v_post <= s2（信息永不减少）
        I2 min(m, x) <= m_post <= max(m, x)
        I3 m_post == (m/v + x/s2) / (1/v + 1/s2)
    """
    if v <= 0.0 or s2 <= 0.0:
        raise ValueError("variances must be positive")
    prec = 1.0 / v + 1.0 / s2
    v_post = 1.0 / prec
    m_post = v_post * (m / v + x / s2)
    return m_post, v_post


def gaussian_conjugate_update_batch(m0: float, v0: float, xs, s2: float):
    """批次高斯共轭：1/v_n = 1/v_0 + n/s2，m_n = v_n (m0/v0 + sum(x)/s2)。

    不变量：I4 与逐个序贯更新结果一致（1e-10）；I5 v0 极大时 m_n -> xbar。
    """
    n = len(xs)
    if v0 <= 0.0 or s2 <= 0.0:
        raise ValueError("variances must be positive")
    prec = 1.0 / v0 + n / s2
    v_n = 1.0 / prec
    m_n = v_n * (m0 / v0 + math.fsum(xs) / s2)
    return m_n, v_n


def beta_binomial_update(a: float, b: float, k: int, n: int):
    """Beta-Binomial：a_post = a + k，b_post = b + (n - k)。

    不变量：I7 a_post + b_post == a + b + n（整数精确相等）。
    """
    if not (0 <= k <= n):
        raise ValueError(f"require 0 <= k <= n, got k={k!r} n={n!r}")
    return a + k, b + (n - k)


def beta_posterior_mean(a: float, b: float, k: int, n: int) -> float:
    """后验均值 (a + k) / (a + b + n)。

    不变量
        I8 等于 w * (a/(a+b)) + (1 - w) * (k/n)，w = (a+b)/(a+b+n)（凸组合）
        I9 落在先验均值与 MLE 之间
    """
    ap, bp = beta_binomial_update(a, b, k, n)
    return ap / (ap + bp)


def digamma(x: float) -> float:
    """psi(x)，自实现（标准库无 scipy）。

    递推把 x 抬到 >= 6，再套 6 阶渐近展开：
        psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
                        + 1/(240x^8) - 1/(132x^10)

    不变量
        I16 与 lgamma 中心差分一致（1e-6）
        I17 psi(x + 1) - psi(x) - 1/x == 0（1e-12）
    """
    if x <= 0.0:
        raise ValueError("digamma requires x > 0")
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    s = math.log(x) - 0.5 / x + f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0
                + f * (1.0 / 240.0 + f * (-1.0 / 132.0)))))
    return s + r


def elbo_gaussian(x, mN: float, sNsq: float, aN: float, bN: float,
                  m0: float, s0sq: float, a0: float, b0: float) -> float:
    """单变量高斯 CAVI 的 ELBO。

    不变量
        I14 n = 0 且 q = 先验（mN = m0, sNsq = s0^2, aN = a0, bN = b0）时 ELBO == 0
        I12 沿 CAVI 迭代序列单调不减
    """
    n = len(x)
    E_tau = aN / bN
    E_log_tau = digamma(aN) - math.log(bN)
    e_mu = -0.5 * math.log(2.0 * math.pi * s0sq) - 0.5 * ((mN - m0) ** 2 + sNsq) / s0sq
    e_tau = a0 * math.log(b0) - math.lgamma(a0) + (a0 - 1.0) * E_log_tau - b0 * E_tau
    sq = math.fsum((xi - mN) ** 2 + sNsq for xi in x)
    e_x = 0.5 * n * E_log_tau - 0.5 * n * math.log(2.0 * math.pi) - 0.5 * E_tau * sq
    h_mu = 0.5 * math.log(2.0 * math.pi * math.e * sNsq)
    h_tau = aN - math.log(bN) + math.lgamma(aN) + (1.0 - aN) * digamma(aN)
    return e_mu + e_tau + e_x + h_mu + h_tau


def cavi_gaussian(x, m0: float, s0sq: float, a0: float, b0: float,
                  iters: int = 200, tol: float = 1e-12):
    """CAVI 坐标上升求 q(mu) q(tau)。

    返回 dict(m_N, s_N_sq, a_N, b_N, elbo)。

    不变量
        I12 ELBO 单调不减（min 增量 >= -1e-9）
        I13 收敛：相邻两轮 ELBO 差 < 1e-10
        I18 全程 s_N^2 > 0、a_N > 0、b_N > 0，无 nan
    更新顺序不可交换：先验 E[tau]（旧的 a_N/b_N）-> q(mu) -> 用新的 m_N, s_N^2 更新 q(tau)。
    """
    n = len(x)
    xbar = math.fsum(x) / n if n else 0.0
    mN = m0
    sNsq = s0sq
    aN = a0
    bN = b0
    hist = []
    for _ in range(iters):
        E_tau = aN / bN
        sNsq = 1.0 / (1.0 / s0sq + n * E_tau)
        mN = sNsq * (m0 / s0sq + n * E_tau * xbar)
        aN = a0 + 0.5 * n
        # 必须加 s_N^2：E_q(mu)[(x_i - mu)^2] = (x_i - m_N)^2 + s_N^2。
        bN = b0 + 0.5 * math.fsum((xi - mN) ** 2 + sNsq for xi in x)
        hist.append(elbo_gaussian(x, mN, sNsq, aN, bN, m0, s0sq, a0, b0))
        if len(hist) >= 2 and abs(hist[-1] - hist[-2]) < tol:
            break
    return {"m_N": mN, "s_N_sq": sNsq, "a_N": aN, "b_N": bN, "elbo": hist}
