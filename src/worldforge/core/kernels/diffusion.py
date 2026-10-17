"""扩散模型前向 / 反向与采样器（模块 7）。

做什么
    cosine / linear 噪声 schedule、DDPM 前向加噪、由 eps_theta 反解 x0、
    DDIM 采样步、DDPM 采样步。

索引约定（全模块统一）
    t 从 0 开始，alpha_bars[0] 对应第一步；t_prev < 0 表示回到 x_0，
    此时 alpha_bar 取 1.0，且强制 sigma = 0（最后一步不再注入噪声）。

数学核心
    前向 q(x_t | x_0) = N(sqrt(ab_t) x_0, (1 - ab_t) I)
        x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps
    反解 x0_hat = (x_t - sqrt(1 - ab_t) eps_theta) / sqrt(ab_t)
    DDIM: x_{t_prev} = sqrt(ab_p) x0_hat + sqrt(max(1 - ab_p - sigma^2, 0)) eps_theta + sigma z
          sigma = eta * sqrt((1 - ab_p)/(1 - ab_t)) * sqrt(1 - ab_t/ab_p)
    DDPM: mu = (1/sqrt(alpha_t)) (x_t - beta_t/sqrt(1 - ab_t) eps_theta)
          sigma^2 = beta_t（原始）或 beta_tilde_t = beta_t (1-ab_p)/(1-ab_t)（改进）
    可验证：eta = 1 的 DDIM 与 beta_tilde 版 DDPM 单步分布完全相同（均值与方差逐项相等）。

不变量
    I1  用生成 x_t 时同一个 eps 调 predict_x0，max abs(x0_hat - x0) < 1e-10
    I2  alpha_bar 单调不增
    I3  beta 合法区间 [0, 0.999]；alpha_bar 末端 < 1e-3
    I4  终点 x_T 为标准正态（均值 0、标准差 1，5e-2）
    I5  终点分布与起点无关
    I6  DDIM eta = 0 确定性：同输入两次结果逐位相同
    I7  DDIM 单步到 alpha_bar = 1 时输出 == x0_hat
    I8  DDIM eta = 1 与 DDPM(beta_tilde) 分布一致
    I10 1 - ab_p - sigma^2 >= -1e-12
    I11 整条采样链无 nan

常见坑
    - sqrt(alpha_bar) 前必须 max(ab, 1e-8)，否则末端下溢到 0 会 0 除。
    - beta clamp 到 [0, 0.999] 后必须重新累积 alpha_bar = cumprod(1 - beta)，
      否则 alpha_bar 与 beta 不自洽。
    - sigma == 0 时不要调用 rng，保证随机流干净（I6 依赖这一点）。
"""

import math

__all__ = [
    "get_cosine_schedule",
    "get_linear_schedule",
    "q_sample",
    "predict_x0",
    "ddim_step",
    "ddpm_step",
]


def get_cosine_schedule(T: int, s: float = 0.008, beta_max: float = 0.999):
    """cosine schedule（Nichol & Dhariwal 2021）。

    返回 (betas, alphas, alpha_bars)，长度均为 T。

    不变量
        I2 alpha_bars 单调不增
        I3 0 <= beta_t <= beta_max；alpha_bars[-1] < 1e-3
        自洽性：alpha_bars[t] == prod_{s<=t} alphas[s]，故 beta_t == 1 - ab_t/ab_{t-1}
    """
    if T <= 0:
        raise ValueError("T must be positive")
    f0 = math.cos((s / (1.0 + s)) * math.pi / 2.0) ** 2
    ab_cos = []
    for t in range(1, T + 1):
        ft = math.cos(((t / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
        ab_cos.append(min(ft / f0, 1.0))
    betas = []
    prev = 1.0
    for t in range(T):
        b = 1.0 - ab_cos[t] / prev
        betas.append(min(max(b, 0.0), beta_max))
        prev = ab_cos[t]
    alphas = [1.0 - b for b in betas]
    # clamp 改变了 beta，必须重新累积，否则 alpha_bar 与 beta 不自洽（坑 D5）。
    logs = 0.0
    alpha_bars = []
    for a in alphas:
        logs += math.log(max(a, 1e-12))
        alpha_bars.append(math.exp(logs))
    return betas, alphas, alpha_bars


def get_linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """线性 schedule，作为 cosine 的对照。

    不变量：beta 在 [beta_start, beta_end] 上线性递增；alpha_bar 单调不增。
    """
    if T <= 0:
        raise ValueError("T must be positive")
    betas = [beta_start + (beta_end - beta_start) * t / (T - 1) if T > 1 else beta_start
             for t in range(T)]
    alphas = [1.0 - b for b in betas]
    logs = 0.0
    alpha_bars = []
    for a in alphas:
        logs += math.log(max(a, 1e-12))
        alpha_bars.append(math.exp(logs))
    return betas, alphas, alpha_bars


def q_sample(x0, t: int, alpha_bars, eps):
    """前向加噪 x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps。

    不变量：I4 取 t = T-1 且 ab_T 足够小时，x_T 分布趋于标准正态，与 x_0 无关（I5）。
    """
    ab = max(alpha_bars[t], 1e-8)
    c0 = math.sqrt(ab)
    c1 = math.sqrt(max(1.0 - ab, 0.0))
    return [c0 * a + c1 * b for a, b in zip(x0, eps, strict=False)]


def predict_x0(xt, t: int, alpha_bars, eps_theta):
    """x0_hat = (x_t - sqrt(1 - ab_t) eps_theta) / sqrt(ab_t)。

    不变量：I1 传入生成 x_t 时同一个 eps，则 max abs(x0_hat - x0) < 1e-10。
    """
    ab = max(alpha_bars[t], 1e-8)
    c0 = math.sqrt(ab)
    c1 = math.sqrt(max(1.0 - ab, 0.0))
    return [(x - c1 * e) / c0 for x, e in zip(xt, eps_theta, strict=False)]


def _sigma_ddim(ab_t: float, ab_p: float, eta: float) -> float:
    """DDIM 的 sigma_t = eta * sqrt((1-ab_p)/(1-ab_t)) * sqrt(1 - ab_t/ab_p)。

    不变量：eta = 1 时 sigma^2 == beta_tilde_t = beta_t (1-ab_p)/(1-ab_t)。
    """
    if eta == 0.0:
        return 0.0
    denom = max(1.0 - ab_t, 1e-12)
    return eta * math.sqrt(max((1.0 - ab_p) / denom, 0.0)) * math.sqrt(max(1.0 - ab_t / ab_p, 0.0))


def ddim_step(xt, t: int, t_prev: int, alpha_bars, eps_theta, eta: float, rng):
    """一步 DDIM 反向。

    参数
        t        当前步索引
        t_prev   目标步索引；< 0 表示回到 x_0（ab = 1.0，sigma = 0）
        eta      0 为完全确定的 ODE，1 与 DDPM 边际分布一致
    不变量
        I6  eta = 0 时同输入两次结果逐位相同（且不消耗随机流）
        I7  t_prev < 0（ab_p = 1）时输出 == x0_hat
        I10 1 - ab_p - sigma^2 >= -1e-12（sqrt 前已用 max(...,0) 保护）
    """
    ab_t = max(alpha_bars[t], 1e-8)
    ab_p = alpha_bars[t_prev] if t_prev >= 0 else 1.0
    x0h = predict_x0(xt, t, alpha_bars, eps_theta)
    sigma = 0.0 if t_prev < 0 else _sigma_ddim(ab_t, ab_p, eta)
    coef = math.sqrt(max(1.0 - ab_p - sigma * sigma, 0.0))
    c0 = math.sqrt(ab_p)
    if sigma > 0.0:
        return [c0 * a + coef * b + sigma * rng.gauss() for a, b in zip(x0h, eps_theta, strict=False)]
    # sigma == 0 时绝不调用 rng，保证随机流干净（I6）。
    return [c0 * a + coef * b for a, b in zip(x0h, eps_theta, strict=False)]


def ddpm_step(xt, t: int, t_prev: int, alpha_bars, betas, eps_theta, rng,
              variance: str = "tilde"):
    """一步 DDPM 反向：x_{t-1} = mu_t + sigma_t z。

    参数
        variance  "tilde" 用 beta_tilde_t = beta_t (1-ab_p)/(1-ab_t)；"beta" 用 beta_t
    不变量
        I8  variance = "tilde" 且与 ddim_step(eta=1) 用同一随机流时，两者逐点相同
        最后一步（t_prev < 0）强制 sigma = 0，输出 == x0_hat
    """
    ab_t = max(alpha_bars[t], 1e-8)
    ab_p = alpha_bars[t_prev] if t_prev >= 0 else 1.0
    beta_t = betas[t]
    alpha_t = max(1.0 - beta_t, 1e-8)
    x0h = predict_x0(xt, t, alpha_bars, eps_theta)
    if t_prev < 0:
        return list(x0h)
    sigma2 = beta_t * (1.0 - ab_p) / max(1.0 - ab_t, 1e-12) if variance == "tilde" else beta_t
    sigma = math.sqrt(max(sigma2, 0.0))
    inv = 1.0 / math.sqrt(alpha_t)
    denom = math.sqrt(max(1.0 - ab_t, 0.0)) or 1e-12
    mu = [inv * (x - (beta_t / denom) * e) for x, e in zip(xt, eps_theta, strict=False)]
    if sigma > 0.0:
        return [a + sigma * rng.gauss() for a in mu]
    return mu
