"""优势估计与策略优化（模块 8）。

做什么
    GAE 优势估计（含 episode 边界截断）、优势归一化、PPO 裁剪替代目标、
    价值函数裁剪损失、策略熵。

数学核心
    delta_t = r_t + gamma V(s_{t+1}) (1 - done_t) - V(s_t)
    A_t = sum_{l>=0} (gamma lambda)^l delta_{t+l}
        = delta_t + gamma lambda (1 - done_t) A_{t+1}       （反向单趟递推）
    端点：lambda = 0 退化为 TD 残差；lambda = 1 退化为 MC 回报减基线
        sum_{l=0..L} gamma^l delta_{t+l} = sum_l gamma^l r_{t+l} + gamma^{L+1} V_{t+L+1} - V_t
    PPO: r_t = exp(logp - logp_old)（必须走对数域，否则极小概率下溢为 inf）
         L^CLIP = E[ min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) ]
         L^VF   = E[ max((V - V_targ)^2, (V_old + clip(V - V_old, -eps, eps) - V_targ)^2) ]
    取 min / max 保证是真实目标的悲观下界（pessimistic bound）。

不变量
    I1  lambda = 0 时 A_t == delta_t（1e-12）
    I2  lambda = 1 时 A_t + V_t == R_t（MC 回报，1e-10）
    I3  任意 lambda 下 A_t + V_t == returns_t（1e-12）
    I4  episode 边界截断：dones[t0] = True 时前半段与独立计算逐位一致
    I5  dones[T-1] = True 时 A_{T-1} + V_{T-1} == r_{T-1}
    I6  归一化后均值 0、标准差 1（只统计有效步）
    I7  logp == logp_old 时 L^CLIP == A（1e-12）
    I8  裁剪饱和：A > 0 且 r > 1+eps 时 L == (1+eps) A；A < 0 且 r < 1-eps 时 L == (1-eps) A
    I9  L^CLIP <= r A（悲观下界）
    I10 r == exp(logp - logp_old)
    I11 abs(v - v_old) <= eps 时 L^VF == (v - v_targ)^2
    I12 L^VF >= (v - v_targ)^2
    I13 0 <= S <= log(|A|)

复杂度
    compute_gae O(T)（必须反向单趟，正向做要 O(T^2)）
    ppo_clip_loss / value_clip_loss / policy_entropy 均为 O(N)

常见坑
    - 递推漏乘 (1 - done_t)：episode 结束后的回报会"漏"到下一段。
    - 概率比必须在对数域算：p_new / p_old 在极小概率下会下溢成 inf。
    - values 必须是 critic 输出 V(s_t)，不是回报 returns；长度 T+1。
"""

import math

from .numeric import safe_exp, softmax
from .prob import entropy

__all__ = [
    "compute_gae",
    "normalize_advantages",
    "ppo_clip_loss",
    "value_clip_loss",
    "policy_entropy",
]


def compute_gae(rewards, values, dones, gamma: float = 0.99, lam: float = 0.95,
                last_value: float = 0.0):
    """广义优势估计。

    参数
        rewards  (T,)
        values   (T+1,)；若长度为 T 则用 last_value 补出 bootstrap 值
        dones    (T,) bool / 0-1
    返回
        (advantages, returns)，均长 T

    不变量：I1 / I2 / I3 / I4 / I5。
    """
    T = len(rewards)
    if len(values) == T:
        values = list(values) + [last_value]
    if len(values) != T + 1:
        raise ValueError("values must have length T or T+1")
    adv = [0.0] * T
    gae = 0.0
    for t in range(T - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        # nonterminal = 0 时整个累积量清零，回报不会跨 episode 泄漏。
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns


def normalize_advantages(adv, mask=None, eps: float = 1e-8):
    """优势归一化（均值 0、标准差 1），只统计有效步。

    不变量：I6 有效步上 abs(mean) < 1e-12、abs(std - 1) < 1e-12。
    """
    idx = range(len(adv)) if mask is None else [i for i in range(len(adv)) if mask[i]]
    idx = list(idx)
    out = list(adv)
    if not idx:
        return out
    n = len(idx)
    mean = math.fsum(adv[i] for i in idx) / n
    var = math.fsum((adv[i] - mean) ** 2 for i in idx) / n
    std = math.sqrt(max(var, 0.0))
    if std < eps:
        for i in idx:
            out[i] = 0.0
        return out
    for i in idx:
        out[i] = (adv[i] - mean) / std
    return out


def ppo_clip_loss(logp, logp_old, adv, eps: float = 0.2):
    """PPO 裁剪替代目标（逐元素）。

    返回 (loss (N,), ratio (N,), clip_frac)。

    不变量
        I7  logp == logp_old 时 loss == adv
        I8  裁剪饱和值精确等于 (1 +/- eps) * A
        I9  loss <= r * A + 1e-12（符号写反会立刻违反）
    """
    if len(logp) == 0:
        return [], [], 0.0
    ratios = [safe_exp(a - b) for a, b in zip(logp, logp_old, strict=False)]
    loss = []
    for r, a in zip(ratios, adv, strict=False):
        rc = min(max(r, 1.0 - eps), 1.0 + eps)
        loss.append(min(r * a, rc * a))
    clip_frac = sum(1 for r in ratios if abs(r - 1.0) > eps) / len(ratios)
    return loss, ratios, clip_frac


def value_clip_loss(v, v_old, v_targ, eps: float = 0.2):
    """价值函数裁剪损失（逐元素）。

    不变量
        I11 abs(v - v_old) <= eps 时 loss == (v - v_targ)^2
        I12 loss >= (v - v_targ)^2 - 1e-12（取 max 保证悲观）
    """
    res = []
    for a, b, t in zip(v, v_old, v_targ, strict=False):
        d = a - b
        vc = b + min(max(d, -eps), eps)
        res.append(max((a - t) ** 2, (vc - t) ** 2))
    return res


def policy_entropy(logits) -> float:
    """策略熵 H = entropy(softmax(logits))。

    不变量：I13 0 <= S <= log(|A|) + 1e-12；均匀 logits 时取等号。
    """
    return entropy(softmax(logits))
