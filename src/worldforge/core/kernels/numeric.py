"""数值稳定原语（模块 1，P0 地基）。

做什么
    提供 logsumexp / softmax / log_softmax / safe_exp / safe_log，
    以及全局唯一的容差常量（EPS 系列与 ATOL 系列）。所有上层模块的 exp / log
    一律经由此处，禁止裸调 math.exp。

不变量
    I1 行和为一：abs(fsum(softmax(x)) - 1.0) < 1e-12
    I2 平移不变：softmax(x + c) == softmax(x)，c 取 1e3 / -1e3 / 1e6
    I3 LSE 夹逼：max(x) <= lse(x) <= max(x) + log(n)
    I4 对数一致：log(softmax(x)_i) == log_softmax(x)_i
    I5 溢出免疫：x = [1000.0] * 4 时 lse == 1000 + log(4)，全程无 inf
    I6 严格保序：x_i > x_j 推出 softmax(x)_i > softmax(x)_j

复杂度
    logsumexp / softmax 均为 O(n) 时间、O(1) 或 O(n) 空间，
    瓶颈是两趟扫描（max 后用 math.fsum 补偿求和）。

常见坑
    - 全 -inf 行：-inf - (-inf) = nan，必须在 logsumexp 内拦截。
    - 平移漂移：只允许在 logsumexp 内部减一次 max，边算边减会破坏"同一个 m"。
"""

import math

EPS = 1e-12
EPS_SMALL = 1e-30
TINY = 1e-300
LOG_TINY = -690.0
LOG_MAX = 709.0
ATOL_STRICT = 1e-12
ATOL_TIGHT = 1e-10
ATOL_LOOSE = 1e-8

__all__ = [
    "EPS",
    "EPS_SMALL",
    "TINY",
    "LOG_TINY",
    "LOG_MAX",
    "ATOL_STRICT",
    "ATOL_TIGHT",
    "ATOL_LOOSE",
    "clamp",
    "logsumexp",
    "softmax",
    "log_softmax",
    "safe_exp",
    "safe_log",
]


def clamp(x: float, lo: float, hi: float) -> float:
    """把 x 钳到 [lo, hi]。

    不变量：返回值恒落在闭区间内，且区间内输入原样返回。
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def logsumexp(x):
    """平移恒等式 lse(x) = m + log(sum_i exp(x_i - m))，m = max_i x_i。

    不变量
        max(x) <= lse(x) <= max(x) + log(n)（I3）
        lse 对任意常量平移 c 满足 lse(x + c) == lse(x) + c
        空序列返回 -inf；全 -inf 返回 -inf；含 +inf 返回 +inf
    """
    if len(x) == 0:
        return float("-inf")
    m = max(x)
    if m == float("-inf"):
        return float("-inf")
    if m == float("inf"):
        return float("inf")
    # 至少有一项 exp(x_i - m) == 1.0，故 s >= 1，log 不会得到 -inf。
    s = math.fsum(math.exp(xi - m) for xi in x)
    return m + math.log(s)


def softmax(x):
    """softmax(x)_i = exp(x_i - lse(x))。

    不变量
        I1 行和为一（1e-12）
        I2 平移不变（1e-12）
        I5 全 -inf 退化为均匀分布而非 nan
    """
    n = len(x)
    if n == 0:
        return []
    lse = logsumexp(x)
    if lse == float("-inf"):
        return [1.0 / n] * n
    m = max(x)
    s = math.fsum(math.exp(xi - m) for xi in x)
    # 写成 exp(x_i - m) / s 而不是 exp(x_i - lse)：后者在 |x| 很大时
    # x_i - lse 会发生抵消，损失约 ulp(|x|) 的绝对精度（x ~ 1e6 时是 1e-10 量级）。
    return [safe_exp(xi - m) / s for xi in x]


def log_softmax(x):
    """log_softmax(x)_i = x_i - lse(x)。

    数值上比 log(softmax(x)) 更稳（softmax 可能下溢为 0 后再取 log 得 -inf）。

    不变量
        I4 max_i abs(log(softmax(x)_i) - log_softmax(x)_i) < 1e-12（在 softmax 未下溢的量级上）
    """
    n = len(x)
    if n == 0:
        return []
    lse = logsumexp(x)
    if lse == float("-inf"):
        return [-math.log(n)] * n
    return [xi - lse for xi in x]


def safe_exp(z: float) -> float:
    """带钳位的 exp：上界防 overflow，下界防 0 除。

    上下界刻意不对称：LOG_MAX = 709 是 float64 上溢临界，
    LOG_TINY = -690 保证结果是正规数（后续作分母时不会 0 除）。

    不变量
        z < LOG_TINY  ->  0.0
        z > LOG_MAX   ->  +inf
        其余           ->  math.exp(z)，且永不抛 OverflowError
    """
    if z < LOG_TINY:
        return 0.0
    if z > LOG_MAX:
        return float("inf")
    return math.exp(z)


def safe_log(z: float) -> float:
    """带钳位的 log：z <= 0 时返回 LOG_TINY 而不是 -inf。

    不变量
        返回值恒为有限值，绝不返回 -inf，因此不会污染下游的 fsum 求和。
    """
    if z <= 0.0:
        return LOG_TINY
    return math.log(z)
