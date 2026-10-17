"""信息论与分布度量（模块 3）。

做什么
    归一化、熵、交叉熵、KL、JS、1-D Wasserstein、互信息。

约定
    0 log 0 := 0；所有 log 为自然对数；概率向量必须先 clamp(EPS_SMALL) 再重新归一化；
    JS 的 base 参数决定上界（base=e 时上界 ln2，base=2 时上界 1）。

不变量
    I1  KL(p||q) >= -1e-15
    I2  KL(p||p) == 0（1e-12）
    I3  KL 不对称（反例校验）
    I4  0 <= JS <= ln2（base=e）/ <= 1（base=2）
    I5  JS 对称
    I6  H(p) <= log(n)，均匀分布取等号
    I7  Gibbs：H(p,q) >= H(p)
    I8  I(X;Y) >= -1e-15
    I9  独立则 MI == 0
    I10 I(X;X) == H(X)
    I11 W1(delta_a, delta_b) == abs(a - b)
    I12 W1 是度量：W1(P,P) == 0、对称、满足三角不等式
    I13 I(X;Y) == H(X) - H(X|Y)

复杂度
    entropy / cross_entropy / kl / js 为 O(n)；
    wasserstein_1d 为 O((n+m) log(n+m))（排序主导）；
    mutual_information 为 O(n*m)。

常见坑
    - 归一化必须由 normalize() 走 fsum，否则 KL 会出现 -1e-9 的假负值。
    - W1 的第 k 个区间宽度是 z_{k+1} - z_k，最后一个支撑点不贡献区间。
"""

import math

from .numeric import EPS_SMALL, safe_log

__all__ = [
    "AllZeroProbabilityError",
    "normalize",
    "entropy",
    "cross_entropy",
    "kl_divergence",
    "js_divergence",
    "wasserstein_1d",
    "mutual_information",
    "conditional_entropy",
]


class AllZeroProbabilityError(ValueError):
    """输入概率向量全零（或和为 0），无法归一化。

    不静默返回均匀分布：退化策略应由调用方决定。
    """


def normalize(p):
    """把非负向量按 fsum 归一化成概率向量。

    不变量：abs(fsum(result) - 1.0) < 1e-12；全零输入抛 AllZeroProbabilityError。
    """
    s = math.fsum(p)
    if s <= 0.0:
        raise AllZeroProbabilityError("cannot normalize a zero / negative-sum vector")
    return [x / s for x in p]


def entropy(p) -> float:
    """H(p) = -sum_i p_i log p_i。

    不变量：I6 0 <= H(p) <= log(n) + 1e-12，均匀分布取等号。
    """
    p = normalize(p)
    return -math.fsum(x * safe_log(max(x, EPS_SMALL)) for x in p)


def cross_entropy(p, q) -> float:
    """H(p, q) = -sum_i p_i log q_i。

    不变量：I7 Gibbs 不等式 H(p, q) >= H(p) - 1e-12；q == p 时两者相等。
    """
    p = normalize(p)
    q = normalize(q)
    return -math.fsum(pi * safe_log(max(qi, EPS_SMALL)) for pi, qi in zip(p, q, strict=False))


def kl_divergence(p, q) -> float:
    """KL(p || q) = sum_i p_i log(p_i / q_i)。

    不变量：I1 非负（> -1e-15）；I2 KL(p||p) == 0；I3 不对称。
    """
    p = normalize(p)
    q = normalize(q)
    return math.fsum(pi * (math.log(max(pi, EPS_SMALL)) - math.log(max(qi, EPS_SMALL)))
                     for pi, qi in zip(p, q, strict=False))


def js_divergence(p, q, base: float = math.e) -> float:
    """JS(p||q) = 0.5 KL(p||m) + 0.5 KL(q||m)，m = (p + q) / 2。

    不变量：I4 有界（base=e 上界 ln2，base=2 上界 1）；I5 对称。
    """
    p = normalize(p)
    q = normalize(q)
    m = [0.5 * (a + b) for a, b in zip(p, q, strict=False)]
    v = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    if base == math.e:
        return v
    return v / math.log(base)


def wasserstein_1d(xp, wp, xq, wq) -> float:
    """1-D 闭式 W1 = sum_k |CDF_P(z_k) - CDF_Q(z_k)| * (z_{k+1} - z_k)。

    不变量
        I11 两点质量：W1(delta_a, delta_b) == abs(a - b)（1e-12）
        I12 度量性质：非负、W1(P,P) == 0、对称、三角不等式
    重复支撑点先按坐标聚合再求前缀 CDF，避免后写覆盖前写。
    """
    wp = normalize(wp)
    wq = normalize(wq)
    pm = {}
    qm = {}
    for x, w in zip(xp, wp, strict=False):
        pm[x] = pm.get(x, 0.0) + w
    for x, w in zip(xq, wq, strict=False):
        qm[x] = qm.get(x, 0.0) + w
    pts = sorted(set(pm) | set(qm))
    total = 0.0
    cp = 0.0
    cq = 0.0
    for k in range(len(pts) - 1):
        cp += pm.get(pts[k], 0.0)
        cq += qm.get(pts[k], 0.0)
        total += abs(cp - cq) * (pts[k + 1] - pts[k])
    return total


def mutual_information(joint) -> float:
    """I(X;Y) = sum_{x,y} p(x,y) log(p(x,y) / (p(x) p(y)))。

    输入 joint 为 (n,m)，joint[i][j] = p(X=i, Y=j)，内部自动归一化。

    不变量：I8 非负；I9 独立（joint = outer(px, py)）时为 0；I10 diag 时等于 H(X)。
    """
    n = len(joint)
    m = len(joint[0]) if n else 0
    flat = [v for row in joint for v in row]
    jp = normalize(flat)
    px = [math.fsum(jp[i * m + j] for j in range(m)) for i in range(n)]
    py = [math.fsum(jp[i * m + j] for i in range(n)) for j in range(m)]
    total = 0.0
    for i in range(n):
        for j in range(m):
            pij = jp[i * m + j]
            if pij <= 0.0:
                continue
            total += pij * (math.log(pij) - math.log(max(px[i] * py[j], EPS_SMALL)))
    return total


def conditional_entropy(joint) -> float:
    """H(X|Y) = sum_j p(y_j) * H(X | Y = y_j)。

    不变量：I13 I(X;Y) == H(X) - H(X|Y)（1e-10）。
    """
    n = len(joint)
    m = len(joint[0]) if n else 0
    flat = [v for row in joint for v in row]
    jp = normalize(flat)
    py = [math.fsum(jp[i * m + j] for i in range(n)) for j in range(m)]
    total = 0.0
    for j in range(m):
        if py[j] <= 0.0:
            continue
        col = [jp[i * m + j] / py[j] for i in range(n)]
        total += py[j] * entropy(col)
    return total
