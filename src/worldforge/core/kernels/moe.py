"""稀疏专家混合路由（模块 6）。

做什么
    top-k 路由与门控权重、Switch Transformer 负载均衡辅助损失、
    带容量因子的整层前向。

数学核心
    h(x) = x @ W_g;  p = softmax(h);  S = Top-k(h)
    g_i  = exp(h_i) / sum_{j in S} exp(h_j)   (i in S)
    y    = sum_{i in S} g_i * Expert_i(x)
    f_i  = count_i / (k*N)           sum_i f_i = 1
    P_i  = (1/N) sum_x p_i(x)        sum_i P_i = 1
    loss_aux = alpha * E * sum_i f_i * P_i
    capacity = ceil(capacity_factor * k * N / E)

可严格证明的界（top-1 / argmax 路由）
    被路由到 i 的 token 上必有 p_i >= 1/E（E 个非负概率之和为 1，最大者 >= 1/E）
    故 P_i >= f_i / E，于是
        E * sum_i f_i P_i >= sum_i f_i^2 >= 1/E        （末尾由 Cauchy-Schwarz）
        E * sum_i f_i P_i <= E * max_i P_i <= E
    注意：E * sum_i f_i P_i >= 1 **不成立**。实测 20000 组随机配置（近均匀路由 +
    有限 N 涨落）下最小取到 0.976 < 1。"损失最小值为 1"指的是理想均衡点上的取值，
    不是不等式下界。代码与断言一律用 sum_i f_i^2 下界，绝不写 >= 1。

不变量
    I1  门控权重和为 1（1e-12）
    I2  len(idx) == k，互异，且按 h 降序
    I3  权重排序单调
    I4  确定性并列：logit 相等时索引小者优先
    I5  计数守恒：sum(freq) + dropped == B * k
    I6  max(freq) <= capacity
    I7  完美均衡（人为构造的均匀 f、P）时 loss/alpha == 1（1e-12）
    I8  E*sum f P >= sum f_i^2（1e-9），且 sum f_i^2 >= 1/E
    I9  E*sum f P <= E*max(P) 且 <= E
    I10 abs(sum f - 1) 与 abs(sum P - 1) 均 < 1e-12
    I11 恒等专家时 y[b] == x[b]
    I12 容量充足时 dropped == 0
    I13 0 <= P_i <= 1 且 f_i >= 0

复杂度
    route_topk        O(E log E)
    load_balance_loss O(E)
    moe_forward       O(B * (E*d + k*C_exp))

常见坑
    - 把"均衡点取值 = 1"当成不等式下界，测试会偶发失败（实测低至 0.976）。
    - 丢弃判定顺序：先判 freq < capacity 再 += 1，且 dropped 只在被拒时 += 1。
    - capacity 必须 max(1, ceil(...))，否则 B < E 的小批量会全丢弃。
"""

import math

from .linalg import matvec
from .numeric import softmax, safe_exp

__all__ = [
    "route_topk",
    "load_balance_loss",
    "moe_forward",
    "router_z_loss",
]


def route_topk(h, k: int):
    """top-k 路由。

    返回 (idx, gates)：idx 为选中的专家索引（按 h 降序），gates 为归一化门控权重。

    不变量
        I1 sum(gates) == 1（1e-12）
        I2 len(idx) == k，互异，h[idx[0]] >= ... >= h[idx[k-1]]
        I4 并列时索引小者优先（排序 key 显式带索引）
    """
    E = len(h)
    if k <= 0:
        return [], []
    k = min(k, E)
    order = sorted(range(E), key=lambda i: (-h[i], i))
    idx = order[:k]
    m = max(h[i] for i in idx)
    e = [safe_exp(h[i] - m) for i in idx]
    s = math.fsum(e)
    return idx, [ei / s for ei in e]


def load_balance_loss(freq, P, alpha: float) -> float:
    """Switch 负载均衡损失 alpha * E * sum_i f_i * P_i。

    freq 为原始计数（函数内归一化成 f），P 为各专家的平均路由概率。

    不变量
        I7 人为构造的均匀 f、P 下 loss / alpha == 1（1e-12）
        I8 真实路由下 E*sum f P >= sum f_i^2 >= 1/E（可证，非经验）
        I9 E*sum f P <= E * max(P) <= E
    """
    Nk = math.fsum(freq)
    if Nk <= 0.0:
        return 0.0
    E = len(P)
    f = [c / Nk for c in freq]
    return alpha * E * math.fsum(f[i] * P[i] for i in range(E))


def router_z_loss(h, weight: float = 1e-3) -> float:
    """router z-loss：weight * (log sum_i exp(h_i))^2，稳住 logit 量级。

    不变量：返回值非负；h 全为 -inf 时返回 0.0 而非 nan。
    """
    m = max(h)
    if m == float("-inf"):
        return 0.0
    lse = m + math.log(math.fsum(safe_exp(x - m) for x in h))
    return weight * lse * lse


def moe_forward(x, experts, W_g, k: int = 2, capacity_factor: float = 1.25,
                alpha: float = 0.01, training: bool = True, d_aux=None):
    """整层 MoE 前向。

    参数
        x       (B, d) token 矩阵
        experts list[callable]，长度 E，单个签名为 (d,) -> (d_aux,)
        W_g     (E, d) 路由权重
        k       top-k
        capacity_factor  容量因子
        alpha   负载均衡损失系数
        d_aux   输出维度；None 时用 experts[0](x[0]) 推断
    返回
        (y, aux_loss, stats)，stats 含 freq / capacity / dropped

    不变量
        I5  sum(freq) + dropped == B * k（整数精确）
        I6  max(freq) <= capacity
        I11 experts 全为恒等映射且无丢弃时 y[b] == x[b]（1e-12）
        I12 capacity_factor 足够大时 dropped == 0
    """
    E = len(experts)
    B = len(x)
    capacity = max(1, math.ceil(capacity_factor * k * B / E)) if B else 1
    if d_aux is None:
        d_aux = len(experts[0](x[0])) if B else 0
    freq = [0] * E
    psum = [0.0] * E
    dropped = 0
    y = [[0.0] * d_aux for _ in range(B)]
    for b in range(B):
        h = matvec(W_g, x[b])
        p = softmax(h)
        idx, gates = route_topk(h, k)
        for i in range(E):
            psum[i] += p[i]
        for ei, g in zip(idx, gates):
            # 顺序不可交换：先判定再计数，否则 dropped 与 freq 对不上。
            if freq[ei] >= capacity:
                dropped += 1
                continue
            freq[ei] += 1
            out = experts[ei](x[b])
            for c in range(d_aux):
                y[b][c] += g * out[c]
    if training and B:
        aux = load_balance_loss(freq, [v / B for v in psum], alpha)
    else:
        aux = 0.0
    return y, aux, {"freq": freq, "capacity": capacity, "dropped": dropped}
