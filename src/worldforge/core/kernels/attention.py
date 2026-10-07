"""分块（Tiled）FlashAttention 风格 attention（模块 5，风险最高）。

做什么
    attention_naive 物化 N x N score 矩阵，作为 golden reference 常驻；
    attention_tiled 用 online softmax 递推，只维护运行最大值 m、指数和 l、
    未归一化分子 O_acc，把额外内存从 O(N^2) 降到 O(N)。

数学核心（online softmax 递推）
    把索引集切成已处理块 B1 与新块 B2，合并最大值 m = max(m1, m2raw)，则
        exp(s_k - m) = exp(s_k - m1) * exp(m1 - m)      对 B1 内所有 k 成立
    于是
        l_new     = exp(m_old - m_new) * l_old + sum_{k in B2} exp(s_k - m_new)   (1)
        O_acc_new = exp(m_old - m_new) * O_acc + sum_{k in B2} exp(s_k - m_new) v_k (2)
        O = O_acc / l,   LSE = m + log(l)
    修正因子 alpha = exp(m_old - m_new) 必须同时且同值地作用于 (1) 与 (2)。
    只更新一个会导致输出整体缩放错误，而归一化会掩盖它 —— 必须靠等价性断言来抓。
    m 单调不减，故 alpha <= 1 恒为衰减，不会溢出。

不变量
    I1  分块 vs 朴素等价：max abs(O_tiled - O_naive) < 1e-9（核心）
    I2  LSE 等价（排除全 mask 行）：< 1e-10
    I3  凸组合：每行 O[i][c] 落在 V 的第 c 列的 min / max 之间
    I4  全等 score（K 全 0）时 O[i] == mean_j V[j]
    I5  因果等价
    I6  因果上三角无关性：扰动 K[j] (j > i) 后 O_causal[i] 不变
    I7  块大小无关性（含不整除的块大小）
    I8  全 mask 行：O[i] == 0 且 LSE[i] == -inf，绝不出现 nan
    I9  手动 scale 与默认 1/sqrt(d) 一致
    I11 批处理可加性：按行拆分与整体计算逐行一致

复杂度
    朴素 O(Nq*Nk*d + Nq*Nk*dv) 时间、O(Nq*Nk) 额外空间
    分块 同上 FLOPs，额外空间 O(Br*Bc + Nq*dv)

常见坑
    - 只更新 l 不更新 O（或反之）：与 naive 的差会随块数线性增长到 1e-2 量级。
    - 整行 mask 时 m = -inf：exp(-inf - -inf) = nan。用 l[i] == 0.0 判定并置 O = 0。
    - mask 位一律填 -inf 并用哨兵跳过，不要依赖 -1e9 "足够小"。
    - 因果掩码既要跳过整块（j0 > i），也要跳过块内元素（j > i）。
    - 块边界用 min(j0 + Bc, Nk)，否则 Nk % Bc != 0 时越界。
"""

import math

from .numeric import safe_exp

__all__ = ["attention_naive", "attention_tiled"]


def _scores(Q, K, i, j, d, scale):
    """Q[i] 与 K[j] 的点积乘 scale。

    不变量：d == 0 时返回 0.0（空特征维度不构成任何 score）。
    """
    if d == 0:
        return 0.0
    return scale * math.fsum(Q[i][t] * K[j][t] for t in range(d))


def _masked(mask, causal, i, j) -> bool:
    """判断位置 (i, j) 是否可见。

    不变量：mask 与 causal 同时给定时取逻辑与（两者都要可见）。
    """
    if causal and j > i:
        return True
    if mask is not None and not mask[i][j]:
        return True
    return False


def attention_naive(Q, K, V, mask=None, scale=None, causal: bool = False):
    """朴素 attention：物化完整 score 矩阵后逐行 softmax。

    返回 (O, LSE)，O 为 (Nq, dv)，LSE 为 (Nq,)。

    不变量
        I8 整行 mask 时 O[i] == 0、LSE[i] == -inf，绝不为 nan
        I9 scale 缺省值恒为 1/sqrt(d)
    本函数是分块实现的 golden reference，禁止因为"慢"而删除。
    """
    Nq = len(Q)
    Nk = len(K)
    d = len(Q[0]) if Nq else 0
    dv = len(V[0]) if Nk else 0
    if scale is None:
        scale = 1.0 / math.sqrt(d) if d > 0 else 1.0
    O = []
    LSE = []
    for i in range(Nq):
        row = []
        for j in range(Nk):
            if _masked(mask, causal, i, j):
                row.append(float("-inf"))
            else:
                row.append(_scores(Q, K, i, j, d, scale))
        m = max(row) if row else float("-inf")
        if m == float("-inf"):
            O.append([0.0] * dv)
            LSE.append(float("-inf"))
            continue
        e = [safe_exp(r - m) for r in row]
        L = math.fsum(e)
        O.append([math.fsum(e[j] * V[j][c] for j in range(Nk)) / L for c in range(dv)])
        LSE.append(m + math.log(L))
    return O, LSE


def attention_tiled(Q, K, V, mask=None, scale=None, causal: bool = False,
                    Br: int = 64, Bc: int = 64):
    """分块 attention：online softmax，额外内存 O(N)。

    返回 (O, LSE)，与 attention_naive 逐块大小等价（I1 / I7）。

    不变量
        I1 与朴素实现等价（1e-9）
        I8 全 mask 行安全，无 nan
    注意：alpha 在同一处算出后被 (1)(2) 两处引用，不得在两处分别计算。
    """
    Nq = len(Q)
    Nk = len(K)
    d = len(Q[0]) if Nq else 0
    dv = len(V[0]) if Nk else 0
    if scale is None:
        scale = 1.0 / math.sqrt(d) if d > 0 else 1.0
    if Bc <= 0 or Br <= 0:
        raise ValueError("Br and Bc must be positive")

    m = [float("-inf")] * Nq
    l = [0.0] * Nq
    acc = [[0.0] * dv for _ in range(Nq)]

    for j0 in range(0, Nk, Bc):
        j1 = min(j0 + Bc, Nk)
        width = j1 - j0
        for i0 in range(0, Nq, Br):
            i1 = min(i0 + Br, Nq)
            for i in range(i0, i1):
                if causal and j0 > i:
                    continue
                row_m = float("-inf")
                e_row = [float("-inf")] * width
                for jj in range(width):
                    j = j0 + jj
                    if _masked(mask, causal, i, j):
                        continue
                    sc = _scores(Q, K, i, j, d, scale)
                    e_row[jj] = sc
                    if sc > row_m:
                        row_m = sc
                if row_m == float("-inf"):
                    continue
                m_new = m[i] if m[i] > row_m else row_m
                # alpha 只在此处计算一次；(1) 与 (2) 必须引用同一个值。
                alpha = safe_exp(m[i] - m_new) if m[i] != float("-inf") else 0.0
                contrib = [safe_exp(e_row[jj] - m_new) if e_row[jj] != float("-inf") else 0.0
                           for jj in range(width)]
                # (1) 分母折算
                l[i] = l[i] * alpha + math.fsum(contrib)
                # (2) 分子折算，与 (1) 同 alpha：漏掉这一步会让输出整体缩放错误，
                #     而归一化会掩盖它，只能靠与 naive 的等价性断言发现。
                for c in range(dv):
                    acc[i][c] *= alpha
                for jj in range(width):
                    w = contrib[jj]
                    if w == 0.0:
                        continue
                    vrow = V[j0 + jj]
                    for c in range(dv):
                        acc[i][c] += w * vrow[c]
                m[i] = m_new

    O = []
    LSE = []
    for i in range(Nq):
        if l[i] == 0.0:
            O.append([0.0] * dv)
            LSE.append(float("-inf"))
        else:
            O.append([acc[i][c] / l[i] for c in range(dv)])
            LSE.append(m[i] + math.log(l[i]))
    return O, LSE
