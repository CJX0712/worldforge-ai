"""图传播与谱工具（模块 9）。

做什么
    自环、度、对称归一化 / 随机游走归一化、GCN 传播、GraphSAGE 均值聚合、
    对称归一化 Laplacian。

数学核心
    A~ = A + I,  D~ = diag(A~ 1),  D = diag(A 1)
    GCN:      A_hat = D~^{-1/2} A~ D~^{-1/2},  H_out = A_hat H W
    SAGE:     AGG_v = MEAN over {v} u N(v)（集合含自身）
    L_sym:    I - D^{-1/2} A D^{-1/2}
    二次型:   x^T L_sym x = sum_{(u,v) in E} (x_u/sqrt(d_u) - x_v/sqrt(d_v))^2
    谱范围:   eig(L_sym) in [0, 2]，rho(D^{-1/2} A D^{-1/2}) <= 1，rho(A_hat) == 1
    零特征向量验证:
        L_sym D^{1/2} 1 = D^{1/2} 1 - D^{-1/2} A D^{-1/2} D^{1/2} 1
                        = D^{1/2} 1 - D^{-1/2} A 1 = D^{1/2} 1 - D^{-1/2} d = 0

不变量
    I1  Laplacian 零特征向量：max abs(L_sym @ (D^{1/2} 1)) < 1e-10
    I2  L_sym 半正定：随机 x 的二次型 >= -1e-10
    I3  二次型恒等式与逐边求和一致（1e-10）
    I4  rho(D^{-1/2} A D^{-1/2}) <= 1 + 1e-8
    I5  L_sym 谱范围 [0, 2]（1e-8）
    I6  A_hat 与 L_sym 对称（1e-12）
    I7  随机游走归一化行和为 1（1e-12）
    I8  置换等变性
    I9  A = 0 时 gcn_propagate(A, H) == H
    I10 SAGE 孤立点 AGG[v] == H[v]
    I11 SAGE 均值落在 {v} u N(v) 的最值之间
    I12 fsum(D~ 1) == n + 2|E|
    I13 含孤立点的图经 sym_normalize 后无 nan

复杂度
    sym_normalize / sym_laplacian  O(n^2)
    gcn_propagate                  O(n^2 d_in + n d_in d_out)
    sage_mean_aggregate            O(|E| d)（稀疏友好）

常见坑
    - 归一化用的度必须与加自环后的矩阵同源；孤立点用 max(d, 1.0) 保护。
    - D^{-1} A（随机游走，行和 1 但不对称）与 D^{-1/2} A D^{-1/2}（对称但行和不是 1）
      不可混用断言。
    - 无向图必须双向加边，入口断言 A == A^T。
"""

import math

from .linalg import matmul

__all__ = [
    "add_self_loops",
    "degree",
    "sym_normalize",
    "rw_normalize",
    "gcn_propagate",
    "sage_mean_aggregate",
    "sym_laplacian",
]


def add_self_loops(A):
    """A~ = A + I。

    不变量：I12 fsum(degree(A~)) == n + 2|E|（无向图，自环数 + 2 倍边数）。
    """
    n = len(A)
    return [[A[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]


def degree(A):
    """d_i = sum_j A[i][j]。

    不变量：无向图上 fsum(d) == 2|E| + 自环数。
    """
    return [math.fsum(row) for row in A]


def sym_normalize(A):
    """A_hat = D^{-1/2} A D^{-1/2}（对称归一化）。

    不变量
        I6 A_hat 对称
        I13 含孤立点（d = 0）时无 nan：度数用 max(d, 1.0) 保护
    """
    n = len(A)
    d = degree(A)
    di = [1.0 / math.sqrt(max(x, 1.0)) for x in d]
    return [[di[i] * A[i][j] * di[j] for j in range(n)] for i in range(n)]


def rw_normalize(A):
    """D^{-1} A（随机游走归一化）。

    不变量：I7 每行和为 1（1e-12）；该矩阵不对称，不可套用 I6 的对称性断言。
    """
    n = len(A)
    d = degree(A)
    di = [1.0 / max(x, 1.0) for x in d]
    return [[di[i] * A[i][j] for j in range(n)] for i in range(n)]


def gcn_propagate(A, H, W=None, activation=None):
    """GCN 单层传播：A_hat H（可选线性变换 W 与逐元素激活）。

    不变量
        I9  A = 0（只有自环）时输出 == H（1e-12）
        谱半径 rho(A_hat) == 1，故多层传播不会放大特征量级
    """
    A_hat = sym_normalize(add_self_loops(A))
    Z = matmul(A_hat, H)
    if W is not None:
        Z = matmul(Z, W)
    if activation is None:
        return Z
    return [[activation(v) for v in row] for row in Z]


def sage_mean_aggregate(adj_list, H):
    """GraphSAGE 均值聚合：MEAN over {v} u N(v)。

    不变量
        I10 孤立点（邻接表为空）时 AGG[v] == H[v]
        I11 AGG[v] 落在 {v} u N(v) 特征的最值之间
    """
    out = []
    for v in range(len(H)):
        nb = adj_list[v]
        k = len(nb) + 1
        out.append([(H[v][c] + math.fsum(H[u][c] for u in nb)) / k
                    for c in range(len(H[v]))])
    return out


def sym_laplacian(A):
    """L_sym = I - D^{-1/2} A D^{-1/2}。

    不变量
        I1  L_sym @ (D^{1/2} 1) == 0（1e-10）
        I2  半正定：x^T L_sym x >= -1e-10
        I6  对称
    """
    n = len(A)
    d = degree(A)
    di = [1.0 / math.sqrt(max(x, 1.0)) for x in d]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0 - di[i] * A[i][i] * di[i]
        for j in range(n):
            if i != j and A[i][j] != 0.0:
                L[i][j] = -di[i] * A[i][j] * di[j]
    return L
