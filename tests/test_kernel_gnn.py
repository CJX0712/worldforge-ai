"""gnn.py 不变量测试（模块 9，I1 - I13）。"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from worldforge.core.kernels.gnn import (
    add_self_loops,
    degree,
    gcn_propagate,
    rw_normalize,
    sage_mean_aggregate,
    sym_laplacian,
    sym_normalize,
)
from worldforge.core.kernels.linalg import jacobi_eig, matmul, matvec, power_iteration, transpose
from worldforge.core.kernels.numeric import ATOL_LOOSE, ATOL_STRICT, ATOL_TIGHT
from worldforge.core.kernels.sampling import PCG32

GRAPHS = [
    [[0.0, 1.0, 1.0, 0.0],
     [1.0, 0.0, 1.0, 0.0],
     [1.0, 1.0, 0.0, 1.0],
     [0.0, 0.0, 1.0, 0.0]],
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],
    [[0.0, 1.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 1.0, 0.0]],
]


def _rand_graph(n, seed, p=0.4):
    rng = PCG32(seed)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.next_float() < p:
                A[i][j] = 1.0
                A[j][i] = 1.0
    return A


def test_I1_laplacian_null_vector():
    """I1 max abs(L_sym @ (D^{1/2} 1)) < 1e-10，且该向量非零。"""
    for A in GRAPHS:
        L = sym_laplacian(A)
        d = degree(A)
        z = [math.sqrt(v) for v in d]
        assert math.fsum(abs(v) for v in z) > 0.0
        out = matvec(L, z)
        err = max(abs(v) for v in out)
        assert err < ATOL_TIGHT, err


def test_I2_laplacian_psd():
    """I2 随机 x 的二次型 x^T L_sym x >= -1e-10（100 组）。"""
    rng = PCG32(20250101)
    for A in GRAPHS + [_rand_graph(7, 300 + s) for s in range(4)]:
        L = sym_laplacian(A)
        n = len(A)
        for _ in range(100):
            x = [rng.gauss() for _ in range(n)]
            q = math.fsum(x[i] * v for i, v in enumerate(matvec(L, x)))
            assert q >= -1e-10, q


def test_I3_quadratic_form_identity():
    """I3 x^T L_sym x == sum_{(u,v) in E} (x_u/sqrt(d_u) - x_v/sqrt(d_v))^2（1e-10）。

    恒等式只在无孤立点时成立：孤立点的度用 max(d, 1) 保护后 L_sym 对角为 1，
    会在 lhs 多出 x_v^2 一项，而 rhs 没有对应的边。故这里排除含孤立点的图。
    """
    rng = PCG32(20250102)
    for A in (GRAPHS[0], GRAPHS[2], _rand_graph(6, 999)):
        L = sym_laplacian(A)
        d = degree(A)
        n = len(A)
        for _ in range(20):
            x = [rng.gauss() for _ in range(n)]
            lhs = math.fsum(x[i] * v for i, v in enumerate(matvec(L, x)))
            rhs = 0.0
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u][v] != 0.0:
                        du = 1.0 / math.sqrt(max(d[u], 1.0))
                        dv = 1.0 / math.sqrt(max(d[v], 1.0))
                        rhs += (x[u] * du - x[v] * dv) ** 2
            assert abs(lhs - rhs) < ATOL_TIGHT, (lhs, rhs)


def test_I4_spectral_radius_bound():
    """I4 rho(D^{-1/2} A D^{-1/2}) <= 1 + 1e-8；A_hat 的 rho == 1。"""
    for A in GRAPHS:
        An = sym_normalize(A)
        lam, _ = power_iteration(An, iters=4000)
        assert lam <= 1.0 + 1e-8, lam
        Ah = sym_normalize(add_self_loops(A))
        lam2, _ = power_iteration(Ah, iters=4000)
        assert abs(lam2 - 1.0) < 1e-8, lam2


def test_I5_laplacian_spectrum_in_zero_two():
    """I5 L_sym 的谱范围 [0, 2]（Jacobi 交叉验证，1e-8）。"""
    for A in GRAPHS:
        vals, _ = jacobi_eig(sym_laplacian(A))
        assert min(vals) >= -1e-8, min(vals)
        assert max(vals) <= 2.0 + 1e-8, max(vals)


def test_I6_symmetry():
    """I6 A_hat 与 L_sym 对称（1e-12）。"""
    for A in GRAPHS:
        for M in (sym_normalize(add_self_loops(A)), sym_laplacian(A)):
            err = max(abs(M[i][j] - M[j][i]) for i in range(len(A)) for j in range(len(A)))
            assert err < ATOL_STRICT, err


def test_I7_random_walk_row_sums():
    """I7 随机游走归一化 D~^{-1} A~ 行和为 1（1e-12）。"""
    for A in GRAPHS:
        M = rw_normalize(add_self_loops(A))
        for i in range(len(A)):
            assert abs(math.fsum(M[i]) - 1.0) < ATOL_STRICT


def test_I8_permutation_equivariance():
    """I8 置换等变：P A_hat P^T == sym_normalize(P A P^T)（1e-12）。"""
    A = GRAPHS[0]
    n = len(A)
    perm = [2, 0, 3, 1]
    P = [[1.0 if perm[i] == j else 0.0 for j in range(n)] for i in range(n)]
    Pt = transpose(P)
    lhs = matmul(matmul(P, sym_normalize(add_self_loops(A))), Pt)
    rhs = sym_normalize(add_self_loops(matmul(matmul(P, A), Pt)))
    err = max(abs(lhs[i][j] - rhs[i][j]) for i in range(n) for j in range(n))
    assert err < ATOL_STRICT, err


def test_I9_identity_propagation():
    """I9 A = 0（只有自环）时 gcn_propagate(A, H) == H（1e-12）。"""
    rng = PCG32(20250103)
    n, d = 5, 3
    H = [[rng.gauss() for _ in range(d)] for _ in range(n)]
    Z = gcn_propagate([[0.0] * n for _ in range(n)], H)
    for i in range(n):
        for c in range(d):
            assert abs(Z[i][c] - H[i][c]) < ATOL_STRICT


def test_I10_sage_isolated_node():
    """I10 adj_list[v] == [] 时 AGG[v] == H[v]（1e-12）。"""
    H = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    AGG = sage_mean_aggregate([[1], [], [0, 1]], H)
    assert all(abs(a - b) < ATOL_STRICT for a, b in zip(AGG[1], H[1]))
    # 非孤立点的对照分支：AGG[0] == mean(H[0], H[1]) == [2.0, 3.0]
    assert all(abs(a - b) < ATOL_STRICT for a, b in zip(AGG[0], [2.0, 3.0]))


def test_I11_sage_mean_bounded():
    """I11 AGG[v] 落在 {v} u N(v) 特征的最值之间（1e-12）。"""
    rng = PCG32(20250104)
    n, d = 6, 4
    H = [[rng.gauss() * 3.0 for _ in range(d)] for _ in range(n)]
    adj = [[1, 2], [0, 3], [0, 4], [1], [2], []]
    AGG = sage_mean_aggregate(adj, H)
    for v in range(n):
        group = [v] + adj[v]
        for c in range(d):
            lo = min(H[u][c] for u in group)
            hi = max(H[u][c] for u in group)
            assert lo - ATOL_STRICT <= AGG[v][c] <= hi + ATOL_STRICT


def test_I12_degree_conservation():
    """I12 fsum(D~ 1) == n + 2|E|（1e-10）。"""
    for A in GRAPHS:
        n = len(A)
        edges = sum(1 for i in range(n) for j in range(i + 1, n) if A[i][j] != 0.0)
        At = add_self_loops(A)
        assert abs(math.fsum(degree(At)) - (n + 2 * edges)) < ATOL_TIGHT
        assert abs(math.fsum(degree(A)) - 2 * edges) < ATOL_TIGHT


def test_I13_no_nan_with_isolated_nodes():
    """I13 含孤立点的图经 sym_normalize 后无 nan / inf。"""
    A = [[0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]]
    for M in (sym_normalize(A), sym_normalize(add_self_loops(A)), sym_laplacian(A)):
        assert all(not math.isnan(v) and not math.isinf(v) for row in M for v in row)


def test_gcn_layer_stable_magnitude():
    """rho(A_hat) == 1，故多层 GCN 传播不会把特征量级放大（I4 的推论）。"""
    rng = PCG32(20250105)
    A = GRAPHS[0]
    n = len(A)
    H = [[rng.gauss() for _ in range(3)] for _ in range(n)]
    scale0 = max(abs(v) for row in H for v in row)
    cur = H
    for _ in range(10):
        cur = gcn_propagate(A, cur)
    scale1 = max(abs(v) for row in cur for v in row)
    assert scale1 <= scale0 + ATOL_LOOSE
