"""linalg.py 不变量测试（模块 2，I1 - I11）。"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

import pytest

from worldforge.core.kernels.linalg import (
    DegenerateMatrixError,
    NotPositiveDefiniteError,
    SingularMatrixError,
    cholesky,
    det,
    frobenius_norm,
    gram_schmidt,
    identity,
    jacobi_eig,
    matmul,
    matvec,
    power_iteration,
    solve,
    transpose,
)
from worldforge.core.kernels.numeric import ATOL_LOOSE, ATOL_TIGHT
from worldforge.core.kernels.sampling import PCG32


def _spd(n, seed, scale=1.0):
    """构造对称正定矩阵 A = B @ B.T + n * scale * I。"""
    rng = PCG32(seed)
    B = [[rng.gauss() for _ in range(n)] for _ in range(n)]
    A = matmul(B, transpose(B))
    for i in range(n):
        A[i][i] += n * scale
    return A


def _sym(n, seed):
    """构造对称（未必正定）矩阵。"""
    rng = PCG32(seed)
    B = [[rng.gauss() for _ in range(n)] for _ in range(n)]
    A = matmul(B, transpose(B))
    return [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]


def test_I1_cholesky_reconstruction():
    """I1 Cholesky 重构：max abs((L @ L.T) - A) < 1e-10。"""
    for n in (1, 2, 3, 5, 8):
        A = _spd(n, seed=100 + n)
        L = cholesky(A)
        R = matmul(L, transpose(L))
        err = max(abs(R[i][j] - A[i][j]) for i in range(n) for j in range(n))
        assert err < ATOL_TIGHT, (n, err)


def test_I2_lower_triangular_positive_diagonal():
    """I2 L 下三角且对角严格为正。"""
    for n in (2, 4, 6):
        A = _spd(n, seed=200 + n)
        L = cholesky(A)
        for i in range(n):
            for j in range(n):
                if i < j:
                    assert L[i][j] == 0.0
        assert min(L[i][i] for i in range(n)) > 0.0


def test_I3_jacobi_eigen_residual():
    """I3 Jacobi 特征残差：max_i ||A v_i - lambda_i v_i||_inf < 1e-8。"""
    for n in (2, 3, 4, 6, 8):
        A = _sym(n, seed=300 + n)
        vals, V = jacobi_eig(A)
        for k in range(n):
            v = [V[i][k] for i in range(n)]
            Av = matvec(A, v)
            err = max(abs(Av[i] - vals[k] * v[i]) for i in range(n))
            assert err < ATOL_LOOSE, (n, k, err)


def test_I4_jacobi_orthogonality():
    """I4 Jacobi 正交：max abs(V^T V - I) < 1e-10。"""
    for n in (2, 4, 7):
        A = _sym(n, seed=400 + n)
        _, V = jacobi_eig(A)
        G = matmul(transpose(V), V)
        err = max(abs(G[i][j] - (1.0 if i == j else 0.0))
                  for i in range(n) for j in range(n))
        assert err < ATOL_TIGHT, (n, err)


def test_I5_trace_and_frobenius_conservation():
    """I5 迹 / Frobenius 守恒（1e-8）。"""
    for n in (3, 5, 8):
        A = _sym(n, seed=500 + n)
        vals, _ = jacobi_eig(A)
        tr = math.fsum(A[i][i] for i in range(n))
        f2 = frobenius_norm(A) ** 2
        assert abs(math.fsum(vals) - tr) < ATOL_LOOSE
        assert abs(math.fsum(v * v for v in vals) - f2) < ATOL_LOOSE


def test_I6_power_iteration_residual():
    """I6 幂迭代残差：||A v - lambda v||_inf < 1e-6。"""
    for n in (2, 4, 6):
        A = _spd(n, seed=600 + n)
        lam, v = power_iteration(A, iters=5000)
        Av = matvec(A, v)
        err = max(abs(Av[i] - lam * v[i]) for i in range(n))
        assert err < 1e-6, (n, err)


def test_I7_power_iteration_bracketed_by_spectrum():
    """I7 Rayleigh 商落在 [lambda_min, lambda_max]（与 Jacobi 交叉验证，1e-8）。"""
    for n in (3, 5, 7):
        A = _spd(n, seed=700 + n)
        lam, _ = power_iteration(A, iters=5000)
        vals, _ = jacobi_eig(A)
        assert min(vals) - 1e-8 <= lam <= max(vals) + 1e-8
        assert abs(lam - max(vals)) < 1e-8


def test_I8_linear_solve_residual():
    """I8 线性解残差：max abs(A x - b) < 1e-8（良态）。"""
    for n in (2, 4, 6):
        A = _spd(n, seed=800 + n)
        rng = PCG32(8000 + n)
        b = [rng.gauss() for _ in range(n)]
        x = solve(A, b)
        Ax = matvec(A, x)
        err = max(abs(Ax[i] - b[i]) for i in range(n))
        assert err < ATOL_LOOSE, (n, err)


def test_I9_qr_orthogonality():
    """I9 QR 正交：max abs(Q^T Q - I) < 1e-10。"""
    rng = PCG32(910)
    A = [[rng.gauss() for _ in range(4)] for _ in range(6)]
    Q, _ = gram_schmidt(A)
    G = matmul(transpose(Q), Q)
    err = max(abs(G[i][j] - (1.0 if i == j else 0.0)) for i in range(4) for j in range(4))
    assert err < ATOL_TIGHT, err


def test_I10_qr_reconstruction():
    """I10 QR 重构：max abs(Q @ R - A) < 1e-10。"""
    rng = PCG32(1010)
    A = [[rng.gauss() for _ in range(3)] for _ in range(5)]
    Q, R = gram_schmidt(A)
    Z = matmul(Q, R)
    err = max(abs(Z[i][j] - A[i][j]) for i in range(5) for j in range(3))
    assert err < ATOL_TIGHT, err


def test_I11_determinant_agrees_with_cholesky():
    """I11 det(A) == (prod L[i][i])^2，与 LU 路径相对误差 < 1e-8。"""
    for n in (2, 3, 5):
        A = _spd(n, seed=1100 + n)
        L = cholesky(A)
        p = 1.0
        for i in range(n):
            p *= L[i][i]
        d_chol = p * p
        d_lu = det(A)
        rel = abs(d_chol - d_lu) / abs(d_chol)
        assert rel < 1e-8, (n, d_chol, d_lu, rel)


def test_cholesky_raises_on_not_positive_definite():
    """非正定输入必须抛带 (j, d) 的诊断异常，绝不静默返回 nan。"""
    with pytest.raises(NotPositiveDefiniteError) as info:
        cholesky([[1.0, 2.0], [2.0, 1.0]])
    assert info.value.j == 1
    assert info.value.d < 0.0


def test_solve_raises_on_singular():
    """奇异矩阵抛 SingularMatrixError，不返回 inf。"""
    with pytest.raises(SingularMatrixError):
        solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_gram_schmidt_raises_on_dependent_columns():
    """线性相关列抛 DegenerateMatrixError。"""
    with pytest.raises(DegenerateMatrixError):
        gram_schmidt([[1.0, 2.0], [2.0, 4.0]])


def test_basic_operators():
    """matmul / matvec / transpose / identity 的基础恒等式。"""
    A = [[1.0, 2.0], [3.0, 4.0]]
    assert matmul(A, identity(2)) == A
    assert matmul(identity(2), A) == A
    assert transpose(transpose(A)) == A
    assert matvec(A, [1.0, 1.0]) == [3.0, 7.0]
    assert matmul(A, []) == [[], []]
