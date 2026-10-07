"""稠密线性代数（模块 2）。

做什么
    Cholesky 分解、Jacobi 对称特征分解、幂迭代、LU（部分选主元）线性求解、
    改进型 Gram-Schmidt，以及 matmul / matvec / transpose / det 等基础算子。

不变量
    I1  Cholesky 重构：max abs((L @ L.T) - A) < 1e-10
    I2  L 下三角且对角严格为正
    I3  Jacobi 特征残差：max_i ||A v_i - lambda_i v_i||_inf < 1e-8
    I4  Jacobi 正交：max abs(V^T V - I) < 1e-10
    I5  迹 / Frobenius 守恒：sum(vals) == tr(A)，sum(vals^2) == ||A||_F^2
    I6  幂迭代残差：||A v - lambda v||_inf < 1e-6
    I7  幂迭代 Rayleigh 商落在 [lambda_min, lambda_max]
    I8  线性解残差：max abs(A x - b) < 1e-8（良态）
    I9  QR 正交：max abs(Q^T Q - I) < 1e-10
    I10 QR 重构：max abs(Q @ R - A) < 1e-10
    I11 det(A) == (prod L[i][i])^2，与 LU 行列式相对误差 < 1e-8

复杂度
    cholesky      O(n^3/3)
    jacobi_eig    O(S * n^3)，S 为 sweep 数（典型 6-10），本模块最慢
    power_iter    O(iters * n^2)
    solve / det   O(n^3/3)
    gram_schmidt  O(n * m^2)

常见坑
    - 非正定：sqrt 前必须判负并抛带 (j, d) 的诊断异常，不要静默 nan。
    - Jacobi 收敛判据必须用相对量 tol * ||A||_F，绝对 1e-12 永远达不到。
    - 排序特征值时必须同步排 V 的列，否则值与向量错位。
"""

import math

from .numeric import EPS

__all__ = [
    "NotPositiveDefiniteError",
    "DegenerateMatrixError",
    "SingularMatrixError",
    "matvec",
    "matmul",
    "transpose",
    "identity",
    "frobenius_norm",
    "cholesky",
    "jacobi_eig",
    "power_iteration",
    "solve",
    "det",
    "gram_schmidt",
]


class NotPositiveDefiniteError(ValueError):
    """Cholesky 在第 j 列遇到非正主子式。

    携带 (j, d) 供调用方决定 jitter 量级（约 1e-10 * trace(A)/n）。
    """

    def __init__(self, j: int, d: float):
        super().__init__("not positive definite at column %d: pivot=%r" % (j, d))
        self.j = j
        self.d = d


class DegenerateMatrixError(ValueError):
    """矩阵退化：幂迭代中 A v 的范数塌缩，或 QR 中出现线性相关列。"""


class SingularMatrixError(ValueError):
    """LU 分解遇到零主元（列选主元后仍为零），矩阵奇异。"""


def matvec(A, x):
    """y_i = sum_j A[i][j] * x[j]。

    不变量：A 为 (n,m)、x 为 (m,) 时输出长度 n；m == 0 时输出全 0。
    """
    return [math.fsum(a * b for a, b in zip(row, x)) for row in A]


def matmul(A, B):
    """C = A @ B，A 为 (n,k)，B 为 (k,m)。

    不变量：输出 (n,m)；B 为空（m == 0）时输出 (n,0)。
    """
    if not B:
        return [[] for _ in A]
    Bt = transpose(B)
    return [[math.fsum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def transpose(A):
    """A^T。

    不变量：transpose(transpose(A)) == A。
    """
    if not A:
        return []
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def identity(n: int):
    """n 阶单位阵。

    不变量：matmul(identity(n), A) == A。
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def frobenius_norm(A) -> float:
    """||A||_F = sqrt(sum_ij A[i][j]^2)。

    不变量：正交变换下不变，Jacobi 每轮 sweep 的 Frobenius 范数守恒。
    """
    return math.sqrt(math.fsum(v * v for row in A for v in row))


def cholesky(A):
    """A = L @ L.T，L 下三角且对角为正。

    不变量
        I1 max abs((L @ L.T)[i][j] - A[i][j]) < 1e-10
        I2 L[i][j] == 0 (i < j) 且 L[i][i] > 0
        I11 det(A) == (prod_i L[i][i])^2
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for j in range(n):
        s = math.fsum(L[j][k] * L[j][k] for k in range(j))
        d = A[j][j] - s
        # d 落在 [-tiny, 0] 属浮点噪声，钳到 0；显著为负才是真的非正定。
        if d < -1e-10 * abs(A[j][j]):
            raise NotPositiveDefiniteError(j, d)
        if d < 0.0:
            d = 0.0
        L[j][j] = math.sqrt(d)
        for i in range(j + 1, n):
            t = math.fsum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (A[i][j] - t) / L[j][j]
    return L


def _jacobi_rotate(A, V, p, q, c, s, n):
    """原地做 A <- J^T A J，并同步 V <- V J。

    J 的定义：J[p][p] = c，J[q][q] = c，J[p][q] = -s，J[q][p] = s，其余同单位阵。
    该约定下 A_new[p][k] = c*A[p][k] + s*A[q][k]，A_new[q][k] = -s*A[p][k] + c*A[q][k]。

    不变量：旋转后 A[p][q] == A[q][p] == 0，且 ||A||_F 不变。
    """
    app = A[p][p]
    aqq = A[q][q]
    apq = A[p][q]
    A[p][p] = c * c * app + s * s * aqq + 2.0 * c * s * apq
    A[q][q] = s * s * app + c * c * aqq - 2.0 * c * s * apq
    A[p][q] = 0.0
    A[q][p] = 0.0
    for k in range(n):
        if k == p or k == q:
            continue
        akp = A[p][k]
        akq = A[q][k]
        A[p][k] = c * akp + s * akq
        A[q][k] = -s * akp + c * akq
        A[k][p] = A[p][k]
        A[k][q] = A[q][k]
    for i in range(n):
        vip = V[i][p]
        viq = V[i][q]
        V[i][p] = c * vip + s * viq
        V[i][q] = -s * vip + c * viq


def jacobi_eig(A, tol: float = 1e-12, max_sweeps: int = 100):
    """对称矩阵 Jacobi 特征分解。

    参数
        A  (n,n) 对称
        tol  相对收敛判据：off(A) < tol * ||A||_F 时停止
    返回
        (vals, V)：vals 降序，V[:, i] 是 vals[i] 的单位特征向量

    不变量
        I3 max_i ||A v_i - lambda_i v_i||_inf < 1e-8
        I4 max abs(V^T V - I) < 1e-10
        I5 sum(vals) == tr(A)，sum(vals^2) == ||A||_F^2（1e-8）
    """
    n = len(A)
    A = [row[:] for row in A]
    V = identity(n)
    frob = frobenius_norm(A)
    # 相对判据：绝对 1e-12 在 ||A|| 较大时永远达不到，会在 1e-13 附近震荡。
    thresh = tol * frob
    prev_off = None
    for _ in range(max_sweeps):
        off = math.sqrt(math.fsum(A[i][j] * A[i][j]
                                  for i in range(n) for j in range(n) if i != j))
        if off <= thresh:
            break
        if prev_off is not None and off >= prev_off:
            # 已到浮点可达极限，继续 sweep 只会原地打转。
            break
        prev_off = off
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300:
                    continue
                theta = 0.5 * math.atan2(2.0 * A[p][q], A[p][p] - A[q][q])
                c = math.cos(theta)
                s = math.sin(theta)
                _jacobi_rotate(A, V, p, q, c, s, n)
    vals = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: (-vals[i], i))
    return [vals[i] for i in order], [[V[r][i] for i in order] for r in range(n)]


def _start_vector(n: int):
    """确定性起始向量（固定 seed 的 LCG），非全 1。

    不变量：两次调用返回逐位相同的向量。
    用固定 RNG 而非全 1，是为了避免起始向量恰好与主导特征向量正交导致幂迭代退化。
    """
    x = 1
    v = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        v.append(x / 0x7FFFFFFF - 0.5)
    nrm = math.sqrt(math.fsum(t * t for t in v))
    if nrm < EPS:
        v = [1.0 / math.sqrt(n)] * n
        return v
    return [t / nrm for t in v]


def power_iteration(A, iters: int = 1000, tol: float = 1e-12):
    """幂迭代求按模最大特征对。

    返回 (lambda, v)，lambda 为 Rayleigh 商 v^T A v / (v^T v)。

    不变量
        I6 ||A v - lambda v||_inf < 1e-6
        I7 lambda_min <= lambda <= lambda_max（与 Jacobi 交叉验证，1e-8）

    收敛判据必须盯住**特征向量**而不是 Rayleigh 商：对称矩阵的 Rayleigh 商误差
    是 O(||v - v_1||^2)，用商的变化量判收敛会在特征向量还差 1e-6 时就提前退出，
    使 I6 的残差断言失败。符号约定：把绝对值最大的分量归一为正，保证结果可复现。
    """
    n = len(A)
    v = _start_vector(n)
    for _ in range(iters):
        w = [math.fsum(a * b for a, b in zip(row, v)) for row in A]
        nw = math.sqrt(math.fsum(t * t for t in w))
        if nw < EPS:
            raise DegenerateMatrixError("A v collapsed to zero")
        nv = [t / nw for t in w]
        # 主导特征值为负时 v 会整体变号，故同时考察同向与反向的差。
        df = max(abs(nv[i] - v[i]) for i in range(n))
        db = max(abs(nv[i] + v[i]) for i in range(n))
        v = nv
        if min(df, db) < tol:
            break
    lam = math.fsum(v[i] * math.fsum(a * b for a, b in zip(A[i], v)) for i in range(n))
    k = max(range(n), key=lambda i: abs(v[i]))
    if v[k] < 0.0:
        v = [-t for t in v]
    return lam, v


def solve(A, b):
    """解 A x = b：LU 分解 + 部分选主元 + 前代回代（增广矩阵形式）。

    不变量
        I8 max_i abs((A x)[i] - b[i]) < 1e-8（良态矩阵）
        主元为 0 时抛 SingularMatrixError，绝不返回 inf
    """
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < EPS:
            raise SingularMatrixError("zero pivot at column %d" % k)
        if piv != k:
            M[k], M[piv] = M[piv], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            if f == 0.0:
                continue
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = math.fsum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x


def det(A) -> float:
    """LU（部分选主元）行列式：det = (-1)^swaps * prod_k U[k][k]。

    不变量：I11 与 Cholesky 的 (prod L[i][i])^2 相对误差 < 1e-8。
    """
    n = len(A)
    M = [list(row) for row in A]
    sign = 1.0
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < EPS:
            return 0.0
        if piv != k:
            M[k], M[piv] = M[piv], M[k]
            sign = -sign
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            if f == 0.0:
                continue
            for j in range(k, n):
                M[i][j] -= f * M[k][j]
    p = 1.0
    for k in range(n):
        p *= M[k][k]
    return sign * p


def gram_schmidt(A):
    """改进型 Gram-Schmidt（含二次重正交）：A = Q @ R。

    输入 A 为 (n,m)，按列读成 m 个向量；输出 Q (n,m) 正交列、R (m,m) 上三角。

    不变量
        I9  max abs(Q^T Q - I) < 1e-10
        I10 max abs(Q @ R - A) < 1e-10
    二次重正交是必要的：经典 CGS 在病态输入下 Q^T Q 偏差可达 1e-6。
    """
    n = len(A)
    m = len(A[0]) if n else 0
    cols = [[A[i][j] for i in range(n)] for j in range(m)]
    R = [[0.0] * m for _ in range(m)]
    for k in range(m):
        for _ in range(2):
            for j in range(k):
                d = math.fsum(cols[j][i] * cols[k][i] for i in range(n))
                R[j][k] += d
                for i in range(n):
                    cols[k][i] -= d * cols[j][i]
        nrm = math.sqrt(math.fsum(v * v for v in cols[k]))
        if nrm < EPS:
            raise DegenerateMatrixError("linearly dependent column %d" % k)
        R[k][k] = nrm
        for i in range(n):
            cols[k][i] /= nrm
    Q = [[cols[j][i] for j in range(m)] for i in range(n)]
    return Q, R
