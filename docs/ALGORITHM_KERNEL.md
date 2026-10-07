# WorldForge-AI 自研算法内核技术规格书

> 作者：晨星（CJX0712） / 算法设计：苏推衍
> 适用范围：`src/worldforge/core/` 下的 10 个内核模块
> 运行环境：Python 3.13.14 / Windows 11 / 无 GPU
> 依赖约束：**核心层零第三方依赖**，全部数学内核仅用 `math` / `random` / `itertools` / `functools` 等标准库实现。numpy 仅作为**可选加速路径**（`try: import numpy except ImportError: numpy = None`），缺失时必须自动降级到纯 Python 且行为完全一致。

---

## 0. 全局约定（所有模块共同遵守）

### 0.1 数据表示

| 对象 | 类型签名 | 说明 |
|---|---|---|
| 标量 | `float` | 一律 float64，不主动降精度 |
| 向量 | `list[float]` | 长度 `n` |
| 矩阵 | `list[list[float]]` | **行主序**，`M[i][j]` 为第 i 行第 j 列 |
| 掩码 | `list[bool]` 或 `list[int]`（0/1） | 显式传入，不用 `None` 隐式表达 |
| RNG | `worldforge.core.sampling.PCG32` | 禁止直接用 `random.Random` 作为可复现路径 |

`numpy` 可选路径约定：每个函数内部首行做 `if _np is not None and isinstance(x, _np.ndarray)` 分支，两条路径**必须**通过同一组不变量测试（容差按第 0.3 节）。

### 0.2 数值容差常量（定义在 `numeric.py`，全局唯一来源）

```python
EPS        = 1e-12   # 通用相对容差 / 分母保护
EPS_SMALL  = 1e-30   # 极小正数量级，用于替换 0 后再取 log
TINY       = 1e-300  # 接近下溢的下界，用于 exp 结果保护
LOG_TINY   = -690.0  # log(TINY) 的安全下界（float64 下溢阈值约 -745）
LOG_MAX    =  709.0  # log(max float64)，超过此值 exp 溢出为 inf
ATOL_STRICT= 1e-12   # 恒等式级断言（如 KL(p||p) == 0）
ATOL_TIGHT = 1e-10   # 分解/求逆级断言
ATOL_LOOSE = 1e-8    # 迭代算法收敛级断言
```

### 0.3 容差分级规则

| 断言类别 | 推荐容差 | 典型场景 |
|---|---|---|
| 恒等式（数学上严格相等） | `1e-12` | `KL(p\|\|p) == 0`、`softmax` 行和为 1、`ratio == 1` |
| 分解/重构（直接代数重构） | `1e-10` | `L @ L.T == A`、`Q @ R == A`、`x0_hat == x0` |
| 迭代收敛（迭代法逼近） | `1e-8` | Jacobi 残差、幂迭代、CAVI 收敛 |
| 分块 vs 朴素（浮点重排误差） | float64 `1e-9` / float32 `2e-5` | tiled attention、分块归约 |
| 统计性质（蒙特卡洛） | `1e-2` | 采样均值、方差、分布矩 |

> 误差累积经验：纯 Python 循环求和 `n=10^4` 项的相对误差约 `n * eps * kappa`，`kappa` 为条件数。所有涉及 `sum(...)` 的归约，**优先使用 `math.fsum`**（精确浮点求和），尤其是 KL、ELBO、logsumexp 的分子分母。这一条是纯 Python 实现相对 numpy 的关键稳健性来源。

### 0.4 模块依赖图

```
numeric.py   <- 被所有模块依赖（最底层，禁止反向依赖）
   |
   +-- linalg.py
   +-- prob.py ---------+-- sampling.py
   +-- attention.py ----+
   +-- moe.py
   +-- diffusion.py
   +-- rl.py
   +-- gnn.py
   +-- bayes.py
```

`numeric.py` 不得 import 任何其他 core 模块；`linalg.py` 不得 import `prob.py` 及以上。

---

## 1. `numeric.py` -- 数值稳定原语

### 1.1 直觉

神经网络里几乎所有"数值爆炸"都源于同一个动作：对未归一化的 logits 直接 `exp`。`logsumexp` 通过先减去最大值，把 `exp` 的自变量压到 `(-inf, 0]` 区间，从而让指数运算永不溢出；`softmax` 是它的归一化副产品。整个模块是所有其他模块的数值地基。

### 1.2 数学核心

```
m = max_i x_i
logsumexp(x) = m + log( sum_i exp(x_i - m) )                        # 平移恒等式
softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)                    # 平移不变
             = exp( x_i - logsumexp(x) )
log_softmax(x)_i = x_i - logsumexp(x)                               # 最稳，优先用
safe_exp(z) = exp( clamp(z, LOG_TINY, LOG_MAX) )
```

平移不变性证明：`softmax(x + c)_i = exp(x_i + c - (m + c)) / sum_j exp(x_j + c - (m + c)) = exp(x_i - m)/sum_j exp(x_j - m)`。分子分母的 `exp(c)` 精确抵消----注意在浮点里，只有当 `c` 以**同一个数**出现在 `m` 与所有 `x_i` 上时才是严格抵消，因此实现上必须先算 `m` 再统一减，不能边算边减。

`logsumexp` 的界：

```
max_i x_i  <=  logsumexp(x)  <=  max_i x_i + log(n)
```

`safe_exp` 的 `clamp` 上界取 `LOG_MAX = 709.0` 是因为 `exp(709.78)` 已超 float64 上界；下界取 `-690.0` 保证结果是正规数而非 0（避免后续除零），二者不对称是刻意的。

### 1.3 伪代码

```python
def logsumexp(x: list[float]) -> float:
    # input:  x, shape (n,)
    # output: scalar
    if len(x) == 0:
        return float("-inf")
    m = max(x)
    if m == float("-inf"):            # 全为 -inf
        return float("-inf")
    if m == float("inf"):             # 存在 +inf
        return float("inf")
    # 至少一个 exp(x_i - m) == 1.0，故 s >= 1，log 不会 -inf，除法不会 0 除
    s = math.fsum(math.exp(xi - m) for xi in x)
    return m + math.log(s)

def softmax(x: list[float]) -> list[float]:
    # input:  x, shape (n,)   output: shape (n,)
    lse = logsumexp(x)
    if lse == float("-inf"):          # 全 -inf：退化为均匀分布
        n = len(x); return [1.0 / n] * n
    return [safe_exp(xi - lse) for xi in x]

def log_softmax(x: list[float]) -> list[float]:
    lse = logsumexp(x)
    return [xi - lse for xi in x]

def safe_exp(z: float) -> float:
    if z < LOG_TINY: return 0.0
    if z > LOG_MAX:  return float("inf")
    return math.exp(z)

def safe_log(z: float) -> float:
    if z <= 0.0: return LOG_TINY      # 不返回 -inf，避免污染下游求和
    return math.log(z)
```

### 1.4 复杂度

| 函数 | 时间 | 空间 | 瓶颈 |
|---|---|---|---|
| `logsumexp` | `O(n)` | `O(1)` | 两趟扫描（`max` + `fsum`），`fsum` 比 `sum` 慢约 3-5x |
| `softmax` | `O(n)` | `O(n)` | `exp` 调用 n 次 |
| `safe_exp/log` | `O(1)` | `O(1)` | 分支预测 |

### 1.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | 行和为一 | `abs(fsum(softmax(x)) - 1.0)` | `1e-12` |
| I2 | 平移不变（整数值输入） | `softmax(x + c) == softmax(x)` 逐位相等（Sterbenz 引理保证 `x_i - m` 精确），`c` 取 `1e3 / -1e3 / 1e6 / 1e12` | 精确（`==`） |
| I2b | 平移不变（浮点 frac 输入） | `max_i abs(softmax(x+c)_i - softmax(x)_i) <= ulp(max(x)+c)` | 随 `c` 变化，见下方 ulp 定律 |
| I3 | LSE 上下界 | `max(x) - 1e-12 <= lse(x) <= max(x) + log(n) + 1e-12` | `1e-12` |
| I4 | 对数一致性（良态方向，首选） | `max_i abs(softmax(x)_i - exp(log_softmax(x)_i))`，**在下溢区恒有定义** | `1e-12` |
| I4b | 对数一致性（取 log 方向） | `max_{i: softmax_i > 0} abs(log(softmax_i) - log_softmax_i)`，仅对未下溢项断言 | `1e-12` |
| I4c | 诚实下溢 | 凡是 `softmax_i == 0.0` 的下标，必有 `log_softmax_i <= LOG_TINY` | 精确 |
| I5 | 溢出免疫 | `lse([1000.0] * 4) == 1000.0 + log(4)`；`softmax([1000]*4) == [0.25]*4`，且过程中无 `inf` | `1e-12` |
| I6 | 单调性 | `x_i > x_j  =>  softmax(x)_i > softmax(x)_j`（严格，无等号退化） | 严格 `>` |

#### I2 / I4 的实测数值定律（实测数据，非估计）

平移不变性的误差**不是常数**，它随平移量级按 ulp 增长。实测（`x = [0.1, -0.1, 0.05, -0.05, 0.0]`）：

| `c` | 误差（含小数输入） | 误差（整数值输入） | `ulp(max(x)+c)` |
|---|---|---|---|
| `1e3` | `9.5e-15` | `0.0`（逐位相等） | `1.1e-13` |
| `1e6` | `9.8e-12` | `0.0`（逐位相等） | `1.2e-10` |
| `1e8` | `1.2e-09` | `0.0`（逐位相等） | `1.5e-08` |
| `1e10` | `1.6e-07` | `0.0`（逐位相等） | `1.9e-06` |
| `1e12` | `1.0e-05` | `0.0`（逐位相等） | `1.2e-04` |

两条结论：

1. **整数值输入恒为逐位相等**，哪怕 `c = 1e12`。因为 `x_i + c` 与 `m` 同量级且间距为整数，由 Sterbenz 引理 `x_i - m` 是精确减法。所以一般来说 I2 可以对整数值样本断言 `==`（最强形式）。
2. **含小数的输入受 ulp 限制**，误差约为 `ulp(|x| + |c|) / 12`。因此这里的容差**不能**写死 `1e-12`；正确写法是相对判据：

```python
err <= 4.0 * math.ulp(max(abs(v) + abs(c) for v in x))
```

> 这也解释了为什么 softmax 必须实现为 `exp(x_i - m) / fsum(exp(x_j - m))` 而不是 `exp(x_i - logsumexp(x))`：后者要在 `|x| ~ 1e6` 的量级上做 `(x_i + c) - (m + c + log S)`，中间结果 `lse` 本身就带 `ulp(1e6) = 1.16e-10` 的舍入误差，直接把最大标量往下拽 `1e-10`。实测该写法在 `c = 1e6` 时误差 `2.15e-11`，而 `exp(x_i - m)/s` 写法为 `0.0`。

I4 的三个条件里，**I4（良态方向）优先**：`softmax_i - exp(log_softmax_i)` 在 softmax 下溢为 0 的区域**恒等于 0.0 且有定义**（实测 `[-1e3, 0, 1e3]` 上误差 `0.0`），可以直接断言，包括原本被排除的大跨度用例。而 I4b 的 `log(softmax_i)` 方向在下溢时抛 `ValueError: math domain error` ---- 这是 log_softmax 存在的全部理由，不应通过删用例绕过，应改用 I4 方向 + I4c 诚实下溢检查覆盖。

### 1.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| `softmax` 输出全 `nan` | `max(x) == -inf`（整行被 mask 成 `-inf`），`-inf - (-inf) = nan` | 在 `logsumexp` 里显式拦截 `m == -inf` 返回 `-inf`；`softmax` 再退化为均匀分布 |
| `OverflowError: math range error` | 直接 `math.exp(1000)`，未减 `max` | 所有 `exp` 一律走 `safe_exp` |
| `softmax` 平移后数值漂移 `1e-7` | 先减 `max` 再在别处又加常量，破坏了"同一个 `m`" | 只允许在 `logsumexp` 内部减一次 |
| 概率和等于 `1.0000000002` | 用内建 `sum` 逐项累加 | 换 `math.fsum` |
| 全 0 logits 时 softmax 输出 nan | 除以 `lse` 时 `lse` 为 `-inf` | 见 I5 的退化分支 |
| `log(0)` 变 `-inf` 后污染整条 loss | 下游 `safe_log` 返回 `-inf` | `safe_log` 钳到 `LOG_TINY`，并在 `prob.py` 里用 `0*log0 := 0` 约定 |

### 1.7 经典论文

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. ---- 浮点误差分析与补偿求和（`fsum` 的理论依据）。
- Bridle, J. S. (1990). *Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition*. ---- softmax 作为概率输出的经典用法。
- Goldberg, D. (1991). *What Every Computer Scientist Should Know About Floating-Point Arithmetic*. ACM Computing Surveys 23(1). ---- 下溢/上溢阈值的权威参考。

---

## 2. `linalg.py` -- 稠密线性代数

### 2.1 直觉

四个工具解决四类问题：**Cholesky** 把对称正定矩阵拆成 `L @ L.T`，让"解方程 + 采样多元高斯"变成两次三角回代；**Jacobi 特征分解**用一连串平面旋转把对称矩阵"拧"成对角阵，是最稳（也最慢）的对称特征解法；**幂迭代**只求最大特征对，用重复乘法代替分解；**Gram-Schmidt** 把一组向量拧成标准正交基。

### 2.2 数学核心

**Cholesky** (`A = L @ L.T`，`L` 下三角且对角为正)：

```
L[j][j] = sqrt( A[j][j] - sum_{k<j} L[j][k]^2 )
L[i][j] = ( A[i][j] - sum_{k<j} L[i][k]*L[j][k] ) / L[j][j]      i > j
```

**Jacobi 旋转**：对 `(p, q)` 平面，记 `a = A[p][p]`, `b = A[q][q]`, `d = A[p][q]`：

```
theta = 0.5 * atan2(2*d, a - b)          # tan(2 theta) = 2d / (a - b)
c = cos(theta), s = sin(theta)
A_new = J^T A J
更新后: A_new[p][q] = A_new[q][p] = 0
        A_new[p][p] = c^2 a + s^2 b + 2 c s d
        A_new[q][q] = s^2 a + c^2 b - 2 c s d
        A_new[p][k] = c*A[p][k] + s*A[q][k]      k != p,q
        A_new[q][k] = -s*A[p][k] + c*A[q][k]
```
`V <- V @ J` 累积旋转，`V` 的列是特征向量。收敛判据：`off(A) = sqrt(sum_{i != j} A[i][j]^2) < tol`。

**幂迭代**：`v <- A v / ||A v||`；Rayleigh 商 `lambda = v^T A v / (v^T v)`。收敛速率由 `|lambda_2 / lambda_1|^k` 决定。

**LU + 部分选主元** 解 `A x = b`：第 k 列选 `|A[i][k]|` 最大的行做主元，消元后回代。

**改进型 Gram-Schmidt（MGS）**：逐列投影并即时减去已正交分量（比经典 CGS 数值稳定性高一档）。

```
for k in range(n):
    r_kk = || a_k ||
    q_k  = a_k / r_kk
    for j in range(k+1, n):
        r_kj = <q_k, a_j>
        a_j  = a_j - r_kj * q_k        # 立即更新，这是 MGS 与 CGS 的唯一区别
```

### 2.3 伪代码

```python
def cholesky(A: list[list[float]]) -> list[list[float]]:
    # input:  A (n,n) 对称正定    output: L (n,n) 下三角
    n = len(A); L = [[0.0]*n for _ in range(n)]
    for j in range(n):
        s = math.fsum(L[j][k]*L[j][k] for k in range(j))
        d = A[j][j] - s
        if d <= 0.0: raise NotPositiveDefiniteError(j, d)   # 见坑 C1
        L[j][j] = math.sqrt(d)
        for i in range(j+1, n):
            t = math.fsum(L[i][k]*L[j][k] for k in range(j))
            L[i][j] = (A[i][i*0 + j] - t) / L[j][j]          # 即 A[i][j]
    return L

def jacobi_eig(A: list[list[float]], tol=1e-12, max_sweeps=100) -> (list[float], list[list[float]]):
    # input:  A (n,n) 对称   output: (vals (n,), V (n,n))  V[:,i] 是 vals[i] 的特征向量
    n = len(A); A = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for sweep in range(max_sweeps):
        off = math.fsum(A[i][j]**2 for i in range(n) for j in range(n) if i != j)
        if math.sqrt(off) < tol: break
        for p in range(n-1):
            for q in range(p+1, n):
                if abs(A[p][q]) < 1e-300: continue
                theta = 0.5*math.atan2(2.0*A[p][q], A[p][p]-A[q][q])
                c, s = math.cos(theta), math.sin(theta)
                rot(A, p, q, c, s)          # 原地做 J^T A J 的 2x2 平面更新
                rot_cols(V, p, q, c, s)     # V <- V J
    vals = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -vals[i])
    return [vals[i] for i in order], [[V[r][i] for i in order] for r in range(n)]

def power_iteration(A, iters=1000, tol=1e-12):
    # input:  A (n,n)   output: (lambda_1 (float), v (n,))
    n = len(A); v = [1.0/math.sqrt(n)]*n
    for _ in range(iters):
        w = [math.fsum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
        nw = math.sqrt(math.fsum(t*t for t in w))
        if nw < EPS: raise DegenerateMatrixError()
        v = [t/nw for t in w]
    lam = math.fsum(v[i]*math.fsum(A[i][j]*v[j] for j in range(n)) for i in range(n))
    return lam, v

def solve(A: list[list[float]], b: list[float]) -> list[float]:
    # input:  A (n,n) 非奇异, b (n,)   output: x (n,)
    # LU with partial pivoting -> 前代 + 回代
    ...

def gram_schmidt(A: list[list[float]]) -> (list[list[float]], list[list[float]]):
    # input:  A (n,m) 列向量组（按列读）   output: Q (n,m) 正交列, R (m,m) 上三角
    ...
```

### 2.4 复杂度

| 函数 | 时间 | 空间 | 瓶颈 |
|---|---|---|---|
| `cholesky` | `O(n^3/3)` | `O(n^2)` | 内层 `fsum` 使常数因子约 5x，n>200 时应切 numpy |
| `jacobi_eig` | `O(S * n^3)`，`S` 为 sweep 数（典型 6-10） | `O(n^2)` | 每 sweep 有 `n(n-1)/2` 次旋转，每次 `O(n)` -> **最慢的模块之一** |
| `power_iteration` | `O(iters * n^2)` | `O(n)` | 矩阵向量乘；`iters` 由谱比决定 |
| `solve` | `O(n^3/3)` 分解 + `O(n^2)` 回代 | `O(n^2)` | 消元 |
| `gram_schmidt` | `O(n * m^2)` | `O(n*m)` | 投影累加 |

### 2.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | Cholesky 重构 | `max_ij abs((L @ L.T)[i][j] - A[i][j])` | `1e-10` |
| I2 | 下三角 + 正对角 | `L[i][j] == 0 for i < j`；`min_i L[i][i] > 0` | 严格 |
| I3 | Jacobi 特征残差 | `max_i \|\| A v_i - lambda_i v_i \|\|_inf` | `1e-8` |
| I4 | Jacobi 正交 | `max_ij abs((V^T V)[i][j] - delta_ij)` | `1e-10` |
| I5 | 迹 / Frobenius 守恒 | `abs(sum(vals) - trace(A))`；`abs(sum(vals^2) - \|\|A\|\|_F^2)` | `1e-8` |
| I6 | 幂迭代残差（**仅当 `converged == True`**） | `\|\| A v - lambda v \|\|_inf` | `1e-8`（收敛时可收紧到 `1e-12`） |
| I6b | 谱间隙 / 迭代预算关系 | 收敛需 `iters >= ln(tol) / ln(\|lambda_2/lambda_1\|)` | 见下方定律表 |
| I6c | 非收敛必须可见 | 耗尽预算时必须返回 `converged=False` 或抛异常，**禁止静默返回** | 精确 |
| I7 | 幂迭代 Rayleigh 商夹逼 | `lambda_min <= lambda <= lambda_max`（用 Jacobi 结果交叉验证） | `1e-8` |

#### I6 的关键限制：残差是谱间隙的函数，不是常数（实测）

**幂迭代无法对所有矩阵保证固定的残差容差。** 这是算法性质，不是实现 bug。400 组随机对称矩阵的实测：

| `\|lambda_2 / lambda_1\|` | 样本数 | 最大残差 | I6 是否成立 |
|---|---|---|---|
| `<= 0.90` | 376 | `4.68e-12` | 成立 |
| `> 0.98` | 4 | **`1.35e-02`** | **不成立（超 4 个数量级）** |

所需迭代数（`iters >= ln(tol)/ln(ratio)`，默认预算 `1000`）：

| `ratio` | tol=`1e-6` | tol=`1e-8` | tol=`1e-12` |
|---|---|---|---|
| `0.90` | 131 | 175 | 262 |
| `0.99` | **1375** | 1833 | 2749 |
| `0.999` | **13809** | 18411 | 27617 |

结论：**在默认 1000 次预算下，`ratio = 0.99` 就已经不够了**。因此：

1. 收敛判据必须盯**特征向量的变化量** `min(\|\|v_{k+1} - v_k\|\|, \|\|v_{k+1} + v_k\|\|) < tol`（注意主导特征值为负时整体变号，故正负号都要考察），**不能**盯 Rayleigh 商的变化量。对称矩阵的 Rayleigh 商误差是 `O(\|\|v - v_1\|\|^2)`，用商判收敛会在向量还差 `1e-6` 时就提前退出----这正是残差卡在 `1e-6` 量级的直接原因。
2. `power_iteration` 必须**显式报告收敛状态**。当前签名 `(A, iters=1000, tol=1e-12)` 只返回 `(lambda, v)`，调用方无法区分"可信特征值"和"残差 1.35e-2 的垃圾值"。接口应为：

```python
def power_iteration(A, iters=1000, tol=1e-12) -> tuple[float, list[float], bool]:
    # 返回 (lambda, v, converged)；converged=False 时调用方必须自行处理
    ...
    return lam, v, converged
```

3. I6 的残差断言必须写成条件形式，`pytest.skip` 或直接断言 `not converged` 是两种合法处理，但**绝不能无条件断言** `residual < 1e-6`----那样一旦有人在 Jacobi/normalized-Laplacian 谱半径计算里传入接近退化的矩阵，测试就会偶发红灯，且现象看起来像随机不稳定。

> 涉及的地方：`gnn.py` 的 I4（用幂迭代求 `rho(D^{-1/2} A D^{-1/2})`）是这条性质的**下游受害者**----幂迭代在那里用于验证谱半径 `<= 1`，而归一化邻接矩阵在接近正则图时谱间隙天然很小。该处应显式加大 `iters`，或改用 Jacobi（`gnn` 的图规模很小，`n <= 10^3` 时 Jacobi 完全够用且给出全谱）。
| I8 | 线性解残差 | `max_i abs((A x)[i] - b[i])` | `1e-8`（良态）/ `1e-6`（`kappa > 1e6`） |
| I9 | QR 正交 | `max_ij abs((Q^T Q)[i][j] - delta_ij)` | `1e-10` |
| I10 | QR 重构 | `max_ij abs((Q @ R)[i][j] - A[i][j])` | `1e-10` |
| I11 | 行列式关系（Cholesky） | `det(A) == (prod_i L[i][i])^2`，与 LU 路径交叉验证相对误差 | `1e-8` 相对 |

### 2.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| `ValueError: math domain error` 于 `sqrt` | 矩阵非正定（如只有半正定的 `X^T X`，或加了 0 抖动） | 抛出带 `(j, d)` 的诊断异常；调用方加 jitter `A + 1e-8 * I`，jitter 量级应约 `1e-10 * trace(A)/n` |
| 结果 `nan` 而非报错 | `d` 为 `-1e-16`（数值负零附近），`sqrt` 前未判断 | 判断改成 `if d < -1e-10 * abs(A[j][j]): raise`，`d` 在 `[-tiny, 0]` 时钳到 `0.0` |
| Jacobi 一直不收敛，`off` 在 `1e-13` 附近震荡 | `tol=1e-12` 小于浮点可达精度（对 `\|\|A\|\| ~ 1e3` 的矩阵不可能达到） | 判据改成相对：`off < tol * frobenius_norm(A)` |
| Jacobi 特征值顺序乱、与向量对不上 | 排序时只排了 `vals` 没同步排 `V` 的列 | 见伪代码：用 `order` 同时索引两者 |
| `atan2(0, 0)` 当 `A[p][p] == A[q][q]` 且 `d == 0` | 已收敛的 `2x2` 块仍进入旋转 | 加 `abs(A[p][q]) < 1e-300: continue` 短路 |
| 幂迭代收敛极慢或不收敛 | `\|lambda_2/lambda_1\| -> 1`，或初始 `v` 与 `v_1` 正交 | 检测 `iters > 200` 且残差不降时报错；初始向量用固定 seeded RNG 而非全 1 |
| LU 解出全 `inf` | 主元为 0，未做部分选主元 | 每列必须选绝对值最大行并交换 |
| MGS 输出 `Q^T Q != I`（误差 `1e-6`） | 写成了经典 CGS（用原始 `a_k` 做内积） | MGS 必须"减完立刻用新值"；极端病态时做二次重正交（re-orthogonalization） |

### 2.7 经典论文

- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press. ---- Cholesky / LU / QR 的标准参考。
- Jacobi, C. G. J. (1846). *Ueber ein leichtes Verfahren die in der Theorie der Saecularstoerungen vorkommenden Gleichungen numerisch aufzuloesen*. Crelle's Journal. ---- Jacobi 旋转原始文献。
- von Mises, R., & Pollaczek-Geiringer, H. (1929). *Praktische Verfahren der Gleichungsaufloesung*. ZAMM. ---- 幂迭代。
- Demmel, J. (1997). *Applied Numerical Linear Algebra*. SIAM. ---- 相对收敛判据与条件数分析。

---

## 3. `prob.py` -- 信息论与分布度量

### 3.1 直觉

这一模块回答"两个分布有多像、一个分布有多乱"。熵是不确定性，KL 是用 `q` 编码 `p` 时多付的代价（不对称），JS 是 KL 的对称有界化，1-D Wasserstein 是把一堆"土"搬到另一堆所需的最小功，互信息是两个变量共享的比特数。

### 3.2 数学核心

约定：`0 log 0 := 0`；所有 `log` 为自然对数；输入概率向量必须先 `clamp(EPS_SMALL)` 再**重新归一化**。

```
H(p)      = - sum_i p_i log p_i
H(p, q)   = - sum_i p_i log q_i                        # 交叉熵
KL(p||q)  =   sum_i p_i log(p_i / q_i)
JS(p||q)  = 0.5 * KL(p||m) + 0.5 * KL(q||m),   m = (p + q) / 2
I(X;Y)    = H(X) + H(Y) - H(X,Y)
          = sum_{x,y} p(x,y) log( p(x,y) / (p(x) p(y)) )
W1(P,Q)   = int |F_P(t) - F_Q(t)| dt                   # 1-D 闭式解
          = sum_{k=1..K-1} |CDF_P(z_k) - CDF_Q(z_k)| * (z_{k+1} - z_k)
```
`W1` 离散算法：把 `P`、`Q` 的支撑点合并去重排序得 `z_1 < ... < z_K`，在每个 `z_k` 处计算左连续 CDF 差，乘以后继区间宽度累加。

关键不等式（都是可断言的）：
```
H(p) <= log(n)            Gibbs:  H(p,q) >= H(p)
KL(p||q) >= 0             0 <= JS(p||q) <= ln 2   (自然对数下)
I(X;Y) >= 0               I(X;X) = H(X)           W1 >= 0, W1(P,P) = 0
```

### 3.3 伪代码

```python
def normalize(p: list[float]) -> list[float]:
    # input:  p (n,) 非负   output: (n,)
    s = math.fsum(p)
    if s <= 0.0: raise AllZeroProbabilityError()      # 见坑 P1
    return [x / s for x in p]

def _safe_logp(p): return [safe_log(max(x, EPS_SMALL)) for x in p]

def entropy(p) -> float:            # input (n,) -> scalar
    p = normalize(p); return -math.fsum(x*math.log(max(x, EPS_SMALL)) for x in p)

def cross_entropy(p, q) -> float:   # input (n,), (n,) -> scalar
    p = normalize(p); q = normalize(q)
    return -math.fsum(pi*safe_log(max(qi, EPS_SMALL)) for pi, qi in zip(p, q))

def kl_divergence(p, q) -> float:   # KL(p || q)
    p = normalize(p); q = normalize(q)
    return math.fsum(pi*(math.log(max(pi, EPS_SMALL)) - math.log(max(qi, EPS_SMALL)))
                     for pi, qi in zip(p, q))

def js_divergence(p, q, base=math.e) -> float:
    p = normalize(p); q = normalize(q)
    m = [0.5*(a+b) for a, b in zip(p, q)]
    v = 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)
    return v / math.log(base) if base != math.e else v     # base=2 时上界为 1

def wasserstein_1d(xp, wp, xq, wq) -> float:
    # input:  xp (n,), wp (n,) 与 xq (m,), wq (m,)：支撑点 + 权重（自动归一化）
    # output: scalar W1
    wp = normalize(wp); wq = normalize(wq)
    pts = sorted(set([(x, 'p', w) for x, w in zip(xp, wp)] + [(x, 'q', w) for x, w in zip(xq, wq)]))
    ...  # 按点聚合 -> 前缀 CDF -> 逐区间 |dCDF| * dx 累加

def mutual_information(joint: list[list[float]]) -> float:
    # input:  joint (n, m)，joint[i][j] = p(X=i, Y=j)   output: scalar
    flat = [v for row in joint for v in row]
    jp = normalize(flat)
    px = [math.fsum(jp[i*m+j] for j in range(m)) for i in range(n)]
    py = [math.fsum(jp[i*m+j] for i in range(n)) for j in range(m)]
    return math.fsum(jp[i*m+j]*safe_log(jp[i*m+j]/(px[i]*py[j]))
                     for i in range(n) for j in range(m) if jp[i*m+j] > 0.0)
```

### 3.4 复杂度

| 函数 | 时间 | 空间 |
|---|---|---|
| `entropy` / `cross_entropy` / `kl` / `js` | `O(n)` | `O(n)` |
| `wasserstein_1d` | `O((n+m) log(n+m))` | `O(n+m)`（排序主导） |
| `mutual_information` | `O(n*m)` | `O(n*m)` |

### 3.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | KL 非负 | `KL(p\|\|q) >= -1e-15` | `1e-15` |
| I2 | KL 自反为零 | `abs(KL(p\|\|p))` | `1e-12` |
| I3 | KL 不对称（反例校验） | `p=[0.9,0.1], q=[0.5,0.5]`：`abs(KL(p\|\|q) - 0.3681) < tol` 且 `KL(p\|\|q) != KL(q\|\|p)` | `1e-4` |
| I4 | JS 有界 | `-1e-15 <= JS(p\|\|q) <= ln2 + 1e-12`（`base=e`）；`base=2` 时上界为 `1.0` | `1e-12` |
| I5 | JS 对称性 | `abs(JS(p,q) - JS(q,p))` | `1e-12` |
| I6 | 熵上界 | `H(p) <= log(n) + 1e-12`；`H(uniform) == log(n)` | `1e-12` |
| I7 | Gibbs 不等式 | `H(p,q) - H(p) >= -1e-12`；`q == p` 时 `abs(H(p,q) - H(p))` | `1e-12` |
| I8 | MI 非负 | `I(X;Y) >= -1e-15` | `1e-15` |
| I9 | 独立则 MI 为零 | 取 `joint = outer(px, py)`，则 `abs(I)` | `1e-12` |
| I10 | 自信息 | `I(X;X) == H(X)`（用 `joint = diag(px)`） | `1e-12` |
| I11 | W1 两点质量 | `P = delta_a, Q = delta_b` -> `abs(W1 - abs(a-b))` | `1e-12` |
| I12 | W1 度量性质 | `W1(P,P) == 0`；`abs(W1(P,Q) - W1(Q,P))`；三角不等式 `W1(P,R) <= W1(P,Q) + W1(Q,R) + 1e-12` | `1e-12` |
| I13 | MI 与条件熵 | `I(X;Y) == H(X) - H(X\|Y)`，独立实现交叉验证 | `1e-10` |

### 3.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| `nan` 或 `-inf` 出现在 KL 里 | `0 * log(0)` 或 `log(0)` | 统一约定 `0 log 0 := 0`（用 `if pi > 0` 过滤），`q_i` 钳到 `EPS_SMALL` |
| `JS` 算出 `0.75 > ln2 = 0.693` | 用了 `log2` 却按 `ln` 的上界断言 | 显式 `base` 参数：`base=e` 上界 `ln2`，`base=2` 上界 `1` |
| KL 出现 `-1e-9` 的微小负值 | 两个概率向量归一化后和不为 1 | 归一化必须走 `normalize()`（用 `fsum`），且断言 `abs(sum-1) < 1e-12` |
| W1 结果偏大一倍 / 偏小 | 支撑点未排序，或区间宽度用了 `z_k - z_{k-1}` 而非 `z_{k+1} - z_k`（差一个偏移） | 明确：第 `k` 个区间宽度是 `z_{k+1} - z_k`，最后一个点不贡献区间 |
| W1 重复支撑点被覆盖 | 用 dict 聚合时后写覆盖前写 | 先按 `(x, tag)` 分组求和，再合并 |
| MI 输入 joint 的行和不等于 `px` | 调用方传的 `joint` 未归一化 / 维度错 | 入口断言 `abs(fsum(flat) - 1) < 1e-9`，并对 `n*m` 形状做显式校验 |
| 归一化时 `ZeroDivisionError` | 输入全零（如全被 mask 的 logits 经 softmax） | 抛 `AllZeroProbabilityError`，让调用方决定退化策略，**不要**静默返回均匀分布 |

### 3.7 经典论文

- Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal 27(3). ---- 熵与互信息。
- Kullback, S., & Leibler, R. A. (1951). *On Information and Sufficiency*. Annals of Mathematical Statistics 22(1). ---- KL 散度。
- Lin, J. (1991). *Divergence Measures Based on the Shannon Entropy*. IEEE Trans. Information Theory 37(1). ---- JS 散度及其有界性证明。
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein GAN*. ICML. ---- 1-D Wasserstein 的闭式解在机器学习中的普及。
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley. ---- 全部不等式（Gibbs、数据处理不等式）的标准出处。

---

## 4. `sampling.py` -- 解码策略与可复现随机源

### 4.1 直觉

语言模型的输出是 `V` 个 logit。temperature 控制"有多敢冒险"，top-k 与 top-p 控制"砍掉多少尾巴"，beam search 则是"不采样、而是并行保留 B 条最优前缀"。RNG 部分必须自己实现----`random.Random` 在不同 Python 版本间不保证流一致，而我们要的是跨机器可复现。

### 4.2 数学核心

```
temperature:   p_i = exp(z_i / T) / sum_j exp(z_j / T)
               = softmax(z / T)                          # T -> 0+ 时收敛到 argmax 的 one-hot
top-k:         保留最大的 k 个 logit，其余置为 -inf，再 softmax
top-p:         降序排列 p_(1) >= ... >= p_(V)，
               取最小前缀 K = min{ k : sum_{i<=k} p_(i) >= p }，其余置 0，再归一化
beam search:   score(y_{1:t}) = sum_{s<=t} log p(y_s | y_{<s})
               beam_t = Top-B_{by score}( { beam_{t-1} 的每个元素 + 每个候选 token } )
```

PCG32（O'Neill 2014）状态转移（纯整数，跨平台严格一致）：

```
state = (state * 6364136223846793005 + INC) mod 2^64      # INC 为常量（非 0）
xorshifted = ((state >> 18) ^ state) >> 27   mod 2^32
rot        = state >> 59                     mod 2^32
output     = rotr32(xorshifted, rot)
```

### 4.3 伪代码

```python
class PCG32:
    MASK64 = (1 << 64) - 1
    def __init__(self, seed: int, stream: int = 0):
        self.state = (seed * 6364136223846793005 + (stream << 1 | 1)) & self.MASK64
        self.inc   = (stream << 1 | 1) & self.MASK64
    def next_u32(self) -> int: ...
    def next_float(self) -> float:  return self.next_u32() / 4294967296.0   # [0,1)
    def gauss(self) -> float:       # Box-Muller，成对缓存
        ...

def apply_temperature(logits, T):        # (V,) -> (V,)
    if T <= 0.0: raise TemperatureZeroError()
    return [z / T for z in logits]

def top_k_filter(logits, k):             # (V,) -> (V,)  未选中位置为 -inf
    if k <= 0 or k >= len(logits): return logits[:]
    thr = sorted(logits, reverse=True)[k-1]           # 第 k 大值（含并列）
    return [z if z >= thr else float("-inf") for z in logits]

def top_p_filter(logits, p):             # (V,) -> (V,)
    if not (0.0 < p <= 1.0): raise InvalidNucleusProbError(p)
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))   # 稳定并列：按索引
    probs = softmax(logits)
    cum = 0.0; keep = []
    for idx in order:
        keep.append(idx); cum += probs[idx]
        if cum >= p - 1e-9: break                      # 至少保留一个（见坑 S3）
    mask = [False]*len(logits)
    for idx in keep: mask[idx] = True
    return [logits[i] if mask[i] else float("-inf") for i in range(len(logits))]

def sample_categorical(probs, rng, T=1.0, k=0, p=1.0) -> int:
    # input:  probs (V,) 或 logits (V,)   output: 索引 in [0, V)
    probs = normalize(probs)
    if T != 1.0: probs = softmax(apply_temperature([safe_log(x) for x in probs], T))
    if k: probs = softmax(top_k_filter([safe_log(x) for x in probs], k))
    if p < 1.0: probs = softmax(top_p_filter([safe_log(x) for x in probs], p))
    u = rng.next_float(); acc = 0.0
    for i, pi in enumerate(probs):
        acc += pi
        if u < acc: return i
    return len(probs) - 1                              # 浮点兜底（见坑 S5）

def beam_search(step_logprobs_fn, B, max_len, eos_id):
    # input:  step_logprobs_fn(prefix)->(V,) 已归一化的 log 概率
    # output: list[(score: float, ids: list[int])]，长度 <= B，按 score 降序
    beams = [(0.0, [])]
    for _ in range(max_len):
        cand = []
        for score, ids in beams:
            lp = step_logprobs_fn(ids)
            for v in top_k_indices(lp, B):             # 每束只扩 B 个，总候选 B*B
                cand.append((score + lp[v], ids + [v]))
        beams = sorted(cand, key=lambda c: -c[0])[:B]  # 并列时按 ids 字典序，保证确定性
        if all(ids[-1] == eos_id for _, ids in beams): break
    return beams
```

### 4.4 复杂度

| 函数 | 时间 | 空间 | 瓶颈 |
|---|---|---|---|
| `top_k_filter` | `O(V log V)` | `O(V)` | 排序；`V > 1e5` 改用堆 `O(V log k)` |
| `top_p_filter` | `O(V log V)` | `O(V)` | 排序 + 前缀扫描 |
| `sample_categorical` | `O(V)` | `O(V)` | 线性 CDF 扫描（可用别名表降到 `O(1)`，见扩展项） |
| `beam_search` | `O(L * B * V + L * B^2 log(B^2))` | `O(B * L)` | 每步排序 `B^2` 候选 |
| `PCG32.next_u32` | `O(1)` | `O(1)` | Python 大整数乘法（比 C 慢约 50x） |

### 4.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | 过滤后仍归一 | 每次过滤后 `abs(fsum(probs) - 1.0)` | `1e-12` |
| I2 | top-k 基数 | `count(probs > 0) == k`（无并列时） | 精确 |
| I3 | top-p 极小性 | `cumsum(keep) >= p - 1e-9` 且 `cumsum(keep[:-1]) < p` | `1e-9` |
| I4 | temperature 单调性 | `T1 < T2 => H(softmax(z/T1)) <= H(softmax(z/T2)) + 1e-9` | `1e-9` |
| I5 | temperature 极限 | `T=1e-8`：`max(softmax(z/T)) > 1 - 1e-6` 且 argmax 位置与 `argmax(z)` 一致 | `1e-6` |
| I6 | `T=1` 恒等 | `max_i abs(softmax(z/1.0)_i - softmax(z)_i)` | `1e-15` |
| I7 | beam 有序性 | `scores[0] >= scores[1] >= ... >= scores[B-1]` | 精确 |
| I8 | beam 退化为贪心 | `B=1` 时输出序列 == 逐步 `argmax` 的序列 | 精确（同 tie-break） |
| I9 | RNG 可复现 | 同 `seed` 两次实例化，`[next_u32() for _ in range(1000)]` 逐位相等 | 精确（`==`） |
| I10 | RNG 均匀性 | `n=100000`：`abs(mean - 0.5) < 5e-3`；`abs(var - 1/12) < 5e-3`；卡方检验 `chi2 < chi2_{0.001, 99}` | `5e-3` |
| I11 | RNG 无短周期 | `10^6` 次采样中，不存在长度 `< 2^16` 的重复模式（用 `set` 检测重复 32 位值出现次数符合泊松） | 统计 |
| I12 | 采样合法 | 返回索引恒满足 `0 <= idx < V` | 精确 |
| I13 | 高斯矩 | `PCG32.gauss()` 取 `10^5` 样本：`abs(mean) < 0.01`，`abs(std - 1) < 0.01`，`abs(skew) < 0.05` | `1e-2` |

### 4.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| `top_p_filter` 死循环 / 返回空集 | 概率全 0 或全 `-inf`（`cum` 永远到不了 `p`） | 循环外强制 `keep = keep or [order[0]]`；并在入口断言 `fsum(probs) > 0` |
| `ZeroDivisionError` on temperature | `T = 0` | 显式抛错；调用方若要贪心，直接走 `argmax`，不要走 `T=0` |
| 采样分布与 numpy 对不上 | 用 `random.Random` 而非自实现 PCG | 核心路径禁用 `random.Random`，只允许在 `PCG32` 内部封装 |
| 同 seed 两次结果不同 | 排序不稳定（`sort` 对相等元素不保证顺序，尤其含 `-inf`/`nan`） | 所有排序 key 显式带上索引：`key=lambda i: (-logits[i], i)` |
| 采样概率和 `1.0000001` 导致最后一个索引永远取不到 | 浮点累加误差 | 用 `math.fsum` 构造 CDF；末尾加 `return V-1` 兜底 |
| beam search 输出越来越短 / 全是 EOS | 用累加 log 概率（负数）打分，短序列天然占优 | 加长度归一化 `score / (len^alpha)`，`alpha ~ 0.6-0.7`；或明确要求 `max_len` 步后强制截断 |
| beam 里相同序列重复出现 | 候选去重只按 score | 候选 key 用 `(score, tuple(ids))`，排序前 `set` 去重 |
| `nan` 混入 logits 后 `sorted` 结果随机 | `nan` 与任何比较都返回 False，排序不交换 | 入口断言 `all(not math.isnan(z) for z in logits)` |

### 4.7 经典论文

- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. ICLR. ---- nucleus (top-p) sampling。
- Fan, A., Lewis, M., & Dauphin, Y. (2018). *Hierarchical Neural Story Generation*. arXiv:1805.04833. ---- top-k sampling。
- O'Neill, M. E. (2014). *PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation*. ---- PCG32。
- Graves, A. (2012). *Sequence Transduction with Recurrent Neural Networks*. ICML Representation Learning Workshop. ---- 现代 beam search 在序列模型中的用法。
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS. ---- seq2seq beam search 基线。

---

## 5. `attention.py` -- 分块（Tiled）FlashAttention 风格实现

### 5.1 直觉

朴素 attention 要先物化一个 `N x N` 的 score 矩阵，内存 `O(N^2)`，是长上下文的第一瓶颈。FlashAttention 的核心洞察是：softmax 的归一化分母可以**边扫边更新**----维护一个"当前见过的最大值 `m`"和"按当前 `m` 折算的指数和 `l`"，每读一个新块就用 `exp(m_old - m_new)` 把旧累加器**折算**到新基准上。这样永远不需要存完整的 `N x N`，而结果与朴素实现逐位（浮点意义上）等价----它是**精确**的，不是近似。

### 5.2 数学核心：online softmax 递推推导

设某一行的 score 向量为 `s = [s_1, ..., s_N]`（已完成 `/ sqrt(d)` 缩放与 mask 填充），目标输出

```
O = ( sum_k exp(s_k) * v_k ) / ( sum_k exp(s_k) )
```

令 `m = max_k s_k`，定义 `e_k = exp(s_k - m)`，`L = sum_k e_k`，则 `O = (sum_k e_k v_k) / L`。

把索引集切分为**已处理块** `B1` 与**新块** `B2`：

```
B1 已有:  m1 = max_{k in B1} s_k
          L1 = sum_{k in B1} exp(s_k - m1)
          O1 = sum_{k in B1} exp(s_k - m1) * v_k
新块:     m2raw = max_{k in B2} s_k
合并最大值: m  = max(m1, m2raw)
```

代入目标式，关键一步是**把 B1 的每一项从基准 `m1` 换算到基准 `m`**：

```
exp(s_k - m) = exp(s_k - m1) * exp(m1 - m)          # 对 B1 内所有 k 成立
```

于是

```
L = sum_{k in B1} exp(s_k - m) + sum_{k in B2} exp(s_k - m)
  = exp(m1 - m) * sum_{k in B1} exp(s_k - m1) + sum_{k in B2} exp(s_k - m)
  = exp(m1 - m) * L1 + L2                                            ... (1)

O_acc = sum_{k in B1} exp(s_k - m) v_k + sum_{k in B2} exp(s_k - m) v_k
      = exp(m1 - m) * O1 + ( exp(S2 - m) @ V2 )                      ... (2)

最终:  O = O_acc / L            (逐行除法)
      LSE = m + log(L)         (供 log-sum-exp 复用，如合并多条路径 / 计算 log-prob)
```

**要点**：修正因子 `alpha = exp(m1 - m)` 必须**同时且相同地**作用于分母累加器 `L` 与分子累加器 `O_acc`。只更新一个（最常见的 bug）会导致输出整体缩放错误，但因为 `O` 是归一化后的量，错误会被部分掩盖----这正是它危险的原因，必须靠显式的等价性断言来抓。

`m` 单调不减（`m_new >= m_old`），故 `alpha = exp(m_old - m_new) <= 1`，永远是衰减而非放大，**不会溢出**。这是该算法数值稳定的结构性保证。

### 5.3 伪代码

```python
def attention_naive(Q, K, V, mask=None, scale=None, causal=False):
    # input:  Q (Nq, d), K (Nk, d), V (Nk, dv), mask (Nq, Nk) bool 可选
    # output: O (Nq, dv), LSE (Nq,)
    d = len(Q[0]); s = scale if scale else 1.0/math.sqrt(d)
    S = [[s * math.fsum(Q[i][t]*K[j][t] for t in range(d)) for j in range(Nk)]
         for i in range(Nq)]
    if causal:   S[i][j] = -inf when j > i
    if mask:     S[i][j] = -inf where not mask[i][j]
    O = []; LSE = []
    for i in range(Nq):
        row = S[i]; m = max(row)
        if m == -inf:                       # 整行被 mask
            O.append([0.0]*dv); LSE.append(float("-inf")); continue
        e = [safe_exp(r - m) for r in row]; L = math.fsum(e)
        O.append([math.fsum(e[j]*V[j][c] for j in range(Nk))/L for c in range(dv)])
        LSE.append(m + math.log(L))
    return O, LSE

def attention_tiled(Q, K, V, mask=None, scale=None, causal=False,
                    Br=64, Bc=64):
    # input/output 形状同上；Br/Bc 为行块 / 列块大小
    Nq, Nk, d, dv = len(Q), len(K), len(Q[0]), len(V[0])
    s = scale if scale else 1.0/math.sqrt(d)
    m = [float("-inf")] * Nq          # 运行最大值
    l = [0.0] * Nq                    # 运行 exp 和（基准为 m）
    O = [[0.0]*dv for _ in range(Nq)] # 未归一化的累加器
    for j0 in range(0, Nk, Bc):
        j1 = min(j0 + Bc, Nk)
        # --- 列块 [j0, j1) 的 K, V 常驻，只算一次 S 子块 ---
        for i0 in range(0, Nq, Br):
            i1 = min(i0 + Br, Nq)
            for i in range(i0, i1):
                if causal and j0 > i: continue          # 整块在因果掩码外，直接跳过
                row_m = float("-inf")
                e_row = [0.0]*(j1-j0)
                for jj, j in enumerate(range(j0, j1)):
                    if causal and j > i: e_row[jj] = 0.0; continue
                    if mask is not None and not mask[i][j]: e_row[jj] = 0.0; continue
                    sc = s * math.fsum(Q[i][t]*K[j][t] for t in range(d))
                    e_row[jj] = sc                      # 先存 score，稍后统一减 m
                    if sc > row_m: row_m = sc
                if row_m == float("-inf"): continue     # 该块整行被 mask，无贡献
                m_new = m[i] if m[i] > row_m else row_m
                alpha = safe_exp(m[i] - m_new) if m[i] != float("-inf") else 0.0
                # --- (1) 分母折算 ---
                l[i] = l[i]*alpha + math.fsum(safe_exp(e_row[jj] - m_new)
                                               for jj in range(j1-j0)
                                               if e_row[jj] != float("-inf"))
                # --- (2) 分子折算，必须与 (1) 用同一个 alpha ---
                for c in range(dv):
                    O[i][c] = O[i][c]*alpha + math.fsum(
                        safe_exp(e_row[jj] - m_new) * V[j0+jj][c]
                        for jj in range(j1-j0) if e_row[jj] != float("-inf"))
                m[i] = m_new
    for i in range(Nq):
        if l[i] == 0.0:                     # 整行全被 mask
            O[i] = [0.0]*dv
        else:
            O[i] = [O[i][c]/l[i] for c in range(dv)]
    LSE = [m[i] + math.log(l[i]) if l[i] > 0.0 else float("-inf") for i in range(Nq)]
    return O, LSE
```

### 5.4 复杂度

| 实现 | 时间 | 空间（额外） | 瓶颈 |
|---|---|---|---|
| 朴素 | `O(Nq * Nk * d + Nq * Nk * dv)` | `O(Nq * Nk)` | 物化 score 矩阵；`N=8192` 时 float32 需 256 MB |
| 分块 | 同上（同数量级，FLOPs 完全一致） | `O(Br * Bc + Nq * dv)`，即 `O(N)` | HBM 读写量从 `O(N^2)` 降到 `O(N^2 * d / Bc)`；在纯 Python 下真正的收益是**避免 `N^2` 的 Python list 分配** |

> 注：纯 Python 无 GPU，IO-aware 的收益有限，但 `O(N^2) -> O(N)` 的**内存**收益是真实的，且等价性测试本身就是对 online softmax 正确性的最强证明。`N <= 2048` 时朴素路径可作为 golden reference 常驻测试。

### 5.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | **分块 vs 朴素等价（核心）** | `max_{i,c} abs(O_tiled[i][c] - O_naive[i][c])` | float64 `1e-9`；float32 `2e-5` |
| I2 | LSE 等价 | `max_i abs(LSE_tiled[i] - LSE_naive[i])`（排除全 mask 行） | `1e-10` |
| I3 | 凸组合（行落在 V 的凸包内） | 对每行 `i`：`min_j V[j][c] - tol <= O[i][c] <= max_j V[j][c] + tol`，对所有 `c` | `1e-9` |
| I4 | 全等 score -> 均匀权重 | 令 `K` 的列为全 0（score 全 0），则 `O[i] == mean_j V[j]` | `1e-10` |
| I5 | 因果等价 | `O_causal[i] == softmax over {j <= i} 仅用前 i+1 个 key 重算的结果` | `1e-9` |
| I6 | 因果上三角无关性 | 扰动 `K[j] (j > i)` 后，`O_causal[i]` 不变 | `1e-12` |
| I7 | 块大小无关性 | `Br, Bc` 取 `(16,16) / (64,64) / (7,13)`（故意不整除）结果一致 | `1e-9` |
| I8 | 全 mask 行安全 | 整行 mask 时 `O[i] == 0` 且 `LSE[i] == -inf`，**不得出现 `nan`** | 精确 |
| I9 | 缩放一致性 | 手动传 `scale` 与默认 `1/sqrt(d)` 在 `d` 相同时结果一致 | `1e-15` |
| I10 | 能量守恒 | `sum_c O[i][c] * w_c`（任意固定 `w`）在 tiled/naive 间一致 | `1e-9` |
| I11 | 批处理可加性 | `Q` 按行拆成两半分别计算，与一次算完拼接后逐行一致 | `1e-9` |

### 5.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| tiled 与 naive 差 `1e-2` 量级，且随块数增加 | **只更新了 `l` 没更新 `O`（或反之）**，即漏了 `alpha` | 把 `alpha` 的计算提到块内唯一位置，两处引用同一变量；用 I1 断言在 CI 里强制 |
| 结果随 `Bc` 变大而变差 | 在块内对每个元素用了不同的 `m`（如用块内 `row_m` 但外层又混用全局 `m`） | 每个块必须**先**算出该块的 `row_m`，合并出 `m_new`，再**统一**用 `m_new` 做 `exp` |
| 出现 `nan` 输出 | 整行 mask 时 `m = -inf`，`exp(-inf - (-inf)) = exp(nan) = nan` | 用 `l[i] == 0.0` 判定全 mask，跳过并置 `O = 0`；`m[i] == -inf` 时 `alpha` 显式取 `0.0` |
| mask 位置仍贡献了权重 | 用 `-1e9` 填充再 `exp`，`exp(-1e9 - m)` 下溢为 0（其实没问题），但若 `m` 也是 `-1e9` 量级则 `exp(0) = 1` 出错 | mask 位一律填 `float("-inf")` 并在 `e_row` 里用哨兵跳过；不要依赖 `-1e9` 的"足够小" |
| 因果掩码下 `O[i]` 依赖了未来 | 只跳过了 `j0 > i` 的整块，没跳过块内 `j > i` | 块内仍需逐元素判断 `j > i`（见伪代码内层 `continue`） |
| 最后一块越界 `IndexError` | 用固定 `Bc` 切片而 `Nk % Bc != 0` | 用 `j1 = min(j0 + Bc, Nk)`，并在 I7 里专门测不整除的块大小 |
| `scale` 未除以 `sqrt(d)`，训练时梯度爆炸 | 默认参数被覆盖 | 默认 `scale = 1/sqrt(d)`，并断言 `scale > 0` |
| 用 `float32` 时等价性断言失败 | 容差仍用 `1e-9` | 按第 0.3 节分级：float32 用 `2e-5`（相对误差） |
| 纯 Python 下 `N=4096` 跑几分钟 | 三重 Python 循环 + `fsum` 生成器开销 | `d` 维内积改用 `zip` + 局部 `sum`（`d <= 64` 时 `sum` 更快且误差可接受），或走 numpy 可选路径 |

### 5.7 经典论文

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS. ---- scaled dot-product attention。
- Milakov, M., & Gimelshev, N. (2018). *Online Normalizer Calculation for Softmax*. arXiv:1805.02867. ---- **online softmax 递推的原始出处**，本文档 5.2 节推导的直接来源。
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS. ---- 分块 / IO-aware 实现。
- Rabe, M. N., & Staats, C. (2021). *Self-attention Does Not Need O(n^2) Memory*. arXiv:2112.05682. ---- 同样的内存复杂度结论，独立提出。
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv:2307.08691. ---- 循环顺序与块划分的工程优化参考。

---

## 6. `moe.py` -- 稀疏专家混合路由

### 6.1 直觉

每个 token 只看 `k` 个专家（而不是全部 `E` 个），从而把参数量与计算量解耦。难点不在"选谁"，而在"别让所有 token 都挤到同一个专家"----这就是负载均衡辅助损失；以及"每个专家一次最多处理多少 token"----这就是容量因子。

### 6.2 数学核心

```
router logits:   h(x) = x @ W_g                       # (d,) -> (E,)
route probs:     p = softmax(h(x))                    # (E,)
top-k 选择:      S = Top-k indices of h(x)（并列按索引升序，保证确定性）
门控权重:        g_i = exp(h_i) / sum_{j in S} exp(h_j)     i in S；其余为 0
输出:            y = sum_{i in S} g_i * Expert_i(x)
```

**负载均衡辅助损失**（Switch Transformer, Fedus et al. 2021）：

对一批 `N` 个 token、`E` 个专家：

```
f_i = (1 / (k*N)) * #{ tokens routed to expert i }      # 归一化，sum_i f_i = 1
P_i = (1 / N) * sum_{x} p_i(x)                          # 路由概率均值，sum_i P_i = 1
loss_aux = alpha * E * sum_{i=1..E} f_i * P_i
```

该项在完美均衡（`f_i = P_i = 1/E`）时**恰好**取 `alpha * 1.0`（实测 `E * sum_i f_i P_i = 1.000000000000`）。

**可严格证明的界**（top-1 / argmax 路由，`sum_i f_i = sum_i P_i = 1`）：

```
对每个 i，凡被路由到 i 的 token 上必有 p_i >= 1/E（E 个非负概率之和为 1，其最大者 >= 1/E）
故  P_i = (1/N) * sum_x p_i(x) >= (count_i / N) / E = f_i / E
于是  E * sum_i f_i P_i >= E * sum_i f_i * (f_i / E) = sum_i f_i^2 >= 1/E     （末尾由 Cauchy-Schwarz）
上界  E * sum_i f_i P_i <= E * max_i P_i <= E
```

> **重要更正：`E * sum_i f_i P_i >= 1` 不是通用保证。** 实测 20000 组随机配置（近均匀路由 + 有限 `N` 涨落）下最小取到 `0.976 < 1`。Switch Transformer 论文中"损失最小值为 1"指的是**理想均衡点**上的取值，而非不等式下界。实现与测试必须按上面的 `sum_i f_i^2` 下界断言，不要写 `>= 1`（见坑 M5）。

**容量因子**：

```
capacity = ceil( capacity_factor * k * N / E )
每个 expert 每批最多接收 capacity 个 token；超出部分 -> dropped（或走残差直通）
```

### 6.3 伪代码

```python
def route_topk(h, k):
    # input:  h (E,) router logits   output: (indices (k,), gates (k,))
    order = sorted(range(len(h)), key=lambda i: (-h[i], i))     # 确定性并列
    idx = order[:k]
    m = max(h[i] for i in idx)
    e = [safe_exp(h[i] - m) for i in idx]
    s = math.fsum(e)
    return idx, [ei / s for ei in e]

def load_balance_loss(freq, P, alpha):
    # input:  freq (E,) 每个专家被分配的 token 计数（原始计数，函数内归一化）
    #         P (E,) 每个专家的平均路由概率
    # output: scalar
    Nk = math.fsum(freq)
    if Nk <= 0: return 0.0
    f = [c / Nk for c in freq]
    return alpha * len(P) * math.fsum(f[i]*P[i] for i in range(len(P)))

def moe_forward(x, experts, W_g, k=2, capacity_factor=1.25, alpha=0.01,
                training=True):
    # input:  x (B, d)  tokens; experts: list[callable]，长度 E
    # output: y (B, d_aux), aux_loss (float), stats (dict)
    E = len(experts); B = len(x)
    capacity = max(1, math.ceil(capacity_factor * k * B / E))
    freq = [0]*E; Psum = [0.0]*E; dropped = 0
    y = [[0.0]*d_aux for _ in range(B)]
    for b in range(B):
        h = matvec(W_g, x[b])                 # (E,)
        p = softmax(h)
        idx, gates = route_topk(h, k)
        for i in range(E): Psum[i] += p[i]
        for rank, (ei, g) in enumerate(zip(idx, gates)):
            if freq[ei] >= capacity: dropped += 1; continue
            freq[ei] += 1
            out = experts[ei](x[b])           # (d_aux,)
            for c in range(d_aux): y[b][c] += g * out[c]
    aux = load_balance_loss(freq, [v/B for v in Psum], alpha) if training else 0.0
    return y, aux, {"freq": freq, "capacity": capacity, "dropped": dropped}
```

### 6.4 复杂度

| 函数 | 时间 | 空间 |
|---|---|---|
| `route_topk` | `O(E log E)` | `O(E)` |
| `load_balance_loss` | `O(E)` | `O(1)` |
| `moe_forward`（含专家计算） | `O(B * (E*d + k * C_exp))`，`C_exp` 为单专家前向成本 | `O(B * d_aux + E)` |

### 6.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | 门控权重归一 | 每个 token：`abs(sum(gates) - 1.0)` | `1e-12` |
| I2 | top-k 基数 | `len(idx) == k` 且 `idx` 互异，且 `h[idx[0]] >= ... >= h[idx[k-1]]` | 精确 |
| I3 | 权重排序单调 | 若 `h_a > h_b` 且二者都在 `S` 内，则 `g_a > g_b` | 精确 |
| I4 | 确定性并列 | 两个 logit 完全相等时，索引小者优先（两次调用结果逐位相同） | 精确 |
| I5 | 计数守恒 | `sum_i freq[i] + dropped == B * k` | 精确（整数） |
| I6 | 容量约束 | `max_i freq[i] <= capacity` | 精确（整数） |
| I7 | 完美均衡时损失为 1 | `freq` 均匀、`P` 均匀时 `abs(loss_aux / alpha - 1.0)` | `1e-12` |
| I8 | top-1 可证下界 | 用真实 argmax 路由产生的 `(f, P)`：`(E * sum_i f_i P_i) - (sum_i f_i^2) >= -1e-9`，且 `sum_i f_i^2 >= 1/E - 1e-12` | `1e-9` |
| I9 | 上界 | `E * sum_i f_i P_i <= E * max_i P_i + 1e-12` 且 `<= E + 1e-12` | `1e-12` |
| I10 | `f` 与 `P` 各自归一 | `abs(sum(f) - 1)` 与 `abs(sum(P) - 1)` | `1e-12` |
| I11 | 恒等专家还原输入 | 所有 `experts` 为恒等映射时，`max abs(y[b] - x[b])` | `1e-12` |
| I12 | 容量充足则零丢弃 | `capacity_factor >= E/k * (max_load / B)` 时 `dropped == 0`；尤其 `capacity_factor >= 2` 且 `k=1` 的均匀负载下 `dropped == 0` | 精确 |
| I13 | 概率合法 | `0 <= P_i <= 1` 且 `f_i >= 0` | 精确 |

### 6.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| 训练几步后所有 token 走同一个专家，其余专家死掉 | 无辅助损失，或 `alpha` 太小 | `alpha` 典型 `0.01`（Switch）到 `0.1`（Mixtral 用更小）；同时加 router z-loss（`log(sum exp h)` 的平方）稳住 logit 量级 |
| 测试里 `assert aux_loss / alpha >= 1.0` 偶发失败（实测低至 `0.976`） | 误把"均衡点取值 = 1"当成了不等式下界 | **删除该断言**。改用可证下界 `E*sum f P >= sum f_i^2`（且 `sum f_i^2 >= 1/E`）与上界 `E*sum f P <= E*max(P)`。均衡点断言只在**人为构造的均匀 `f`、`P`** 上做（I7） |
| 同一输入两次路由结果不同 | `sorted` 对相等 logit 不稳定 | key 显式带索引：`(-h[i], i)` |
| `dropped` 计数与 `freq` 对不上 | 丢弃后仍加了 `freq`，或先加后判断 | 严格顺序：先判 `freq < capacity` 再 `+= 1`，且 `dropped` 只在被拒时 `+= 1`；用 I5 断言 |
| `capacity = 0` 导致全部丢弃 | `math.ceil` 前向下取整（用 `int()` 截断） | 用 `math.ceil` 并 `max(1, ...)`；`B < E` 的小批量尤其危险 |
| 门控权重之和 `1.0000001`，输出被放大 | 逐项 `sum` | `math.fsum` |
| 路由概率全相等（`p_i = 1/E`）时负载均衡损失仍很大 | `f` 用的是原始计数未除以 `k` | `sum_i f_i` 必须等于 `1`；`top-k` 时除以 `k*N` 而非 `N` |
| 辅助损失被加到 loss 里导致主任务退化 | `alpha` 过大 | 打印 `loss_main` 与 `loss_aux` 分项，若 `aux` 占比 > 5% 则调小 |

### 6.7 经典论文

- Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR. ---- 稀疏门控 MoE 与原始负载均衡损失。
- Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR. ---- top-1 路由 + 容量因子 + `E * sum f_i P_i` 辅助损失。
- Lepikhin, D., Lee, H., Xu, Y., et al. (2020). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. ICLR. ---- top-2 路由与专家容量工程实践。
- Jiang, A. Q., Sablayrolles, A., Roux, A., et al. (2024). *Mixtral of Experts*. arXiv:2401.04088. ---- top-2（后为 top-k）门控的开源实践。

---

## 7. `diffusion.py` -- 扩散模型前向 / 反向与采样器

### 7.1 直觉

前向过程把一张图用 `T` 步逐渐加成纯噪声；反向过程学一个网络去猜"这一步加进去的噪声是什么"，然后一步步减回去。DDPM 的反向是随机的（每步都注入新噪声，慢但质量高），DDIM 把它改造成确定性的常微分方程路径（可用 20-50 步完成 1000 步的采样）。cosine schedule 让 `beta` 在两端不要变化太剧烈，避免"开头几步就把信息全毁掉"。

### 7.2 数学核心

**前向（DDPM, Ho et al. 2020）**：

```
alpha_t     = 1 - beta_t
alpha_bar_t = prod_{s=1..t} alpha_s            # 用 exp(cumsum(log alpha)) 计算，防下溢
q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0 , (1 - alpha_bar_t) * I )
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
```

**反向均值（由 `eps_theta` 参数化）**：

```
x0_hat = ( x_t - sqrt(1 - alpha_bar_t) * eps_theta ) / sqrt(alpha_bar_t)      # (A)
mu_t   = (1/sqrt(alpha_t)) * ( x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta )
       = sqrt(alpha_bar_{t-1}) * x0_hat + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_theta
sigma_t^2 可选 beta_t（原始）或 beta_tilde_t = beta_t * (1-alpha_bar_{t-1})/(1-alpha_bar_t)（改进）
x_{t-1} = mu_t + sigma_t * z,   z ~ N(0, I)；最后一步（t -> 0）强制 sigma = 0
```

**DDIM（Song et al. 2020）**：

```
x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat
        + sqrt( max(1 - alpha_bar_{t-1} - sigma_t^2, 0) ) * eps_theta
        + sigma_t * z
sigma_t = eta * sqrt( (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) ) * sqrt( 1 - alpha_bar_t / alpha_bar_{t-1} )
eta = 0  -> 完全确定性（ODE）；eta = 1 -> 与 DDPM 的边际分布一致
```

**Cosine schedule（Nichol & Dhariwal 2021）**：

```
f(t)   = cos( ((t/T + s) / (1 + s)) * pi / 2 )^2 ,      s = 0.008
alpha_bar_t = f(t) / f(0)
beta_t = 1 - alpha_bar_t / alpha_bar_{t-1},  并 clamp 到 <= 0.999
```

### 7.3 伪代码

```python
def get_cosine_schedule(T, s=0.008, beta_max=0.999):
    # output: betas (T,), alphas (T,), alpha_bars (T,)
    f0 = math.cos((s/(1+s)) * math.pi/2)**2
    ab = []
    for t in range(1, T+1):
        ft = math.cos(((t/T + s)/(1+s)) * math.pi/2)**2
        ab.append(min(ft/f0, 1.0))
    betas = []
    prev = 1.0
    for t in range(T):
        b = 1.0 - ab[t]/prev
        betas.append(min(max(b, 0.0), beta_max))     # clamp，见坑 D5
        prev = ab[t]
    alphas = [1.0 - b for b in betas]
    return betas, alphas, ab

def q_sample(x0, t, alpha_bars, eps):
    # input:  x0 (D,), t int, alpha_bars (T,), eps (D,)   output: x_t (D,)
    ab = max(alpha_bars[t], 1e-8)                    # 防 0 除，见坑 D1
    return [math.sqrt(ab)*x0_i + math.sqrt(1.0-ab)*e_i for x0_i, e_i in zip(x0, eps)]

def predict_x0(xt, t, alpha_bars, eps_theta):
    # output: x0_hat (D,)
    ab = max(alpha_bars[t], 1e-8)
    return [(x - math.sqrt(1.0-ab)*e)/math.sqrt(ab) for x, e in zip(xt, eps_theta)]

def ddim_step(xt, t, t_prev, alpha_bars, eps_theta, eta, rng):
    # input:  xt (D,), 时间索引 t / t_prev, eta in [0,1]
    # output: x_{t_prev} (D,)
    ab_t  = max(alpha_bars[t], 1e-8)
    ab_p  = alpha_bars[t_prev] if t_prev >= 0 else 1.0
    x0h   = predict_x0(xt, t, alpha_bars, eps_theta)
    if eta == 0.0:
        sigma = 0.0
    else:
        sigma = eta*math.sqrt(max((1-ab_p)/(1-ab_t), 0.0))*math.sqrt(max(1-ab_t/ab_p, 0.0))
    coef  = math.sqrt(max(1.0 - ab_p - sigma*sigma, 0.0))
    z = [rng.gauss() for _ in xt] if sigma > 0 else [0.0]*len(xt)
    return [math.sqrt(ab_p)*a + coef*b + sigma*zz for a, b, zz in zip(x0h, eps_theta, z)]
```

### 7.4 复杂度

| 函数 | 时间 | 空间 |
|---|---|---|
| `get_cosine_schedule` | `O(T)` | `O(T)` |
| `q_sample` | `O(D)` 每步 | `O(D)` |
| `ddim_step` / `ddpm_step` | `O(D)` 每步 | `O(D)` |
| 完整采样（`S` 步） | `O(S * C_model + S * D)` | `O(D)`（只需保存 `x_t`） |

### 7.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | **真实噪声可精确还原 `x0`** | 用生成 `x_t` 时**同一个** `eps` 调用 `predict_x0`，则 `max abs(x0_hat - x0)` | `1e-10` |
| I2 | `alpha_bar` 单调不增 | `alpha_bars[t] <= alpha_bars[t-1] + 1e-15` 对所有 `t` | `1e-15` |
| I3 | `beta` 合法区间 | `0 <= beta_t <= 0.999` 对所有 `t`；`alpha_bar_T < 1e-3`；`alpha_bar_0 > 1 - 1e-6` | `1e-12` |
| I4 | 终点为标准正态 | 采样 `10^4` 条 `x_T`：`abs(mean) < 0.05`，`abs(std - 1) < 0.05` | `5e-2` |
| I5 | 终点与起点无关 | `x_0 = 5.0` 与 `x_0 = -5.0` 生成的 `x_T` 分布（各 `10^4` 样本）均值差 `< 0.05` | `5e-2` |
| I6 | DDIM 确定性 | `eta = 0` 时，同 `x_T`、同 `eps_theta` 跑两次，`max abs(diff) == 0.0` | 精确（逐位） |
| I7 | DDIM 单步到 `t=0` 等于 `x0_hat` | `t_prev` 对应 `alpha_bar = 1.0` 时，`max abs(x_out - x0_hat)` | `1e-12` |
| I8 | DDIM vs DDPM 分布一致 | `eta = 1` 与 DDPM（用 `beta_tilde` 方差）各采样 `10^4` 条，一阶矩差 `< 0.03`、二阶矩差 `< 0.05` | `5e-2` |
| I9 | 步数单调收敛 | 用真值 `eps` 时，DDIM `eta=0` 在任意步数 `S` 下还原的 `x0_hat` 与 `x0` 的误差与 `S` 无关（因为 `x0_hat` 只依赖当前 `x_t`） | `1e-10` |
| I10 | 方差非负 | `1 - alpha_bar_{t-1} - sigma_t^2 >= -1e-12`（`sqrt` 前已被 `max(...,0)` 保护） | `1e-12` |
| I11 | 无 `nan` | 整条采样链（含 `t=0`）输出中 `not any(isnan(v))` | 精确 |

### 7.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| `ZeroDivisionError` 或输出 `inf` | `alpha_bar_T` 太小（甚至为 0），`sqrt(alpha_bar)` 为 0 | 所有 `sqrt(alpha_bar)` 前 `max(ab, 1e-8)`；`alpha_bar` 用 `exp(cumsum(log alpha))` 计算而非连乘 |
| 最后一步输出有高频噪点 | 在 `t -> 0` 那一步仍加了 `sigma * z` | 最后一步强制 `sigma = 0`（伪代码里 `t_prev < 0` 分支） |
| `alpha_bar` 递减到负数 | cosine schedule 的 `beta` 未 clamp，`alpha_bar_t / alpha_bar_{t-1}` 跳变 | `beta` clamp 到 `[0, 0.999]`；clamp 后**必须**重新累积 `alpha_bar = cumprod(1-beta)`，否则 `alpha_bar` 与 `beta` 不自洽 |
| 索引差一导致 `IndexError: -1` | 用 `alpha_bars[t-1]` 而 `t=0` | 统一约定：`t` 从 `0` 开始，`alpha_bars[0]` 对应第一步；`t_prev = -1` 时取 `alpha_bar = 1.0`（表示 `x_0`） |
| DDIM 与 DDPM 结果差很多 | 误以为两者逐步相等 | 两者只在**边际分布**上一致（I8），逐步路径不同；断言只能比较分布矩，不能比较单点 |
| `eta=0` 两次运行结果不同 | 用了未固定 seed 的 `rng` 或在 `sigma=0` 时仍调用了 `rng.gauss()`（消耗了随机流不影响结果，但如果 `sigma>0` 分支误入则会变） | `sigma == 0` 时**不**调用 RNG（保证随机流干净） |
| cosine schedule 在 `t` 很小时 `beta` 接近 0，训练信号过弱 | 正常的 schedule 特性 | 保持 `s = 0.008`；若需更强早期信号，改用 linear schedule 做对照 |
| 反向采样方差在 `t` 小时爆炸 | 用了 `beta_tilde` 但 `alpha_bar_{t-1}` 取错 | `beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)`，`alpha_bar_{-1} := 1` |

### 7.7 经典论文

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. ---- DDPM。
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML. ---- 扩散模型奠基。
- Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. ICLR. ---- DDIM。
- Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML. ---- cosine noise schedule 与 `beta_tilde` 方差选择。
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., et al. (2020). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR. ---- DDIM 的 ODE / SDE 统一视角。

---

## 8. `rl.py` -- 优势估计与策略优化

### 8.1 直觉

GAE 回答"这个动作比平均水平好多少"，用一个 `lambda` 在"只看一步 TD 误差"和"看完整条轨迹的蒙特卡洛回报"之间连续插值。PPO 回答"怎么更新策略才不会一次更新跑飞"----把新旧策略的概率比裁到 `[1-eps, 1+eps]`，并取裁剪前后的**悲观值**（min），从而在不加二阶约束的情况下得到单调改进的下界。

### 8.2 数学核心

**GAE（Schulman et al. 2015）**：

```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)              # TD 残差
A_t^{GAE(gamma,lambda)} = sum_{l=0}^{inf} (gamma*lambda)^l * delta_{t+l}
反向递推:  A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
```

端点特例（把 `lambda` 代入即可验证）：

```
lambda = 0 :  A_t = delta_t                                   （一步 TD 残差）
lambda = 1 :  A_t = sum_{l=0}^{L} gamma^l r_{t+l} + gamma^{L+1} V_{t+L+1} - V(s_t)
                  = R_t - V(s_t)                              （蒙特卡洛回报减基线）
```
中间那步的裂项相消：
```
sum_{l=0..L} gamma^l delta_{t+l}
 = sum_{l} gamma^l r_{t+l} + sum_{k=1..L+1} gamma^k V_{t+k} - sum_{k=0..L} gamma^k V_{t+k}
 = sum_{l} gamma^l r_{t+l} + gamma^{L+1} V_{t+L+1} - V_t
```

**PPO clipped surrogate（Schulman et al. 2017）**：

```
ratio:      r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
                       = exp( log pi_theta(a_t|s_t) - log pi_theta_old(a_t|s_t) )     # 必须走对数域
L^CLIP(theta) = E_t[ min( r_t(theta) * A_t , clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]
L^VF(theta)   = E_t[ max( (V_theta(s_t) - V_targ_t)^2 ,
                          (V_old(s_t) + clip(V_theta - V_old, -eps, +eps) - V_targ_t)^2 ) ]
L = -( L^CLIP - c1 * L^VF + c2 * S[pi_theta] )          # 优化器最小化 L
```
`A_t > 0` 时，裁剪阻止 `r_t` 超过 `1+eps`（防止过度放大好动作）；`A_t < 0` 时，裁剪阻止 `r_t` 低于 `1-eps`（防止过度压制坏动作）。取 `min` 保证目标是真实目标的**下界**（pessimistic bound）。

### 8.3 伪代码

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95,
                last_value=0.0):
    # input:  rewards (T,), values (T+1,) 或 (T,) 若为后者则补 last_value
    #         dones (T,) bool, gamma float, lam float
    # output: advantages (T,), returns (T,)
    T = len(rewards)
    if len(values) == T: values = values + [last_value]
    adv = [0.0]*T
    gae = 0.0
    for t in range(T-1, -1, -1):                       # 必须反向
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma*values[t+1]*nonterminal - values[t]
        gae = delta + gamma*lam*nonterminal*gae         # done 处截断（见坑 R1）
        adv[t] = gae
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns

def ppo_clip_loss(logp, logp_old, adv, eps=0.2):
    # input:  logp (N,), logp_old (N,), adv (N,)   output: (loss (N,), ratio (N,), clip_frac)
    out = []
    for lp, lpo, a in zip(logp, logp_old, adv):
        r = safe_exp(lp - lpo)                          # 对数域计算比值
        rc = min(max(r, 1.0-eps), 1.0+eps)
        out.append(min(r*a, rc*a))
    clip_frac = sum(1 for lp, lpo in zip(logp, logp_old)
                    if abs(safe_exp(lp-lpo) - 1.0) > eps) / len(logp)
    return out, [safe_exp(lp-lpo) for lp, lpo in zip(logp, logp_old)], clip_frac

def value_clip_loss(v, v_old, v_targ, eps=0.2):
    # input:  (N,) x3   output: (N,)
    res = []
    for a, b, t in zip(v, v_old, v_targ):
        d = a - b
        vc = b + min(max(d, -eps), eps)
        res.append(max((a - t)**2, (vc - t)**2))
    return res
```

### 8.4 复杂度

| 函数 | 时间 | 空间 |
|---|---|---|
| `compute_gae` | `O(T)` | `O(T)` |
| `ppo_clip_loss` | `O(N)` | `O(N)` |
| `value_clip_loss` | `O(N)` | `O(N)` |

GAE 对长为 `T` 的轨迹必须反向单趟扫描，这是 `O(T)` 的关键（正向做需要 `O(T^2)` 或截断）。

### 8.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | **`lambda=0` 退化为 TD 残差** | `lam=0` 时 `max_t abs(A_t - delta_t)`，其中 `delta_t = r_t + gamma*V_{t+1} - V_t` | `1e-12` |
| I2 | **`lambda=1` 退化为 MC 回报减基线** | `lam=1` 时 `max_t abs((A_t + V_t) - R_t)`，`R_t = sum_{l} gamma^l r_{t+l} + gamma^{T-t} V_T` | `1e-10` |
| I3 | 优势与回报自洽 | `max_t abs((A_t + V_t) - returns_t)` 对**任意** `lambda` 成立 | `1e-12` |
| I4 | episode 边界截断 | 令 `dones[t0] = True`，则 `A_{t < t0}` 与把轨迹在 `t0` 处切开后独立计算的 `A` 逐位一致 | `1e-12` |
| I5 | 终点处理 | `dones[T-1] = True` 时 `A_{T-1} + V_{T-1} == r_{T-1}`（不 bootstrap） | `1e-12` |
| I6 | 归一化 | `normalize_advantages` 后 `abs(mean(A)) < 1e-12`，`abs(std(A) - 1) < 1e-12`（**只对未 padding 的有效步**统计） | `1e-12` |
| I7 | **PPO `ratio=1` 时 surrogate == advantage** | 令 `logp == logp_old`，则 `max abs(L^CLIP_t - A_t)` | `1e-12` |
| I8 | PPO 裁剪饱和 | `A_t > 0` 且 `r > 1+eps` 时 `L == (1+eps)*A`；`A_t < 0` 且 `r < 1-eps` 时 `L == (1-eps)*A` | `1e-12` |
| I9 | PPO 悲观下界 | `L^CLIP_t <= r_t * A_t + 1e-12` 恒成立 | `1e-12` |
| I10 | 比值恒等 | `logp == logp_old` 时 `abs(r - 1.0)`；且 `r == exp(logp - logp_old)` 与直接概率比一致 | `1e-12` |
| I11 | 价值裁剪无副作用 | `abs(v - v_old) <= eps` 时 `L^VF == (v - v_targ)^2` | `1e-12` |
| I12 | 价值裁剪悲观 | `L^VF >= (v - v_targ)^2 - 1e-12` 恒成立 | `1e-12` |
| I13 | 熵范围 | `0 <= S <= log(\|A\|)`；均匀分布时 `abs(S - log(\|A\|))` | `1e-12` |
| I14 | 裁剪比例合理 | 训练初期 `clip_frac` 在 `[0, 0.2]`；若 `> 0.5` 说明学习率过大（作为监控断言而非硬断言） | 监控 |

### 8.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| 优势在 episode 结束后仍把下一段的回报"漏"过来 | 递推时没有乘 `(1 - done_t)` | `gae = delta + gamma*lam*nonterminal*gae`，`nonterminal = 0` 时**整个**累积量清零 |
| 训练初期 loss 突然爆炸 | 概率比在概率域直接算（`p_new / p_old`），极小概率下浮点下溢为 `inf` | 一律 `exp(logp - logp_old)`；`logp` 来自 `log_softmax` |
| `A_t` 与 `returns_t` 对不上 | `values` 长度是 `T` 而非 `T+1`，最后一步用了错误的 bootstrap | 统一要求 `len(values) == T+1`；若调用方只给 `T` 个，必须显式传 `last_value` |
| 归一化后 `mean != 0` | 对 padding 的 0 也做了统计 | 归一化接受 `mask`，只统计有效步；断言用 I6 |
| 优势全为 0 | `values` 用成了 `V_targ`（回报）而非 `V(s_t)`（价值估计） | 类型命名区分 `values`（critic 输出）与 `returns`（目标） |
| PPO loss 变成正数且越来越大 | 符号写反（把 `min` 写成 `max`，或 loss 少了负号） | 用 I9 断言：`L^CLIP <= r*A`；若违反即符号错 |
| 裁剪后策略几乎不更新 | `eps` 太小（如 `0.001`）或 `clip_frac` 接近 1 | 监控 `clip_frac`，典型 `eps = 0.2`（MuJoCo）/ `0.1`（RLHF） |
| 熵项把 loss 带向负无穷 | 熵项符号写错（写成 `loss += c2*H` 而非 `-=`，取决于 loss 定义） | 明确约定：`objective = L^CLIP - c1*L^VF + c2*H`，优化器最小化 `-objective`；用 I13 断言熵非负 |
| `lambda` 参数与 `gamma` 混淆 | 调用时位置参数写反 | 一律关键字传参；I1/I2 两个端点断言能立刻暴露 |

### 8.7 经典论文

- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438. ---- GAE。
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. ---- PPO clipped objective 与价值裁剪。
- Kakade, S., & Langford, J. (2002). *Approximately Optimal Approximate Reinforcement Learning*. ICML. ---- 保守策略迭代（PPO 的理论前身）。
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). *Trust Region Policy Optimization*. ICML. ---- TRPO，PPO 的一阶近似对象。
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. ---- TD 残差与 n-step 回报的标准参考。

---

## 9. `gnn.py` -- 图传播与谱工具

### 9.1 直觉

图神经网络的本质是"把自己的邻居的信息按某种权重平均过来"。GCN 用的是加了自环再做**对称归一化**的邻接矩阵，这一步归一化是关键----不做的话，高度数节点的特征量级会爆炸，`L` 层之后数值失控。GraphSAGE 的均值聚合更朴素也更省内存，适合大规模和归纳式（inductive）场景。

### 9.2 数学核心

记号：`A` 邻接矩阵 `(n,n)`（无自环），`I` 单位阵，`A~ = A + I`，`D~ = diag(A~ 1)`，`D = diag(A 1)`。

```
GCN 传播（Kipf & Welling 2016）:
    A_hat = D~^{-1/2} * A~ * D~^{-1/2}
    H^{(l+1)} = sigma( A_hat * H^{(l)} * W^{(l)} )
    单层无激活无变换时: H_out = A_hat @ H

GraphSAGE 均值聚合（Hamilton et al. 2017）:
    h_v^{(l+1)} = sigma( W * MEAN( {h_v^{(l)}} u {h_u^{(l)} : u in N(v)} ) )
    即: AGG = MEAN over {self} u N(v)，注意集合包含自身

对称归一化 Laplacian:
    L_sym = I - D^{-1/2} A D^{-1/2}
    二次型:  x^T L_sym x = sum_{(u,v) in E} ( x_u/sqrt(d_u) - x_v/sqrt(d_v) )^2
    谱范围:  eigenvalues(L_sym) in [0, 2]；0 特征向量为 D^{1/2} * 1
    对应:    eigenvalues(D^{-1/2} A D^{-1/2}) in [-1, 1]
             eigenvalues(A_hat) in (-1, 1]，rho(A_hat) == 1
```

`L_sym * (D^{1/2} 1) = 0` 的验证：
```
L_sym D^{1/2} 1 = D^{1/2} 1 - D^{-1/2} A D^{-1/2} D^{1/2} 1
                = D^{1/2} 1 - D^{-1/2} A 1
                = D^{1/2} 1 - D^{-1/2} d
                = D^{1/2} 1 - D^{1/2} 1 = 0
```

### 9.3 伪代码

```python
def add_self_loops(A):
    # input:  A (n,n)   output: A~ (n,n)
    return [[A[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]

def degree(A):
    # input:  A (n,n)   output: d (n,)
    return [math.fsum(row) for row in A]

def sym_normalize(A):
    # input:  A (n,n) 邻接（可含自环）  output: A_hat (n,n)
    d = degree(A)
    di = [1.0/math.sqrt(max(x, 1.0)) for x in d]      # 孤立点保护，见坑 G1
    return [[di[i]*A[i][j]*di[j] for j in range(n)] for i in range(n)]

def gcn_propagate(A, H, W=None, activation=None):
    # input:  A (n,n) 原始邻接, H (n, d_in), W (d_in, d_out) 可选
    # output: H_out (n, d_out)
    A_hat = sym_normalize(add_self_loops(A))
    Z = matmul(A_hat, H)                               # (n, d_in)
    if W is not None: Z = matmul(Z, W)                 # (n, d_out)
    return [activation(z) for z in Z] if activation else Z

def sage_mean_aggregate(adj_list, H):
    # input:  adj_list: list[list[int]]（邻居索引，不含自身）, H (n, d)
    # output: AGG (n, d)
    out = []
    for v in range(len(H)):
        nb = adj_list[v]
        k = len(nb) + 1                                # 含自身
        out.append([ (H[v][c] + math.fsum(H[u][c] for u in nb)) / k
                     for c in range(len(H[v])) ])
    return out

def sym_laplacian(A):
    # input:  A (n,n)   output: L_sym (n,n)
    d = degree(A)
    di = [1.0/math.sqrt(max(x, 1.0)) for x in d]
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0 - di[i]*A[i][i]*di[i]            # A 通常无自环 -> 1.0
        for j in range(n):
            if i != j and A[i][j] != 0.0:
                L[i][j] = -di[i]*A[i][j]*di[j]
    return L
```

### 9.4 复杂度

| 函数 | 时间 | 空间 | 瓶颈 |
|---|---|---|---|
| `sym_normalize` | `O(n^2)` | `O(n^2)` | 稠密 `n^2`；`n > 5000` 必须换稀疏表示 |
| `gcn_propagate` | `O(n^2 * d_in + n * d_in * d_out)` | `O(n^2)` | 邻接-特征乘法 |
| `sage_mean_aggregate` | `O(\|E\| * d)` | `O(n*d)` | 只扫边，稀疏友好 |
| `sym_laplacian` | `O(n^2)` | `O(n^2)` | 同 `sym_normalize` |

### 9.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | **Laplacian 零特征向量** | `max_i abs((L_sym @ (D^{1/2} 1))[i])`，且该向量非零 | `1e-10` |
| I2 | `L_sym` 半正定 | 随机 `x`（100 组）：`x^T L_sym x >= -1e-10` | `1e-10` |
| I3 | 二次型恒等式 | `abs(x^T L_sym x - sum_{(u,v) in E} (x_u/sqrt(d_u) - x_v/sqrt(d_v))^2)` | `1e-10` |
| I4 | **谱半径上界** | `rho(D^{-1/2} A D^{-1/2}) <= 1 + 1e-8`（幂迭代求）；`A_hat` 的 `rho == 1`（连通图） | `1e-8` |
| I5 | `L_sym` 谱范围 | 用 Jacobi 分解：`min(eig) >= -1e-8` 且 `max(eig) <= 2 + 1e-8` | `1e-8` |
| I6 | 对称性 | `max_ij abs(A_hat[i][j] - A_hat[j][i])`；同样对 `L_sym` | `1e-12` |
| I7 | 随机游走归一化行和为 1 | `max_i abs(fsum_j (D~^{-1} A~)[i][j] - 1.0)` | `1e-12` |
| I8 | 置换等变性 | 对置换矩阵 `P`：`max abs(P A_hat P^T - sym_normalize(P A P^T))` | `1e-12` |
| I9 | 恒等传播 | `A = 0`（无边，只有自环）时 `gcn_propagate(A, H) == H` | `1e-12` |
| I10 | SAGE 孤立点 | `adj_list[v] == []` 时 `AGG[v] == H[v]` | `1e-12` |
| I11 | SAGE 均值有界 | `min over {v} u N(v) of h <= AGG[v] <= max over {v} u N(v) of h` | `1e-12` |
| I12 | 度守恒 | `fsum(D~ 1) == n + 2*\|E\|`（自环数 + 2 倍边数，无向图） | `1e-10` |
| I13 | 无 `nan` | 含孤立点的图经 `sym_normalize` 后 `not any(isnan)` | 精确 |

### 9.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| 输出出现 `inf` / `nan` | 孤立节点度数为 0，`1/sqrt(0)` | `max(d, 1.0)` 保护；更好的做法是 GCN 的 `A~ = A + I` **先加自环**，此时孤立点度数至少为 1（这是加自环的隐藏好处） |
| 用了 `D`（来自 `A`）而不是 `D~`（来自 `A + I`）做归一化 | 自环加了但度数没重算 | 归一化用的度必须与加了自环的矩阵**同源**；加断言 `fsum(D~) == n + 2*\|E\|` |
| `A_hat` 的最大特征值算出 `1.3` | 归一化写成了 `D^{-1} A`（随机游走）却按对称归一化断言 | 两种归一化不可混用：`D^{-1}A` 行和为 1 但**不对称**；`D^{-1/2} A D^{-1/2}` 对称但行和不为 1。用 I6 + I7 区分 |
| 2 层 GCN 后特征量级爆炸到 `1e6` | 未做对称归一化 | 用 I4 断言 `rho <= 1`；`A_hat` 的谱半径恒为 1，多层传播不会放大 |
| 稀疏图用稠密 `n x n` 内存爆掉 | `n = 10^5` 时 `n^2 = 10^10` | `sage_mean_aggregate` 走 `adj_list`；`gcn_propagate` 在 `n > 5000` 时必须接受 `(row_idx, col_idx, val)` 三元组形式 |
| 自监督/归纳场景下测试集节点被"泄漏" | 用了全图的 `A_hat` 做 transductive 归一化 | SAGE 的聚合只用采样邻居，天然归纳；GCN 是全图传播，测试时须重新切子图 |
| 无向图存成了单向边，导致 `A` 不对称 | 建图时只加 `(u,v)` 没加 `(v,u)` | 入口断言 `max abs(A - A^T) < 1e-12` |
| `L_sym` 算出小的负特征值 `-1e-9` | float 误差 | 容差用 `1e-8`（迭代法），不要写 `>= 0` 的硬断言 |

### 9.7 经典论文

- Kipf, T. N., & Welling, M. (2016). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR. ---- GCN 与对称归一化传播。
- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS. ---- GraphSAGE 与均值 / LSTM / 池化聚合器。
- Chung, F. R. K., & Graham, F. C. (1997). *Spectral Graph Theory*. AMS. ---- 归一化 Laplacian 的谱性质（`[0,2]` 谱范围）。
- Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*. NeurIPS. ---- ChebNet，GCN 的一阶近似来源。
- Velickovic, P., Cucurull, G., Casanova, A., et al. (2018). *Graph Attention Networks*. ICLR. ---- 作为注意力式聚合的对照实现。

---

## 10. `bayes.py` -- 共轭更新与变分推断

### 10.1 直觉

共轭的意义是"后验和先验长得一样，只改几个数字"----Beta-Binomial 只改 `a`、`b` 两个计数，高斯的精度只做加法。变分推断处理的是"后验算不出来"的情形：用一个可分解的 `q` 去逼近真实后验，然后**逐坐标**轮流优化（坐标上升），每步都保证 ELBO 不下降。

### 10.2 数学核心

**单变量高斯共轭（已知观测方差 `s2`，未知均值 `mu`）**：

先验 `mu ~ N(m, v)`，观测 `x ~ N(mu, s2)`：

```
v_post = 1 / (1/v + 1/s2) = v*s2 / (v + s2)
m_post = v_post * ( m/v + x/s2 ) = (m*s2 + x*v) / (v + s2)
批次形式:  1/v_n = 1/v_0 + n/s2 ;  m_n = v_n * ( m_0/v_0 + (sum x_i)/s2 )
```

**Beta-Binomial**：

```
先验 Beta(a, b)，观测 n 次试验中 k 次成功
后验:  a_post = a + k ,  b_post = b + (n - k)
后验均值 = (a+k)/(a+b+n)
         = ((a+b)/(a+b+n)) * (a/(a+b)) + (n/(a+b+n)) * (k/n)
           \_______ 权重 _______/   \__ 先验均值 __/     \__ MLE __/
```
即后验均值是先验均值与 MLE 的**凸组合**，权重由"先验样本量"与"数据量"决定。

**变分推断（CAVI）----单变量高斯，未知均值 `mu` 与未知精度 `tau`**：

模型（Blei et al. 2016 风格的均值场示例）：

```
tau   ~ Gamma(a0, rate b0)              E[tau] = a/b
mu    ~ N(m0, s0^2)                     （与 tau 独立的先验）
x_i   ~ N(mu, 1/tau),  i = 1..n
q(mu, tau) = q(mu) * q(tau)
```

坐标上升更新（记 `E[tau] = a_N/b_N`，`xbar = mean(x)`）：

```
q(mu)  = N(m_N, s_N^2):
    1/s_N^2 = 1/s0^2 + n * E[tau]
    m_N     = s_N^2 * ( m0/s0^2 + n * E[tau] * xbar )

q(tau) = Gamma(a_N, rate b_N):
    a_N = a0 + n/2
    b_N = b0 + 0.5 * sum_i [ (x_i - m_N)^2 + s_N^2 ]      # <-- 必须加 s_N^2，见坑 B4
```

ELBO：

```
ELBO = E[log p(mu)] + E[log p(tau)] + sum_i E[log p(x_i|mu,tau)]
       - E[log q(mu)] - E[log q(tau)]

E[log p(mu)]  = -0.5*log(2 pi s0^2) - 0.5*((m_N - m0)^2 + s_N^2)/s0^2
E[log p(tau)] = a0*log(b0) - lgamma(a0) + (a0-1)*E[log tau] - b0*E[tau]
E[log p(x_i|mu,tau)] = 0.5*E[log tau] - 0.5*log(2 pi) - 0.5*E[tau]*((x_i - m_N)^2 + s_N^2)
H[q(mu)]  = 0.5*log(2 pi e s_N^2)
H[q(tau)] = a_N - log(b_N) + lgamma(a_N) + (1 - a_N)*psi(a_N)

其中  E[log tau] = psi(a_N) - log(b_N),   E[tau] = a_N / b_N
```

`psi`（digamma）需自实现（无 scipy）：

```
递推:      psi(x) = psi(x + k) - sum_{i=0..k-1} 1/(x + i)      # 把 x 抬到 >= 6
渐近:      psi(x) ~ log(x) - 1/(2x) - 1/(12 x^2) + 1/(120 x^4)
                    - 1/(252 x^6) + 1/(240 x^8) - 1/(132 x^10)     (x >= 6)
```
`lgamma` 直接用 `math.lgamma`（标准库自带）。

### 10.3 伪代码

```python
def gaussian_conjugate_update(m, v, x, s2):
    # input:  先验 N(m, v)，观测标量 x，观测方差 s2
    # output: (m_post, v_post)
    prec = 1.0/v + 1.0/s2
    v_post = 1.0/prec
    m_post = v_post * (m/v + x/s2)
    return m_post, v_post

def beta_binomial_update(a, b, k, n):
    # input:  先验 Beta(a,b)，n 次试验 k 次成功   output: (a_post, b_post)
    assert 0 <= k <= n
    return a + k, b + (n - k)

def digamma(x):
    # input:  x > 0   output: psi(x)
    r = 0.0
    while x < 6.0:
        r -= 1.0/x; x += 1.0
    f = 1.0/(x*x)
    s = math.log(x) - 0.5/x + f*(-1.0/12 + f*(1.0/120 + f*(-1.0/252
        + f*(1.0/240 + f*(-1.0/132)))))
    return s + r

def cavi_gaussian(x, m0, s0sq, a0, b0, iters=200, tol=1e-12):
    # input:  x (n,) 数据；先验超参   output: dict(m_N, s_N_sq, a_N, b_N, elbo_history)
    n = len(x); xbar = math.fsum(x)/n
    mN, sNsq = m0, s0sq
    aN, bN = a0, b0
    hist = []
    for it in range(iters):
        Etau = aN/bN
        # --- q(mu) ---
        sNsq = 1.0/(1.0/s0sq + n*Etau)
        mN   = sNsq*(m0/s0sq + n*Etau*xbar)
        # --- q(tau) ---
        aN = a0 + 0.5*n
        bN = b0 + 0.5*math.fsum((xi - mN)**2 + sNsq for xi in x)
        hist.append(elbo_gaussian(x, mN, sNsq, aN, bN, m0, s0sq, a0, b0))
        if it >= 1 and abs(hist[-1] - hist[-2]) < tol: break
    return {"m_N": mN, "s_N_sq": sNsq, "a_N": aN, "b_N": bN, "elbo": hist}
```

### 10.4 复杂度

| 函数 | 时间 | 空间 |
|---|---|---|
| `gaussian_conjugate_update` | `O(1)` | `O(1)` |
| `beta_binomial_update` | `O(1)` | `O(1)` |
| `digamma` | `O(1)`（递推步数有界） | `O(1)` |
| `cavi_gaussian` | `O(iters * n)` | `O(1)`（不含 ELBO 历史） |

### 10.5 可验证不变量

| # | 不变量 | 数学表达式 | 容差 |
|---|---|---|---|
| I1 | 方差收缩 | `v_post <= v` 且 `v_post <= s2`（信息永不减少） | `1e-12` |
| I2 | 后验均值夹逼 | `min(m, x) <= m_post <= max(m, x)` | `1e-12` |
| I3 | 精度加权形式 | `abs(m_post - (m/v + x/s2)/(1/v + 1/s2))` | `1e-12` |
| I4 | 批次 == 序贯 | 一次用 `n` 个观测 vs 逐个更新 `n` 次：`abs(m_batch - m_seq)` | `1e-10` |
| I5 | 平坦先验退化为 MLE | `v = 1e8` 时 `abs(m_post - xbar)` | `1e-6` |
| I6 | 大样本收敛到样本均值 | `n = 10^4` 时 `abs(m_N - xbar)` | `1e-6` |
| I7 | Beta 计数精确 | `a_post + b_post == a + b + n`（整数精确相等） | 精确（`==`） |
| I8 | **Beta 后验均值为凸组合** | `abs( (a+k)/(a+b+n) - ( w*(a/(a+b)) + (1-w)*(k/n) ) )`，`w = (a+b)/(a+b+n)` | `1e-12` |
| I9 | Beta 后验均值夹逼 | `min(a/(a+b), k/n) - tol <= (a+k)/(a+b+n) <= max(a/(a+b), k/n) + tol` | `1e-12` |
| I10 | Beta 序贯 == 批次 | 逐个 Bernoulli 更新 vs 一次 `(k, n)` 更新，结果相等 | `1e-12` |
| I11 | 顺序无关 | 先更新 `D1` 再 `D2` == 先 `D2` 再 `D1` | `1e-10` |
| I12 | **ELBO 单调不减（核心）** | `min_t (ELBO_{t+1} - ELBO_t) >= -1e-9` | `1e-9` |
| I13 | ELBO 收敛 | `abs(ELBO_last - ELBO_prev) < 1e-10` | `1e-10` |
| I14 | 空数据时 ELBO 为 0 | `n = 0` 且 `q` 初始化为先验时 `abs(ELBO)` | `1e-10` |
| I15 | 固定 `tau` 时 CAVI 一步收敛到共轭解 | 把 `n*E[tau]` 换成 `n*tau_true` 固定住，单次 `q(mu)` 更新结果与 Sec.10.1 共轭公式一致 | `1e-10` |
| I16 | digamma 精度 | `abs(psi(x) - (lgamma(x+1e-6) - lgamma(x-1e-6))/(2e-6))`，`x in {0.5, 1, 3, 10, 100}` | `1e-6` |
| I17 | digamma 递推 | `abs(psi(x+1) - psi(x) - 1/x)` | `1e-12` |
| I18 | 正参数 | 迭代全程 `s_N^2 > 0` 且 `a_N > 0` 且 `b_N > 0`，且无 `nan` | 精确 |
| I19 | 大样本恢复真值 | 用 `tau_true` 生成 `n = 2000` 样本，`abs(E[tau]/tau_true - 1) < 0.05` | `5e-2` |

### 10.6 常见坑 / 失败模式

| 现象 | 根因 | 处置 |
|---|---|---|
| **ELBO 在迭代中下降**（最常见也最致命） | `q(tau)` 更新里漏了 `+ s_N^2` 项：`E_q(mu)[(x_i - mu)^2] = (x_i - m_N)^2 + s_N^2` | 用 I12 的单调性断言在 CI 中强制；这一条能抓出 90% 的 CAVI 实现错误 |
| ELBO 出现 `-inf` | `s_N^2` 初始化为 0，或 `log(0)` | 初始化用先验 `s0^2`；所有 `log` 走 `safe_log` |
| `digamma` 在 `x < 1` 时误差 `1e-3` | 直接用了渐近公式而没做递推抬升 | 必须先 `while x < 6: r -= 1/x; x += 1`；用 I16/I17 交叉验证 |
| Gamma 参数化混乱（rate vs scale） | 一处用 `mean = a*b`（scale），另一处用 `a/b`（rate） | 全模块统一用 **rate**：`mean = a/b`，`E[log tau] = psi(a) - log b`；在 docstring 里写死 |
| 后验均值跑到先验均值与观测值之外 | 精度算成了方差（`1/v` 写成 `v`） | 用 I2 的夹逼断言和 I3 的加权形式断言 |
| Beta 更新后 `a_post + b_post` 不等于 `a + b + n` | 用了浮点计数，或 `k`/`n` 传反 | `k`、`n` 必须是 `int`；用 I7 的整数相等断言 |
| CAVI 不收敛，震荡 | `q(mu)` 与 `q(tau)` 在同一次迭代里用了"半新半旧"的值 | 明确更新顺序：先算 `E[tau]`（旧的 `a_N/b_N`）-> 更新 `q(mu)` -> 用**新的** `m_N, s_N^2` 更新 `q(tau)`。顺序反了会导致震荡 |
| `n = 0` 时 `xbar` 除零 | 均值计算未保护 | 入口断言 `n > 0`；I14 的空数据测试走独立分支 |
| ELBO 与 `log p(x)` 比较时"上界"被违反 | 用错了 ELBO 项（如把熵的符号写反） | 熵项必须**减去** `E[log q]`；`H[q(mu)] = 0.5*log(2 pi e s_N^2)` 是**正**的，ELBO = 能量项 **-** 熵项（注意 `E[log q] = -H`） |

### 10.7 经典论文

- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2016). *Variational Inference: A Review for Statisticians*. JASA 112(518). ---- CAVI 与 ELBO 的标准推导（本文档 Sec.10.2 变分部分的直接参考）。
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. ---- 共轭族与变分推断章节。
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). *Stochastic Variational Inference*. JMLR. ---- 可扩展变分推断（本模块的后续演进方向）。
- Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). *An Introduction to Variational Methods for Graphical Models*. Machine Learning 37. ---- 均值场变分的经典教程。
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. ---- Beta-Binomial 与高斯共轭的标准表。

---

## 11. 实现顺序与风险分级

### 11.1 推荐实现顺序（依赖 + 风险双排序）

| 阶段 | 模块 | 理由 |
|---|---|---|
| P0 | `numeric.py` | 所有模块的地基，先做并先测 |
| P1 | `prob.py`、`gnn.py` | 纯代数，无迭代，不变量易验证 |
| P2 | `linalg.py`、`bayes.py` | 需要迭代与自实现特殊函数（`digamma`），不变量仍清晰 |
| P3 | `rl.py`、`moe.py` | 逻辑分支多（done 截断、容量约束），需要构造性测试 |
| P4 | `sampling.py`、`diffusion.py` | 涉及随机性，不变量多为统计性质，需要固定 seed |
| P5 | `attention.py` | 最复杂，必须等 `numeric` 稳定后做等价性对拍 |

### 11.2 风险矩阵

| 模块 | 风险 | 主要风险点 |
|---|---|---|
| `numeric.py` | 低 | 分支多但无状态 |
| `prob.py` | 低 | 约定（log 底、0log0）易不一致 |
| `gnn.py` | 低 | 孤立点、稀疏性 |
| `bayes.py` | 中 | CAVI 更新顺序 + `digamma` 精度 |
| `linalg.py` | 中 | Jacobi 性能（`O(S n^3)`）、非正定诊断 |
| `rl.py` | 中 | done 掩码、对数域比值 |
| `moe.py` | 中 | 负载损失的下界在 top-k 下不成立，容易写出错误断言 |
| `sampling.py` | 中高 | 排序稳定性决定可复现性 |
| `diffusion.py` | 中高 | 索引约定（`t` 从 0 还是 1）、`alpha_bar` 下溢 |
| `attention.py` | **高** | online softmax 的 `alpha` 折算；全 mask 行的 `nan`；块大小不整除 |

### 11.3 统一验收测试清单（建议放在 `tests/test_kernel_invariants.py`）

1. 每个模块的不变量表逐条转成一个 `test_*` 函数，函数名含不变量编号（如 `test_attention_I1_tiled_vs_naive`）。
2. 所有随机性测试用固定 `PCG32(seed=20250101)`，禁止裸 `random`。
3. 交叉验证：至少 3 处**跨模块**对照
   - `attention.py` 的 `logsumexp` 与 `numeric.py` 的 `logsumexp` 一致（`1e-12`）
   - `bayes.py` 的固定 `tau` 的 CAVI 与 Sec.10.1 共轭公式一致（`1e-10`）
   - `gnn.py` 的 `L_sym` 特征值由 `linalg.py` 的 `jacobi_eig` 求解并验证 `[0, 2]` 谱范围（`1e-8`）
4. numpy 可选路径开启时，跑**同一套**测试（容差按 Sec.0.3 分级）。
5. CI 中把 `ATOL` 常量集中导入，禁止在测试里硬编码魔数。

---

## 12. 交付与署名

- 文档版本：v1.0
- 覆盖模块：10 / 10
- 不变量总数：124 条可断言性质（每个模块 10-18 条）
- 已用纯 Python 数值预验的关键不变量（实测误差）：

| 不变量 | 实测 | 声称容差 | 结论 |
|---|---|---|---|
| tiled attention vs 朴素（`Bc=1/3/7/16/40`） | `1.1e-16 ~ 2.2e-16` | `1e-9` | 通过，且块大小无关性成立 |
| GAE `lambda=1` 退化为 MC 回报 | `1.8e-15` | `1e-10` | 通过 |
| GAE `lambda=0` 退化为 TD 残差 | `0.0` | `1e-12` | 通过 |
| diffusion 用真值 `eps` 还原 `x0` | `0.0` | `1e-10` | 通过 |
| `L_sym @ D^{1/2} 1 == 0` | `5.6e-16` | `1e-10` | 通过 |
| `L_sym` 半正定（300 图 x 20 向量） | 最小二次型 `+1.1e-2` | `> -1e-10` | 通过 |
| `JS <= ln 2`（2000 组随机分布） | `0.692630 < 0.693147` | `ln2` | 通过 |
| CAVI ELBO 单调不减 | 最小增量 `-5.7e-14` | `>= -1e-9` | 通过 |
| 自实现 `digamma` 精度 | `2.0e-8` | `1e-6` | 通过 |
| MoE `E*sum f P >= sum f_i^2` | 最小松弛 `+0.398` | `>= -1e-9` | 通过 |
| MoE `E*sum f P >= 1`（旧声称） | 最小 `0.976` | -- | **不成立，已从规格中移除** |
- 作者：晨星（CJX0712）
- 算法设计：苏推衍
