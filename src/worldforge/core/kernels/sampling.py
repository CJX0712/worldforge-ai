"""解码策略与可复现随机源（模块 4）。

做什么
    自实现 PCG32（跨平台逐位可复现）作为唯一随机源；temperature / top-k / top-p
    滤波；categorical 采样；beam search。

不变量
    I1  每次过滤后概率和仍为 1（1e-12）
    I2  top-k 基数：count(probs > 0) == k（无并列时）
    I3  top-p 极小性：保留前缀的累积概率 >= p - 1e-9，去掉最后一个则 < p
    I4  temperature 单调性：T1 < T2 推出 H(softmax(z/T1)) <= H(softmax(z/T2))
    I5  T -> 0+ 收敛到 argmax 的 one-hot
    I6  T = 1 恒等
    I7  beam 分数降序
    I8  B = 1 时退化为逐步 argmax
    I9  RNG 可复现：同 seed 两次实例化输出逐位相等
    I10 RNG 均匀性：均值 0.5、方差 1/12（5e-3）
    I12 采样索引恒在 [0, V)
    I13 高斯矩：均值 0、标准差 1（1e-2）

复杂度
    top_k / top_p  O(V log V)
    sample_categorical  O(V)
    beam_search  O(L * B * V + L * B^2 log(B^2))
    PCG32.next_u32  O(1)

常见坑
    - 排序 key 必须显式带索引 (-v, i)，否则并列元素的顺序不确定，破坏可复现性。
    - top-p 概率全 0 时循环永远到不了 p，必须强制至少保留一个。
    - T = 0 必须显式抛错，贪心走 argmax 而不是 T=0。
"""

import math

from .numeric import softmax, safe_log
from .prob import normalize

__all__ = [
    "TemperatureZeroError",
    "InvalidNucleusProbError",
    "PCG32",
    "apply_temperature",
    "top_k_filter",
    "top_p_filter",
    "top_k_indices",
    "sample_categorical",
    "beam_search",
]


class TemperatureZeroError(ValueError):
    """temperature <= 0。贪心请用 argmax，不要走 T = 0。"""


class InvalidNucleusProbError(ValueError):
    """top-p 的 p 不在 (0, 1] 内。"""


class PCG32:
    """PCG-XSH-RR 32 位输出（O'Neill 2014），纯整数状态转移。

    不变量
        I9  同 (seed, stream) 两次实例化，next_u32 序列逐位相等
        I10 next_float 落在 [0, 1)，均值 0.5、方差 1/12
        I13 gauss 一阶矩 0、二阶矩 1

    之所以不用 random.Random：其流在不同 Python 版本间不保证一致，
    而本模块要求跨机器、跨版本可复现。
    """

    MASK64 = (1 << 64) - 1
    MASK32 = (1 << 32) - 1
    MULT = 6364136223846793005

    def __init__(self, seed: int, stream: int = 0):
        inc = (stream << 1 | 1) & self.MASK64
        self.state = (seed * self.MULT + inc) & self.MASK64
        self.inc = inc
        self._spare = None

    def next_u32(self) -> int:
        """推进状态并返回 32 位无符号输出。

        不变量：输出恒在 [0, 2^32)；状态转移为纯整数运算，跨平台严格一致。
        """
        self.state = (self.state * self.MULT + self.inc) & self.MASK64
        xorshifted = (((self.state >> 18) ^ self.state) >> 27) & self.MASK32
        rot = (self.state >> 59) & self.MASK32
        r = rot & 31
        return ((xorshifted >> r) | (xorshifted << (32 - r))) & self.MASK32

    def next_float(self) -> float:
        """返回 [0, 1) 内的均匀浮点数。

        不变量：0.0 <= x < 1.0。
        """
        return self.next_u32() / 4294967296.0

    def gauss(self) -> float:
        """Box-Muller 标准正态，成对缓存第二个值。

        不变量：I13 均值 0、标准差 1；sigma == 0 的调用方不应调用本方法，
        以保证随机流干净（DDIM eta=0 的确定性依赖这一点）。
        """
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = self.next_float()
        if u1 <= 0.0:
            u1 = 1e-12
        u2 = self.next_float()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        self._spare = r * math.sin(theta)
        return r * math.cos(theta)

    def uniform(self, lo: float, hi: float) -> float:
        """[lo, hi) 上的均匀标量。

        不变量：lo <= x < hi。
        """
        return lo + (hi - lo) * self.next_float()


def apply_temperature(logits, T: float):
    """z / T。

    不变量：I6 T == 1.0 时逐位不变；T <= 0 抛 TemperatureZeroError。
    """
    if T <= 0.0:
        raise TemperatureZeroError("temperature must be > 0, got %r" % (T,))
    return [z / T for z in logits]


def top_k_indices(values, k: int):
    """返回最大的 k 个值的索引，按值降序。

    不变量：并列时索引小者优先（确定性），故两次调用结果逐位相同。
    """
    if k <= 0:
        return []
    k = min(k, len(values))
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[:k]


def top_k_filter(logits, k: int):
    """保留最大的 k 个 logit，其余置为 -inf。

    不变量
        I2 无并列时正好 k 个位置非 -inf
        过滤后再 softmax 的概率和仍为 1
    """
    if k <= 0 or k >= len(logits):
        return logits[:]
    thr = sorted(logits, reverse=True)[k - 1]
    return [z if z >= thr else float("-inf") for z in logits]


def top_p_filter(logits, p: float):
    """nucleus 滤波：保留累积概率首次达到 p 的最小前缀，其余置 -inf。

    不变量
        I3 保留集累积概率 >= p - 1e-9，且去掉最后一个后 < p
        至少保留一个元素（概率全 0 时也不会返回空集）
    """
    if not (0.0 < p <= 1.0):
        raise InvalidNucleusProbError(p)
    n = len(logits)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (-logits[i], i))
    probs = softmax(logits)
    cum = 0.0
    keep = []
    for idx in order:
        keep.append(idx)
        cum += probs[idx]
        if cum >= p - 1e-9:
            break
    if not keep:
        keep = [order[0]]
    mask = [False] * n
    for idx in keep:
        mask[idx] = True
    return [logits[i] if mask[i] else float("-inf") for i in range(n)]


def sample_categorical(probs, rng, T: float = 1.0, k: int = 0, p: float = 1.0) -> int:
    """按（可选 temperature / top-k / top-p 变换后的）概率采样一个索引。

    不变量
        I1 变换后概率和为 1（1e-12）
        I12 返回值恒满足 0 <= idx < V
    末尾的 return V-1 是浮点累加误差的兜底。
    """
    probs = normalize(probs)
    if T != 1.0:
        probs = softmax(apply_temperature([safe_log(x) for x in probs], T))
    if k:
        probs = softmax(top_k_filter([safe_log(x) for x in probs], k))
    if p < 1.0:
        probs = softmax(top_p_filter([safe_log(x) for x in probs], p))
    u = rng.next_float()
    acc = 0.0
    for i, pi in enumerate(probs):
        acc += pi
        if u < acc:
            return i
    return len(probs) - 1


def beam_search(step_logprobs_fn, B: int, max_len: int, eos_id: int):
    """定长 beam search。

    参数
        step_logprobs_fn: prefix -> (V,) 已归一化的 log 概率
        B: beam 宽度
        max_len: 最大展开步数
        eos_id: 结束符 id
    返回
        [(score, ids), ...]，长度 <= B，按 score 降序

    不变量
        I7 scores[0] >= scores[1] >= ... >= scores[B-1]
        I8 B = 1 时输出序列 == 逐步 argmax 的序列（同 tie-break）
    排序 key 显式带 ids，保证并列时的确定性。
    """
    beams = [(0.0, [])]
    for _ in range(max_len):
        cand = []
        for score, ids in beams:
            lp = step_logprobs_fn(ids)
            for v in top_k_indices(lp, B):
                cand.append((score + lp[v], ids + [v]))
        if not cand:
            break
        best = {}
        for s, i in cand:
            key = tuple(i)
            if key not in best or s > best[key]:
                best[key] = s
        beams = sorted(((s, list(k)) for k, s in best.items()),
                       key=lambda c: (-c[0], c[1]))[:B]
        if all(ids and ids[-1] == eos_id for _, ids in beams):
            break
    return beams
