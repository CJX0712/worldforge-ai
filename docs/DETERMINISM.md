# 确定性约定细则

- 对应架构: `docs/ARCHITECTURE.md` 第 10 节 (确定性策略)
- 相关决策: `docs/decisions/ADR-0002.md` (确定性优先), `docs/decisions/ADR-0003.md` (两环境两列数字)
- 相关内核约定: `docs/ALGORITHM_KERNEL.md` 第 0.1 节 (RNG 与 numpy 可选路径)、第 0.2 节 (容差常量)、第 0.3 节 (容差分级)
- 适用范围: `src/worldforge/core/seeding.py`, `core/sortutil.py`, `core/accel.py`, `eval/`, `observability/`

---

## 0. 结论先行

| 项 | 内容 |
| --- | --- |
| 可复现随机源 | `core.sampling.PCG32` (纯整数状态转移, 跨平台严格一致); **禁止**裸 `random.Random` / `random.seed` 作为可复现路径 |
| 采样 | 评估与默认路径 `temperature = 0`, 不做采样 |
| 排序 | 一律 `stable_sort((-score, id))`, 禁止只按分数排序 |
| 浮点 | 归约优先 `math.fsum` (精确求和); 对外比较走 `deterministic_view(ndims=6)` |
| 墙钟 | 业务对象不含时间字段; 只从 `Observer` 的 span 出; 走独立 `perf` 通道 |
| 双跑校验 | `assert_deterministic(fn, runs=2)` |
| 全新管道 | 评估每个 case 新建 `build_pipeline(env)` |
| 两环境 | `zero-dep` 与 `prod-extras` 各一套基线, 互不比较 |

---

## 1. 固定种子策略

| 层 | 做法 | 说明 |
| --- | --- | --- |
| 全局种子 | `WORLDFORGE_SEED` 默认 `1337`, 由 `seed_everything(seed)` 统一注入 | 只做全局卫生, 不作为算法可复现的依据 |
| 算法随机源 | `core.sampling.PCG32(seed)` 显式实例 | 内核约定的可复现路径; 纯整数状态转移, 不依赖平台浮点或 libc |
| 内核测试固定种子 | `PCG32(seed=20250101)` | 见 `ALGORITHM_KERNEL.md` 第 11.3 节 |
| 采样温度 | `temperature = 0.0` | 评估与默认路径强制; 生产路径可放开但不进确定性断言 |
| `PYTHONHASHSEED` | `seed_everything` 一并设置 | 消除 `set` / `dict` 迭代序差异的源头之一 |

为什么不用 `random.seed`: `random.Random` 的实现依赖 `getrandbits` 与平台相关的 Mersenne Twister, 状态迁移在跨版本 CPython 间不保证一致; `PCG32` 是纯整数运算, 状态转移自实现, 任何平台结果相同。

```python
# 正确
from worldforge.core.sampling import PCG32
rng = PCG32(seed=20250101)
idx = rng.next_int(n)

# 错误 (不可复现路径)
import random
random.seed(1337)
idx = random.randrange(n)
```

---

## 2. 非确定性来源清单

| # | 来源 | 机理 | 典型表现 | 修法 | 守护测试 |
| --- | --- | --- | --- | --- | --- |
| 1 | 墙钟延迟 | `time.time()` / `perf_counter()` 写入结果 | 双跑比对必失败 | 业务对象禁含时间字段; 只从 `Observer` span 出 | `test_no_wallclock_in_types` |
| 2 | 字典迭代序 | `PYTHONHASHSEED` 未固定时 `str` 哈希随机化 | 相同 dict 两次遍历顺序不同 | 输出前按键排序; `deterministic_view` 强制排序 | `test_dict_output_key_sorted` |
| 3 | 集合迭代序 | `set` 迭代序取决于哈希与插入历史 | `list(set(...))` 结果不稳定 | 转 `sorted(list(...))`; `deterministic_view` 对 set 排序 | `test_set_iteration_sorted` |
| 4 | 浮点累加顺序 | 浮点加法不满足结合律; numpy 用 SIMD/分块归约与朴素循环顺序不同 | 末位 (1e-16 级) 差异, 经阈值比较被放大成结果跳变 | 归约用 `math.fsum`; 融合前 `round(x, 6)`; 见第 5 节 | `test_fusion_rounds_scores_before_rank` |
| 5 | 不稳定排序 | 只按 `score` 排序, 同分顺序随输入顺序漂移 | 同分候选顺序两次不同 | `stable_sort((-score, id))` | `test_fusion_tie_break_stable` (`FM-14`) |
| 6 | 状态残留 | 复用管道导致索引/缓存/计数跨次累积 | 评估第二次跑分数变低 | 每个 case 新建管道 | `test_eval_determinism_two_runs_equal` (`FM-18`) |
| 7 | 读 `last_*` 缓存 | 适配器读对象上上一次的缓存字段 | 指标系统性偏低 | 显式捕获本次返回值; 类型上删除 `last_*` | `test_eval_uses_current_result` (`FM-08`) |
| 8 | 并行 / 乱序 | 多线程完成顺序不定 | 结果顺序随机 | 评估不打乱、不并行 | `test_eval_runner_is_sequential` |
| 9 | 外部服务抖动 | 远端 LLM / 向量库返回不稳定 | 分数随机漂移 | 评估只用本地默认栈或同一固定远端; 生产分数单独记录 | 环境 B 门禁标为可选 job |
| 10 | 常量输入的统计量 | 标准差为 0 时 Pearson/Spearman 未定义 | 被兜底成 `1.0`, 掩盖退化 | 常量输入直接返回 `0.0` | `test_rank_corr_constant_returns_zero` (`FM-09`) |

---

## 3. `deterministic_view()` 排除哪些键

```python
# src/worldforge/core/seeding.py
_EXCLUDE_KEYS = frozenset({
    "ts", "timestamp", "started_at", "ended_at", "created", "created_at",
    "latency_ms", "duration_ms", "wall_ms", "elapsed_ms",
    "pid", "hostname", "thread_id", "uptime_s",
})

def deterministic_view(obj: Any, ndims: int = 6) -> Any:
    """把任意对象转成可比较的纯数据结构:
    1) dict: 丢弃 _EXCLUDE_KEYS 中的键, 剩余按键排序
    2) float: round(x, ndims); -0.0 归一为 0.0
    3) set/frozenset: 排序后转 list
    4) dataclass/NamedTuple: 转 dict 后递归
    5) list/tuple: 保序递归
    6) 其他: repr(obj)
    """
```

| 类别 | 键 | 为什么排除 |
| --- | --- | --- |
| 时刻 | `ts`, `timestamp`, `started_at`, `ended_at`, `created`, `created_at` | 每次运行必然不同 |
| 时长 | `latency_ms`, `duration_ms`, `wall_ms`, `elapsed_ms` | 依赖机器负载, 与正确性无关 |
| 运行环境 | `pid`, `hostname`, `thread_id`, `uptime_s` | 环境相关, 不可复现 |

注意 `created` 在 HTTP 响应里是允许存在的 (OpenAI 兼容需要), 但**评估返回体与内部对象**不得出现任何上表字段。

| 变换 | 规则 | 目的 |
| --- | --- | --- |
| float | `round(x, 6)` | 吸收末位浮点差异 |
| `-0.0` | 归一为 `0.0` | 避免 `-0.0 != 0.0` 的假不等 |
| dict | 按键排序 | 消除 hash 序 |
| set | 排序转 list | 消除集合迭代序 |
| 其余对象 | `repr()` | 兜底可比较 |

用法:

```python
from worldforge.core.seeding import deterministic_view, assert_deterministic

# 双跑校验: 同一函数跑两次, 视图必须完全相等
assert_deterministic(lambda: pipe.query("检索层怎么分层?"), runs=2)

# 快照比对
assert deterministic_view(result_a) == deterministic_view(result_b)
```

---

## 4. 延迟走独立 perf 通道

| 环节 | 规则 |
| --- | --- |
| 产生点 | 唯一允许产生墙钟的地方是 `Observer.span()` 的上下文管理器退出时 |
| 业务对象 | `Hit` / `Completion` / `ToolResult` / `Plan` / `Evidence` / `Document` / `Chunk` 一律不含时间字段 |
| 评估侧 | `run_suite` 收集的延迟进返回体的 `perf` 键, 与 `metrics` / `agent_metrics` 平级, 不参与 `gates` |
| HTTP 侧 | 延迟进 `/v1/metrics` 的 `histograms`; 单条 trace 的墙钟只在 `/v1/traces/{id}?wall=1` 返回 |
| 断言 | `deterministic_view` 与 `assert_deterministic` 自动剔除墙钟键, 调用方无需手工清理 |
| 门禁 | `latency_p50_ms` / `latency_p95_ms` 阈值为 `-1` (不校验), 只做相对告警 |

```python
# 评估返回体形状 (延迟与指标物理隔离)
{
  "metrics":       {...},   # 检索 / RAG 聚合, 进 gates
  "agent_metrics": {...},   # 工具路由, 单独计数, 进 gates
  "gates":         [...],   # 不含延迟
  "perf":          {"latency_p50_ms": 41.2, "latency_p95_ms": 118.7}   # 不进 gates
}
```

---

## 5. numpy 路径与纯 Python 路径的末位浮点差异

### 5.1 差异来源

| 来源 | 说明 |
| --- | --- |
| 累加顺序 | 朴素循环是顺序累加; numpy 走 SIMD 与分块归约, 先算部分和再合并 |
| 融合乘加 | 编译期可能使用 FMA, 中间结果不回舍到 float64 |
| 库实现差异 | `cos` / `exp` / `log` 的实现不同, 末位 (1 ULP) 不同 |
| 精度设置 | float32 中间量会把误差放大到 `1e-7` 量级 |

内核文档的实测基线 (`ALGORITHM_KERNEL.md` 第 12 节): tiled attention 与朴素实现的差异为 `1.1e-16 ~ 2.2e-16`; 故容差分级取

| 场景 | 容差 |
| --- | --- |
| 恒等式级 | `1e-12` (`ATOL_STRICT`) |
| 分解 / 重构 | `1e-10` (`ATOL_TIGHT`) |
| 迭代收敛 | `1e-8` (`ATOL_LOOSE`) |
| 分块 vs 朴素 (float64) | `1e-9` |
| 分块 vs 朴素 (float32) | `2e-5` |

对策: 归约优先 `math.fsum` (精确浮点求和), 这是纯 Python 路径相对 numpy 的关键稳健性来源; 所有 `KL` / `ELBO` / `logsumexp` 的分子分母都要用。

### 5.2 为什么 RRF 会把末位差异放大

RRF 的输入是**排名**, 输出是**分数**:

```
rrf_score(d) = sum_over_branches( 1 / (k + rank_branch(d)) )     # k = 60
```

排名由分数排序决定, 所以「末位浮点差异 -> 排名翻转 -> 分数跳变」是一条离散放大链。

具体数值 (k = 60, 单分支):

| 情形 | 两条候选的分差 | 排名 | RRF 贡献 | 结果差 |
| --- | --- | --- | --- | --- |
| 纯 Python 路径 | `0.50000000010` vs `0.50000000009` (差 `1e-11`) | 1, 2 | `1/61 = 0.01639344`, `1/62 = 0.01612903` | `2.64e-4` |
| numpy 路径 (末位翻转) | 同上但排序翻转 | 2, 1 | 互换 | `-2.64e-4` |

`1e-11` 的输入噪声被放大成 `2.6e-4` 的输出跳变, **约 2.6e7 倍**。双分支融合时两条链叠加, 跳变更大。

关键点: 稳定排序 `(-score, id)` **救不了**这个问题 -- 它只解决「分数完全相等时顺序漂移」, 而这里的噪声改变的是分数本身的相对大小, 排序结果随之改变。

### 5.3 缓解措施 (按优先级)

| # | 措施 | 说明 |
| --- | --- | --- |
| 1 | 融合前把分数 `round(x, 6)` | 与 `deterministic_view` 同口径; 末位噪声被吸收, 极接近的分数坍缩为相等, 再由 `chunk_id` 决定顺序 -> 完全确定 |
| 2 | 归约用 `math.fsum` | 减少累加误差的产生量 |
| 3 | 稳定排序 `(-score, chunk_id)` | 处理坍缩后的真并列 |
| 4 | numpy 加速只做加速不换逻辑 | `core/accel.py` 与纯 Python 路径必须过 `atol=1e-9` 对齐测试 (`FM-17`) |
| 5 | 断言走 `deterministic_view` | 比较前统一 `round(x, 6)` |

```python
# retrieval/fusion.py 强制形状
def fuse(dense_hits, sparse_hits, k=60):
    scores: dict[str, float] = {}
    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):          # rank 由已 round 的分数排序得到
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    merged = [Hit(chunk_id=cid, score=round(s, 6), source="fused", meta=...)
              for cid, s in scores.items()]
    return stable_sort(merged, key=lambda h: h.score, tie_key=lambda h: h.chunk_id)
```

守护测试: `test_fusion_rounds_scores_before_rank` -- 构造两条分数只差 `1e-11` 的候选, 断言融合结果在 numpy 开/关两种情况下完全一致。

---

## 6. 「两环境两列数字」对照表怎么写

### 6.1 表头模板

| 指标 | 环境 A floor | 环境 A baseline | 环境 A threshold | 环境 B floor | 环境 B baseline | 环境 B threshold | 类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |

`threshold = max(floor, baseline * 0.95)`, baseline 未回填时 threshold 取 floor。

### 6.2 填写规则

| # | 规则 | 说明 |
| --- | --- | --- |
| 1 | 两个环境分别回填 | `record_baseline.py --env zero-dep` 与 `--env prod-extras` 各跑一次, 不可互相拷贝 |
| 2 | 禁止横向比较 | 环境 A 的 `0.84` 与环境 B 的 `0.91` 不可比; 判定只在同一环境列内进行 |
| 3 | floor 人工写死且会失败 | 环境 A 用零依赖栈能达到的量级, 环境 B 用生产栈能达到的量级, 分别设 |
| 4 | 保留三位小数 | 基线保留 3 位, threshold 由公式计算后保留 4 位 |
| 5 | 差异超过 20% 要说明 | 同一指标在两环境差值过大时, 在 PR 里写明是能力差还是口径 bug |
| 6 | 单指标可覆盖系数 | 某指标抖动超过 5% 时用 `--override ndcg@10=0.90` 单独放宽, 不放宽全局 |

### 6.3 已回填示例 (数值为示意)

| 指标 | A floor | A baseline | A threshold | B floor | B baseline | B threshold | 类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `doc_hit_rate@5` | 0.60 | 0.840 | 0.7980 | 0.60 | 0.910 | 0.8645 | 硬 |
| `recall@10` | 0.55 | 0.712 | 0.6764 | 0.65 | 0.803 | 0.7629 | 硬 |
| `mrr` | 0.50 | 0.641 | 0.6089 | 0.50 | 0.722 | 0.6859 | 硬 |
| `grounding` | 0.55 | 0.873 | 0.8294 | 0.65 | 0.902 | 0.8569 | 硬 |
| `faithfulness` | 0.50 | 0.796 | 0.7562 | 0.60 | 0.845 | 0.8028 | 硬 |
| `tool_route_accuracy` | 0.70 | 0.900 | 0.8550 | 0.85 | 0.940 | 0.8930 | 硬 |
| `tool_call_success` | 0.90 | 1.000 | 0.9500 | 0.90 | 1.000 | 0.9500 | 硬 |
| `ndcg@10` | 无 | 0.688 | 告警 | 无 | 0.751 | 告警 | 软 |
| `rank_corr` | 无 | 0.452 | 告警 | 无 | 0.513 | 告警 | 软 |
| `latency_p95_ms` | 无 | 118.7 | 不校验 | 无 | 640.2 | 不校验 | 软 |

读法: 环境 A 的 `doc_hit_rate@5` 掉到 `0.70` 必挂 (低于 0.7980); 掉到 `0.82` 不挂但会触发软告警 (相对下降超过 2%)。

---

## 7. 确定性分级

| 级别 | 要求 | 适用对象 |
| --- | --- | --- |
| 硬确定性 | 逐位相等 (`deterministic_view` 完全相等) | 检索结果、分块结果、嵌入向量、评估结果、融合排序 |
| 软确定性 | 容差内相等 (按 `ALGORITHM_KERNEL.md` 第 0.3 节分级) | 数值内核 (分解、迭代、采样的统计性质) |
| 非确定性 | 不做相等性断言, 只做区间/方向性检查 | 延迟、吞吐、远端服务返回、流式输出 |

---

## 8. 守护测试清单

| 测试 | 守护内容 |
| --- | --- |
| `test_core_seeding.py` | `deterministic_view` 的排除键、float 收敛、set 排序 |
| `test_no_wallclock_in_types.py` | 类型定义不含墙钟字段 (AST 扫描) |
| `test_fusion_tie_break_stable.py` | 同分并列顺序稳定 (`FM-14`) |
| `test_fusion_rounds_scores_before_rank.py` | 融合前 round, 消除 RRF 放大 (`第 5 节`) |
| `test_core_accel_parity.py` | numpy 与纯 Python 路径 `atol=1e-9` 对齐 (`FM-17`) |
| `test_core_rank_corr.py` | 常量输入返回 0.0 (`FM-09`) |
| `test_eval_determinism_two_runs_equal.py` | 评估连跑两次视图相等 (`FM-18`) |
| `test_eval_uses_current_result.py` | 不读 `last_*` 缓存 (`FM-08`) |
| `test_eval_runner_is_sequential.py` | 用例顺序固定、无并行 |
| `test_sampling_pcg32_reproducible.py` | `PCG32` 跨平台一致, 不涉及裸 `random` |
