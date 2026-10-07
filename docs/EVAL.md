# 评估与基线操作手册

- 对应架构: `docs/ARCHITECTURE.md` 第 9 节 (评估方案)
- 相关决策: `docs/decisions/ADR-0002.md` (确定性优先), `docs/decisions/ADR-0003.md` (阈值门禁策略)
- 适用范围: `src/worldforge/eval/`, `scripts/`

---

## 0. 结论先行

| 项 | 内容 |
| --- | --- |
| 门限公式 | `threshold = max(floor, baseline * 0.95)` |
| floor | 人工设定的硬下限, **必须是会失败的值**, 基线未回填时立即生效 |
| baseline | `scripts/record_baseline.py` 实测均值, 回填后贴着基线设 |
| 两环境 | 环境 A `zero-dep` (零依赖默认栈) 与环境 B `prod-extras` (生产 extras 栈) 各一套数字, 互不比较 |
| 强制约束 1 | **每个 case 都在全新管道上跑**, 禁止复用运行期单例管道 |
| 强制约束 2 | **工具路由类用例不进检索聚合**, 在 `agent_metrics` 中单独计数上报 |
| 门禁载体 | `scripts/verify.py` (零依赖) + `pytest` (全量) + `POST /v1/evaluate` (在线) |

**口径优先级**: 本文档与 `ADR-0003` 优先于 `ARCHITECTURE.md` 第 9.2 节。第 9.2 节写的 `threshold = max(floor, mu - 3*sigma)` 已被 `ADR-0003` 主理人裁决替换为 `baseline * 0.95`; 原因是 5 轮样本的 sigma 估计不可靠, 且 sigma 为 0 时门限退化为零容忍。

---

## 1. 指标定义与口径

### 1.1 检索类指标 (进 `metrics` 聚合)

| 指标 | 公式 | 口径说明 |
| --- | --- | --- |
| `doc_hit_rate@k` | `mean( 1 if \|gold & retrieved@k\| > 0 else 0 )` | 按**查询**平均 (macro), 不是按 chunk; 只要命中一个黄金块就算 1 |
| `recall@k` | `mean( \|gold & retrieved@k\| / \|gold\| )` | 按查询平均; `gold` 为空的用例**跳过**该项, 不计入分母 |
| `mrr` | `mean( 1 / rank_first_relevant )` | 未命中记 `0`; `rank` 从 1 开始 |
| `ndcg@k` | `mean( DCG@k / IDCG@k )` | 二元增益 (`rel = 1`); `IDCG@k = sum(1/log2(i+1)) for i in 1..min(\|gold\|, k)`; 软指标, 不设 floor |

边界约定:

| 情形 | 处理 |
| --- | --- |
| `k` 大于返回结果数 | 按实际返回条数截断, 不补零 |
| 一个用例有多个黄金块 | `recall` 全部计入; `mrr` 只看排名最前的那个 |
| `gold_chunk_ids` 为空 | 该用例不参与 `recall` 与 `mrr`, 但保留在 `doc_hit_rate` 分母之外 (同样跳过) |
| top-k 出现同分并列 | 排序主键 `(-score, chunk_id)` 已保证唯一顺序, 不存在并列 |

### 1.2 RAG 质量指标 (进 `metrics` 聚合)

| 指标 | 公式 | 口径说明 |
| --- | --- | --- |
| `grounding` | `被证据支持的答案句子数 / 答案句子总数` | 句子切分按中英标点; **计算前先剥离** `[id#n]`、`[n]`、`(依据: ...)`、`(来源: ...)`、`【n】` 等自家引用标记, 否则自家标记会被算成未接地 token 而系统性拉低 |
| `faithfulness` | `sum(IDF(t) for t in 答案token if t in 证据) / sum(IDF(t) for t in 答案token)` | IDF 用 Robertson 形式 `ln(1 + (N-n+0.5)/(n+0.5))`, 恒非负; 停用词不计入; 同样先剥离引用标记 |

### 1.3 Agent / 工具类指标 (**不进** `metrics` 聚合)

| 指标 | 公式 | 口径说明 |
| --- | --- | --- |
| `tool_route_accuracy` | `工具名正确的用例数 / agent 用例总数` | 只看工具名, 不看参数; 单独上报在 `agent_metrics` |
| `tool_call_success` | `ok=True 的工具调用数 / 工具调用总数` | 归一化后的失败 (`ToolResult(ok=False)`) 计入分母 |
| `tool_args_accuracy` | 参数精确匹配的比例 | 可选, 软指标 |

### 1.4 重排与延迟指标

| 指标 | 公式 | 口径说明 |
| --- | --- | --- |
| `rank_corr` | 重排前后序与黄金序的 Spearman 相关 | **常量输入返回 0.0** (标准差为 0 时相关未定义, 不得兜底成 1.0); 软指标 |
| `latency_p50_ms` / `latency_p95_ms` | 端到端延迟分位 | 只进 `perf` 通道与 `/v1/metrics`, **不进门禁** |

---

## 2. 门限策略

### 2.1 双层门限

| 层 | 来源 | 何时生效 | 作用 |
| --- | --- | --- | --- |
| `floor` | 人工设定, 入库写死 | 立即 | 绝对质量把关, 防止基线本身偏低时门限退化 |
| `baseline` | 实测均值, 回填 | 回填后 | 贴着实测, 让真回归必挂、抖动不挂 |

```
threshold = max(floor, baseline * 0.95)
```

未回填时 `baseline` 为 `null`, 此时 `threshold = floor`, **不存在门禁空窗期**。

### 2.2 门限表 (两环境两列数字)

| 指标 | 环境 A floor | 环境 A baseline | 环境 B floor | 环境 B baseline | 类型 |
| --- | --- | --- | --- | --- | --- |
| `doc_hit_rate@5` | 0.60 | null | 0.60 | null | 硬 |
| `recall@10` | 0.55 | null | 0.65 | null | 硬 |
| `mrr` | 0.50 | null | 0.50 | null | 硬 |
| `grounding` | 0.55 | null | 0.65 | null | 硬 |
| `faithfulness` | 0.50 | null | 0.60 | null | 硬 |
| `tool_route_accuracy` | 0.70 | null | 0.85 | null | 硬 |
| `tool_call_success` | 0.90 | null | 0.90 | null | 硬 |
| `ndcg@10` | 无 | null | 无 | null | 软 |
| `rank_corr` | 无 | null | 无 | null | 软 |
| `latency_p50_ms` | 无 | null | 无 | null | 软, 不校验 |
| `latency_p95_ms` | 无 | null | 无 | null | 软, 不校验 |

| 环境 | 代号 | 构成 |
| --- | --- | --- |
| 环境 A | `zero-dep` | HashingEmbedder + InMemoryExactIndex + BM25Index + LexicalReranker + ExtractiveMockLLM + RuleBasedPlanner |
| 环境 B | `prod-extras` | fastembed + FAISS(CPU) + 真 LLM (llama-cpp / Ollama) |

软指标不设 floor 是有意设计: `rank_corr` 在常量输入下数学上恒为 0, 延迟本质是抖动量, 给它们设硬门限只会制造误报。

### 2.3 判定与告警

| 判定 | 规则 | 结果 |
| --- | --- | --- |
| 硬指标通过 | `value >= threshold` | `gates[].ok = true` |
| 硬指标未过 | `value < threshold` | `gates[].ok = false`, `passed = false`, CI 抛 `EvalGateError` |
| 软指标告警 | 相对基线下降 `> 2%` | `warnings` 追加一条, 不影响 `passed` |
| 延迟 | 不参与 | 只进 `perf` |

HTTP 层: `POST /v1/evaluate` 门限未过时**仍返回 200** 且 `passed=false`。进程内 CLI 才抛 `EvalGateError`, 退出码非 0。

---

## 3. 命令用法

### 3.1 `python scripts/verify.py`

零依赖一级门禁: 不装任何 extras, 裸 CPython 3.11+ 必须退出码 0。

| 用法 | 命令 | 说明 |
| --- | --- | --- |
| 全量门禁 | `python scripts/verify.py` | 跑核心不变量 + 最小评估冒烟 |
| 只看评估 | `python scripts/verify.py --suite smoke` | 只跑评估冒烟套件 |
| 严格模式 | `python scripts/verify.py --strict` | 后端缺失即失败, 不静默回退 |
| 机器可读 | `python scripts/verify.py --json` | 输出 `verify-report.json` |
| 指定环境 | `python scripts/verify.py --env zero-dep` | 默认即 `zero-dep` |

| 退出码 | 含义 |
| --- | --- |
| 0 | 全部通过 |
| 1 | 有硬门限未过或核心不变量失败 |
| 2 | 参数/配置错误 |

### 3.2 `python scripts/record_baseline.py`

生成实测基线, 写入 `src/worldforge/eval/baseline.json`。

| 用法 | 命令 | 说明 |
| --- | --- | --- |
| 环境 A | `python scripts/record_baseline.py --env zero-dep --runs 5` | 默认 5 轮取均值 |
| 环境 B | `python scripts/record_baseline.py --env prod-extras --runs 5` | 需先装 extras |
| 单套件 | `python scripts/record_baseline.py --suite retrieval --runs 5` | 只刷检索指标 |
| 预演不落盘 | `python scripts/record_baseline.py --runs 5 --dry-run` | 打印将写入的值 |
| 写入 | `python scripts/record_baseline.py --runs 5 --write` | 不加 `--write` 只打印 |
| 单指标覆盖系数 | `python scripts/record_baseline.py --override ndcg@10=0.90` | 某指标抖动超过 5% 时单独放宽 |

输出示例:

```
env=zero-dep  runs=5  suite=default  cases=120
metric                mu      sigma    floor   threshold
doc_hit_rate@5      0.8400   0.0071    0.60      0.7980
recall@10           0.7120   0.0094    0.55      0.6764
mrr                 0.6410   0.0112    0.50      0.6089
grounding           0.8730   0.0052    0.55      0.8294
faithfulness        0.7960   0.0081    0.50      0.7562
tool_route_accuracy 0.9000   0.0000    0.70      0.8550
tool_call_success   1.0000   0.0000    0.90      0.9500
```

### 3.3 其余入口

| 入口 | 命令 | 用途 |
| --- | --- | --- |
| 全量测试 | `python -m pytest` | 单测 + 评估门限测试 |
| 评估测试子集 | `python -m pytest tests/test_eval_runner.py tests/test_eval_gates.py -v` | 只跑评估相关 |
| 在线评估 | `curl -X POST /v1/evaluate -d '{"suite":"default"}'` | 见 `docs/API.md` 第 9 节 |
| 延迟基准 | `python scripts/bench.py` | 输出 `perf.json`, 不进门限 |

---

## 4. 黄金集格式

位置: `src/worldforge/eval/datasets/`, 每行一个 JSON 对象 (JSONL), **行序即执行顺序, 不打乱、不并行**。

### 4.1 `golden_retrieval.jsonl`

```json
{"case_id":"r001","kind":"retrieval","documents":[{"doc_id":"d1","text":"...","meta":{"lang":"zh"}}],"query":"检索层怎么分层?","gold_chunk_ids":["d1#0003"],"gold_doc_ids":["d1"],"k":10,"note":"分层描述"}
```

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `case_id` | string | 是 | 全局唯一, 排序与引用用 |
| `kind` | string | 是 | 固定 `retrieval` |
| `documents` | array | 是 | 本用例的语料, 每个 case 独立摄入 |
| `documents[].doc_id` | string | 是 | 文档标识 |
| `documents[].text` | string | 是 | 正文 |
| `documents[].meta` | object | 否 | 过滤用元数据 |
| `query` | string | 是 | 检索问句 |
| `gold_chunk_ids` | array | 是 | 黄金分块 id 列表, 可为空数组 (该用例跳过 recall/mrr) |
| `gold_doc_ids` | array | 否 | 黄金文档 id, 用于文档级命中统计 |
| `k` | int | 否 | 覆盖默认 `k`, 不填则用套件默认 |
| `note` | string | 否 | 人工备注, 不参与计算 |

### 4.2 `golden_rag.jsonl`

```json
{"case_id":"g001","kind":"rag","documents":[{"doc_id":"d1","text":"...","meta":{}}],"query":"WorldForge 的检索层怎么分层?","gold_chunk_ids":["d1#0003"],"gold_spans":["检索层分为稀疏与稠密两路"],"reference_answer":"检索层分为稀疏与稠密两路, 经 RRF 融合后重排。","k":5}
```

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `case_id` | string | 是 | 全局唯一 |
| `kind` | string | 是 | 固定 `rag` |
| `documents` / `query` / `gold_chunk_ids` / `k` | 同 4.1 | 是/否 | 同上 |
| `gold_spans` | array | 是 | 期望出现在答案中的关键片段, 用于接地判定 |
| `reference_answer` | string | 否 | 参考答句, 人工对照用, 不参与自动打分 |

### 4.3 `golden_agent.jsonl` (工具路由, 单独计数)

```json
{"case_id":"a001","kind":"agent","documents":[],"query":"1+1 等于多少?","expected_tool":"calculator","expected_args":{"expression":"1+1"},"expect_success":true}
{"case_id":"a002","kind":"agent","documents":[],"query":"查一下检索层怎么分层","expected_tool":"retriever","expected_args":{},"expect_success":true}
```

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `case_id` | string | 是 | 全局唯一 |
| `kind` | string | 是 | 固定 `agent` |
| `documents` | array | 否 | 工具用例通常为空 |
| `query` | string | 是 | 触发语句 (含全角标点以覆盖归一化路径) |
| `expected_tool` | string | 是 | 期望工具名, 判定 `tool_route_accuracy` |
| `expected_args` | object | 否 | 期望参数, 判定 `tool_args_accuracy` (软) |
| `expect_success` | bool | 否 | 期望调用是否成功, 判定 `tool_call_success` |

`kind=agent` 的用例**只**贡献 `agent_metrics`, 不进入 `doc_hit_rate` / `recall` / `mrr` / `grounding` 的聚合。

---

## 5. 两条强制约束

### 5.1 每个 case 都在全新管道上跑

| 要求 | 说明 |
| --- | --- |
| 必做 | 套件内每个用例调用 `build_pipeline(env)` 新建管道 |
| 禁止 | 复用运行期单例管道, 或用 `clear()` 代替重建 |
| 原因 | 索引、嵌入缓存、工具调用计数、observer 缓冲会跨 case 残留, 第二次跑分数漂移 (`ARCHITECTURE.md` FM-18) |
| 守护 | `tests/test_eval_runner.py::test_eval_determinism_two_runs_equal`: 同套件连跑两次, `deterministic_view(scorecard)` 必须完全相等 |

```python
# eval/runner.py 骨架 (强制形状)
def run_suite(suite: str, env: Mapping[str, str]) -> dict[str, Any]:
    rows, agent_rows, perf = [], [], []
    for case in load_cases(suite):                 # 固定顺序, 不打乱, 不并行
        pipe = pipeline.build_pipeline(env)        # 强制: 每个 case 全新管道
        if case.documents:
            pipe.ingest(case.documents)
        result = pipe.answer(case.query) if case.kind != "agent" else pipe.run_agent(case.query)
        if case.kind == "agent":
            agent_rows.append(score_agent_case(case, result))   # 工具路由单独收集
        else:
            rows.append(score_case(case, result))               # 检索/RAG 收集
        perf.append(collect_latency(pipe))         # 延迟单独收集
    scorecard = aggregate(rows)
    agent_scorecard = aggregate(agent_rows)
    gates = apply_gates(scorecard, agent_scorecard, load_baseline(env))
    return {"metrics": scorecard,
            "agent_metrics": agent_scorecard,      # 单独上报, 不混入 metrics
            "gates": gates,
            "passed": all(g["ok"] for g in gates if g["hard"]),
            "perf": summarize_latency(perf)}       # 不参与 gates
```

### 5.2 工具路由类用例不进检索聚合

| 要求 | 说明 |
| --- | --- |
| 必做 | `tool_route_accuracy` / `tool_call_success` / `tool_args_accuracy` 只出现在 `agent_metrics` |
| 禁止 | 把工具用例的结果混进 `doc_hit_rate` / `recall` / `mrr` / `grounding` 的平均值 |
| 原因 | 量纲不同 (一个是路由选择正确率, 一个是召回覆盖率), 混合会稀释检索指标, 让检索退化被工具高分掩盖 |
| 守护 | `tests/test_eval_runner.py::test_agent_metrics_not_in_retrieval_aggregate`: 断言 `metrics` 的键集合不含 `tool_*`, 且 `agent_metrics.cases` 等于 agent 用例数 |

---

## 6. 新增用例

| # | 步骤 | 要点 |
| --- | --- | --- |
| 1 | 选文件 | 检索类 -> `golden_retrieval.jsonl`; 问答类 -> `golden_rag.jsonl`; 工具类 -> `golden_agent.jsonl` |
| 2 | 取 `case_id` | 前缀 + 三位序号 (`r024`), 必须与已有不冲突 |
| 3 | 写 `documents` | 每个用例自带语料; 不要依赖别的用例留下的索引 |
| 4 | 定 `gold_chunk_ids` | 分块 id 形如 `d1#0003`; 不确定时先跑一次 `/v1/query` 看实际 chunk_id |
| 5 | 追加到文件末尾 | **不插队**: 行序即执行顺序, 插队会改变后续 case 的执行序号 |
| 6 | 跑一遍 | `python scripts/verify.py --suite smoke` 先看是否报错 |
| 7 | 更新基线 | 新增用例会改变聚合值, 必须 `record_baseline.py --runs 5 --write` 重新回填 |
| 8 | 同 PR 提交 | 用例与基线变更放同一个 PR, 说明期望变化方向 |

纪律:

| 规则 | 说明 |
| --- | --- |
| 不修改既有用例 | 修改等于改度量衡; 确需修改时同步改基线并在 PR 说明 |
| 不删用例 | 删除会让历史基线不可比 |
| 基线只许上调 | 人工下调必须附 root cause 分析 |
| 黄金集入库 | 与代码同版本, 不接受运行时下载 |

---

## 7. 排障

| 症状 | 根因 | 修法 |
| --- | --- | --- |
| 第二次跑分数变低 | 复用了管道 | 检查 `run_suite` 是否每个 case 调 `build_pipeline` (`ARCHITECTURE.md` FM-18) |
| grounding 系统性偏低 | 未剥离自家引用标记 | 核验前调 `core.text.strip_citations()` (`FM-10`) |
| `tool_route_accuracy` 全对但 `recall` 掉 | 工具用例混进了检索聚合 | 按 5.2 分离上报 |
| 常量输入 `rank_corr` 是 1.0 | 标准差为 0 时未定义却被兜底 | 常量输入返回 0.0 (`FM-09`) |
| 基线回填后全部指标挂 | 用环境 B 的基线去判环境 A | 两套基线分环境存储, 确认 `--env` 一致 |
| `verify.py` 报 `EvalGateError` 但 `/v1/evaluate` 返回 200 | 设计如此 | HTTP 层不把门限失败当错误; 客户端读 `passed` |
| 新增用例后 CI 红 | 基线未同步回填 | 跑 `record_baseline.py --write` 并同 PR 提交 |
