# WorldForge AI

> 作者：晨星 | License: MIT | Python 3.11+ | 运行时零第三方依赖

一套端到端可运行的 AI 系统：**数学与算法内核 + 混合检索 + Agent 编排 + HTTP 网关 + 评估门禁**，每一层都可插拔，每一层都有默认的、真正能工作的零依赖实现。

不重复造轮子：外部能力全部通过 Protocol 注入，缺失时静默回退而不是崩溃；核心算法自己写，因为那些不变量正是本项目的价值所在。

---

## 30 秒上手

```bash
git clone https://github.com/CJX0712/worldforge-ai.git
cd worldforge-ai
python scripts/verify.py
```

`verify.py` 会依次跑六个阶段，任一失败立即短路并非零退出：

```
[PASS] p0_char_gate      全仓禁用字符扫描
[PASS] pytest            259 个用例
[PASS] imports           19 个模块裸解释器导入
[PASS] smoke             端到端冒烟（检索/问答/工具/可观测）
[PASS] eval_gate         评估门限判定
[PASS] http_e2e          20 条 HTTP 端到端断言
```

起服务：

```bash
python -m worldforge.serving.app          # http://127.0.0.1:8000
```

喂一篇文档，问一个问题：

```bash
curl -X POST localhost:8000/v1/ingest -H "Content-Type: application/json" \
  -d '{"doc_id":"kb1","text":"BM25 由 Robertson 提出，用逆文档频率衡量词项区分度。"}'

curl -X POST localhost:8000/v1/ask -H "Content-Type: application/json" \
  -d '{"query":"BM25 是谁提出的"}'
```

---

## 架构

```
             ingestion -> retrieval -> agents -> serving -> eval
                              |            |         |
                          adapters      serving   observability
                              |            |
                           ports (10 个 Protocol)
                              |
                    core (算法内核 + 数值工具)
```

| 层 | 位置 | 职责 |
|---|---|---|
| 数学算法内核 | `src/worldforge/core/kernels/` | numeric / linalg / prob / sampling / attention / moe / diffusion / rl / gnn / bayes 十个纯标准库模块 |
| 数值工具 | `src/worldforge/core/` | logsumexp 稳定化、Robertson-IDF BM25、RRF、分词、安全求值、确定性种子 |
| 端口契约 | `src/worldforge/ports/` | 10 个 `typing.Protocol`：LLM / Embedder / VectorStore / LexicalIndex / Reranker / DocStore / Observer / Tool / Planner / Chunker |
| 适配器 | `src/worldforge/adapters/` | 每个端口一个零依赖默认实现 + 若干生产后端（惰性导入 + 降级） |
| 检索 | `src/worldforge/retrieval/` | 分块编排、稀疏/稠密双路、RRF 融合、证据单行化渲染 |
| Agent | `src/worldforge/agents/` | 工具注册与确定性路由、ReAct、规则计划器、DAG 拓扑排序、接地核验 |
| 服务 | `src/worldforge/serving/` | 标准库 ThreadingHTTPServer，12 个端点，含 OpenAI 兼容接口 |
| 评估 | `src/worldforge/eval/` | 检索/忠实度指标、门限判定、黄金集套件 |
| 可观测 | `src/worldforge/observability/` | span、指标、确定性视图、Prometheus 导出 |

依赖方向由 `tests/test_layering.py` 静态校验：核心层不得反向 import 上层，且 `core/` 与 `ports/` 只允许标准库（唯一例外是 `core/accel.py` 的可选 numpy 加速，由单独用例守护其 `try/except ImportError` 结构）。

---

## 核心设计：默认实现必须是真实现

这是本项目与"接口齐全但全是 stub"的 demo 最主要的区别。默认配置就是被持续验证的配置：

| 能力 | 默认实现 | 它真的在做什么 |
|---|---|---|
| 嵌入 | `HashingEmbedder` | BLAKE2b 有符号散列 + 中文 bigram，L2 归一化，确定性可复现 |
| 向量库 | `InMemoryExactIndex` | 精确余弦检索，支持 `where` 过滤与指纹校验 |
| 稀疏索引 | `BM25Index` | 自写 Robertson IDF `ln(1+(N-n+0.5)/(n+0.5))`，恒非负 |
| 重排 | `LexicalReranker` | 查询词 IDF 加权覆盖率 + 位置先验 |
| LLM | `ExtractiveMockLLM` | 从注入证据里做 IDF 加权抽取式回答，不是返回固定字符串 |
| 分块 | `RecursiveCharChunker` | 标题栈继承 + 带重叠的递归切分 |
| 存储 | `MemoryDocStore` | 替换语义（replace 而非 upsert） |
| 网关 | `ThreadingHTTPServer` | 标准库起服务，零框架依赖 |

生产后端（FAISS / Qdrant / fastembed / llama.cpp / Ollama / OpenAI / Elasticsearch / psycopg / OTLP）全部惰性导入，缺失时 `warn_once` + 记 `degraded` 标签 + 回退默认实现，绝不把异常抛给调用方。装了就用，没装也能跑。

---

## 内核的不变量

`docs/ALGORITHM_KERNEL.md` 给出完整推导与伪代码，代码里每条都写成了断言。摘录：

| 模块 | 不变量 | 实测残差 |
|---|---|---|
| numeric | softmax 平移不变 `softmax(x+c) == softmax(x)` | c=1e3 时 1e-13；c=1e6 时 2.2e-11（双精度表示误差下界） |
| linalg | Cholesky 重构 `L Lᵀ == A`；Jacobi 残差 `‖Av-λv‖` | 1e-10 量级 |
| prob | `KL(p‖p)==0`、`JS <= ln2`、`KL>=0` | 1e-12 |
| attention | 分块 FlashAttention 与朴素实现数值等价 | 2.2e-16 |
| moe | 门控权重每行 top-k 和为 1；计数守恒 | 1e-12 / 整数精确 |
| diffusion | 用真值噪声还原 x0 | 0.0 |
| rl | GAE λ=1 时 `A+V==R`；λ=0 时 `A==δ` | 1.8e-15 / 1e-12 |
| gnn | `L_sym @ D^{1/2}1 == 0`；对称归一化谱半径 <= 1 | 1e-10 |
| bayes | CAVI 的 ELBO 单调不减 | min Δ = -5.7e-14 |

**MoE 负载均衡损失的一个常见声称被实测推翻**：论文里 `E·ΣfᵢPᵢ >= 1` 并非不等式下界，20000 组随机配置实测最小值 0.976。本仓库采用可证下界 `E·ΣfᵢPᵢ >= Σfᵢ² >= 1/E` 与上界 `E·max(P)`，断言也跟着改了。

---

## 实测数据

CPU-only 零依赖栈，单机 Python 3.13.14：

| 指标 | 数值 |
|---|---|
| 用例数 | 259 passed / 0 failed |
| HTTP 端到端断言 | 20 passed / 0 failed |
| 检索 p50 / p95 | 0.51 ms / 1.09 ms |
| 检索吞吐 | 1717 QPS |
| 端到端问答均值 | 4.70 ms |
| doc_hit_rate | 1.000 |
| MRR | 0.500 |
| faithfulness | 0.496 |
| 禁用字符门禁 | 145 文件 0 findings |

两环境两列数字的差异来源写在 `docs/DETERMINISM.md`：numpy 路径与纯 Python 路径存在末位浮点差异，RRF 的 rank 翻转会把 1e-11 的分差放大成 2.64e-4 的分数跳变（约 2.6e7 倍），因此融合前统一 `round(score, 6)`。

---

## 门禁策略

评估门限 = `max(floor, baseline * 0.95)`，floor 是会失败的硬下限：

```
doc_hit_rate    >= 0.60
reciprocal_rank >= 0.50
recall_at_5     >= 0.30
faithfulness    >= 0.40
```

`python scripts/record_baseline.py --runs 5` 生成实测基线并回填，`threshold = max(floor, baseline * 0.95)` 只会比 floor 更严。未回填时直接取 floor，没有空窗期。

两条硬约束写在 `docs/EVAL.md`：每个评估 case 都在**全新管道**上跑（复用运行期单例管道会污染指标）；工具路由类用例**不进检索聚合**，单独计数上报。

---

## 目录

```
worldforge-ai/
+-- README.md                    本文件
+-- ARCHITECTURE.md              架构总纲（内容同 docs/ARCHITECTURE.md）
+-- pyproject.toml               运行时依赖为空，extras 分组
+-- Dockerfile                   多阶段构建
+-- docker-compose.yml           可选侧车：Qdrant + Ollama
+-- Makefile                     make verify / test / smoke / baseline / bench
+-- .env.example                 全部环境变量
+-- .github/workflows/ci.yml     ubuntu/windows x py3.11/3.12/3.13 + extras parity
+-- scripts/
|   +-- verify.py                六阶段门禁链
|   +-- smoke.py                 自包含 HTTP E2E
|   +-- bench.py                 吞吐/延迟基准
|   +-- record_baseline.py       生成评估基线
|   +-- scan_emoji.py            P0 字符门禁
+-- src/worldforge/              见上文架构表
+-- tests/                       259 个用例，含分层静态检查
+-- docs/
    +-- ARCHITECTURE.md          分层、10 个 Port 精确签名、失败模式表（18 行）
    +-- ALGORITHM_KERNEL.md      十个内核模块的数学、伪代码、不变量
    +-- OSS_STACK.md             六层顶级开源选型（联网核实版本与许可）
    +-- API.md                   12 个端点参考
    +-- ADAPTERS.md              如何新增一个后端实现
    +-- EVAL.md                  评估与基线操作手册
    +-- DETERMINISM.md           确定性约定细则
    +-- decisions/ADR-0001..0005 架构决策记录
```

---

## 可选后端

```bash
pip install -e ".[vector-embed-fastembed,vector-faiss,llm-llamacpp]"
export WORLDFORGE_EMBEDDER=fastembed
export WORLDFORGE_VECTORSTORE=faiss
export WORLDFORGE_LLM=llamacpp
```

CPU-only 推荐组合与各后端的适用边界见 `docs/OSS_STACK.md`。

**许可红线**（ADR-0005）：MinIO / Garage 为 AGPL-3.0，TensorRT 为 NVIDIA 专有许可，二者**禁止进入任何 extras**，只能作为外部服务端点被对接。CI 有对应扫描门禁。

---

## 已知限制

| 项 | 说明 |
|---|---|
| 内核是教学级实现 | 目标是可验证不变量，不是性能。生产推理请走 `llm-llamacpp` / vLLM |
| `ExtractiveMockLLM` 不做生成 | 它只做抽取式回答。生成能力由 `llm-openai` / `llm-ollama` / `llm-llamacpp` 提供 |
| `SemanticChunker` / `CrossEncoderReranker` 需 fastembed | 缺失时自动降级，不会报错但也不是语义质量 |
| HTTP 层未做鉴权与限流 | 面向内网与本地，生产暴露前必须加网关 |
| 并发写入未加锁 | `ThreadingHTTPServer` 下共享索引的写操作需外部串行化 |

---

## 引用

内核涉及的经典工作：Vaswani et al. 2017（Transformer）、Dao et al. 2022（FlashAttention）、Fedus et al. 2022（MoE）、Ho et al. 2020（DDPM）、Song et al. 2021（DDIM）、Schulman et al. 2017（PPO）、Hamilton et al. 2017（GraphSAGE）、Jordan et al. 1999（变分推断）、Robertson & Zaragoza 2009（BM25 概率化解释）。完整清单与链接见 `docs/ALGORITHM_KERNEL.md`。

## License

MIT，见 `LICENSE`。
