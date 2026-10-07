# worldforge-ai 顶级开源复用选型清单

> 检索日期：2026-10-07（版本号与活跃度均为该日期联网检索结果，来源见文末）
> 维护者：钱瞻远（AI 行业研究）
> 适用环境：Windows 11 + WorkBuddy 沙箱，无 GPU，CPU-only，Python 3.13.14，pip 需走 `pip install --target <dir>`

---

## 0. 核心结论

1. 六层中真正能在本机（CPU-only、无 Docker 实跑、pip 走 `--target`）落地的只有**数学基础层**和**代码工程层**；算法核心层、技术栈层的重型依赖全部降级为「可选 extras + 环境变量切换」，系统基础设施层与工具链层降级为「文档与运维引用」。
2. 架构上坚持「**核心零第三方依赖（裸 CPython 3.13 可跑）+ Protocol 可注入**」：默认注入零依赖真实实现（非 stub），生产后端（PyTorch / vLLM / FAISS / Qdrant / Milvus）只在 extras 安装后才通过环境变量 `WORLDFORGE_BACKEND_*` 激活。
3. 合规上唯一的硬冲突是 **AGPL-3.0**（MinIO、Garage），必须严格限定为「外部服务、不 vendor、不链接」；**TensorRT 为 NVIDIA 专有许可**，不得进入任何默认依赖路径。其余 Apache-2.0 / BSD-3 / MIT 与 MIT 主许可兼容。

---

## 1. 六层复用选型矩阵

### 1.1 数学基础层（数值稳定运算、自动微分、分解、散度）

| 项目 | 当前版本 | License | 用途 | 为何选它（一句话） | 在本仓库中的角色 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |
| CPython 标准库 `math` / `statistics` / `fractions` / `decimal` | 3.13.14（本机实测 2026-10-07） | PSF-2.0 | logsumexp 稳定化、精确有理数、十进制高精度 | 零依赖，是「裸环境可跑」这条硬约束的唯一可靠底座，`math.log1p` / `math.fsum` 直接解决数值稳定问题 | **默认零依赖实现**（core 层唯一允许的无条件依赖） | mpmath（高精度时才用） |
| NumPy | 2.5.3（本机实测 2026-10-07） | BSD-3-Clause | 张量基座、广播、线性代数 | 事实标准 ABI，且本机已装；NumPy 2.x 与 SciPy / sklearn 版本已对齐，无需额外装轮子 | **可选加速后端** `[math-numpy]`，未装时自动回落到纯 Python 列表实现 | JAX numpy API（需 extras） |
| SciPy | 1.18.1（本机实测 2026-10-07） | BSD-3-Clause | 矩阵分解（SVD / QR / Cholesky / eig）、特殊函数、优化 | `scipy.linalg` 的 LAPACK 绑定是 CPU 上唯一成熟的稠密分解实现，自研 SVD 无意义 | **可选后端** `[math-scipy]`，仅用于分解与特殊函数子模块 | numpy.linalg（功能子集） |
| scikit-learn | 1.9.1（本机实测 2026-10-07） | BSD-3-Clause | 距离度量、聚类、降维、评测指标 | 评测与基线对照的现成参考实现，避免自研指标出现口径偏差 | **仅参考 + 测试基线**（不作为运行时依赖，只在 tests 中对照） | 自研零依赖实现为主 |
| JAX | 0.11.1（2026-08-18，PyPI jax-team） | Apache-2.0 | 自动微分（`grad` / `jacrev`）、`jit` / `vmap` / `pmap` 可组合变换 | 函数式变换组合能力是 autograd 里最干净的抽象，适合把「自微分」做成 Protocol 的一个实现；0.11.0 已放弃 Python 3.11，本机 3.13 可用 | **可选后端** `[autodiff-jax]`，实现 `AutodiffProtocol` | PyTorch autograd、`autograd` 轻量包 |
| PyMC | 6.3.1（2026-08-16，PyMC 官方 + Wikipedia） | Apache-2.0 | 贝叶斯推断：NUTS / HMC / ADVI / SMC | 唯一同时具备生产级 NUTS 与变分推断的 Python 原生库，v6 起默认 sampler 可切 Rust 版 nutpie，CPU 上仍可用 | **可选后端** `[bayes-pymc]`（注意：要求 Python >= 3.12，本机 3.13 满足） | 自研零依赖 MCMC（小规模）/ NumPyro（待验证） |
| mpmath | 待验证 | BSD-3-Clause | 任意精度浮点、特殊函数交叉校验 | 用于给自研零依赖数值实现做「高精度真值对照」，是测试资产不是运行时依赖 | **仅测试参考** | `decimal` 标准库 |

### 1.2 算法核心层（Transformer / FlashAttention、MoE、Diffusion、RAG、PPO / RLHF、GNN、贝叶斯推断）

| 项目 | 当前版本 | License | 用途 | 为何选它（一句话） | 在本仓库中的角色 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |
| FlashAttention（FA2 稳定线 / FA4 beta） | FA2 v2.8.3.post1（2026-06，稳定）；FA4 `fa4-v4.0.0.beta33`（2026-09-30，beta） | BSD-3-Clause | IO 感知的精确注意力内核 | 数学上与原注意力完全等价、只改内存搬运，是「注意力必须可切换内核」这一 Protocol 的标杆实现 | **仅参考**（本机无 GPU，FA2/FA3/FA4 全部不可装不可用）；自研分块 softmax 参考其 online-softmax 算法 | PyTorch `scaled_dot_product_attention` 自动派发 |
| PyTorch | 2.14.0 GA（2026-09-02，PyTorch 官方 dev-discuss）；2.14.1 补丁线待验证 | BSD-3-Clause | 张量计算 + autograd + 分布式 + 导出 | 生态默认基座；2.12 起源构建强制 C++20 且最低 CUDA 12.6，CPU wheel 仍是唯一能覆盖全部算法原型的框架 | **可选后端** `[train-torch]`；本机 CPU-only，默认关闭 | JAX（纯函数式）/ 自研零依赖小模型 |
| HuggingFace Transformers | v5.18.0（2026-09-30） | Apache-2.0 | 模型定义层（架构 + 权重转换 + 量化配置） | v5 起官方自我定位就是「模型定义层，不是服务引擎」，正好契合本仓库「复用架构定义、自研编排」的分工 | **可选后端** `[model-hf]`；仅用于加载架构与权重，不用于服务 | 自研零依赖 Transformer 参考实现（默认） |
| vLLM | v0.31.0（2026-10-05） | Apache-2.0 | 高吞吐 LLM 推理服务（PagedAttention / continuous batching / OpenAI 兼容 API） | 开源服务事实标准，0.29 起 Model Runner V2 全量默认、0.31 引入 Fast Restart 权重缓存守护进程 | **仅参考 / 可选后端** `[llm-vllm]`（本机无 GPU，不可实跑） | llama.cpp（CPU）/ SGLang（待验证） |
| llama.cpp | v0.6.0（2026-10-06，ggml v0.26.0） | MIT | GGUF 量化模型的 CPU / 本地推理 | MIT 许可 + 纯 CPU 可跑 + ggml v0.26 新增 CPU 后端 BF16，是本机唯一真正可用的 LLM 推理路径 | **可选后端** `[llm-llamacpp]`（推荐 CPU 场景首选） | `llama-cpp-python` 绑定（版本待验证）/ Ollama |
| TRL（Transformer Reinforcement Learning） | v1.14.0（最新，releases.sh）；v1.13.0（2026-09-10）已移除 PPOTrainer | Apache-2.0 | SFT / DPO / GRPO / 奖励建模 | 官方已明确 GRPO 取代 PPO 为默认对齐路径（v1.13 移除 PPOTrainer），直接跟随主线可避免自研 PPO 的稳定性坑 | **仅参考**（算法设计与超参对照）；本机不可训练 | 自研零依赖 GRPO 简化版（默认演示）/ Axolotl（待验证） |
| PyTorch Geometric（PyG） | 2.8.0.post1（2026-07-20）；滚动文档标注 2.9.0（待验证） | MIT | GNN 层、图小批量采样、异构图 | GNN 事实标准；2.8 要求 PyTorch 2.9+ 且把加速依赖收敛到 pyg-lib 0.7.0，依赖面反而变干净 | **仅参考 / 可选后端** `[gnn-pyg]`（本机不可装） | 自研零依赖消息传递（默认）/ DGL（待验证） |
| LlamaIndex | 0.14.25（2026-09-21） | MIT | 检索优先的索引 / 查询引擎 / Workflows | 检索默认件最全（混合检索、rerank、分块开箱即用），per-query 开销约 6ms 是三者中最低，适合做 RAG Protocol 的参考实现 | **仅参考**；本仓库默认走自研零依赖 BM25 + 稠密向量混合检索 | LangChain 1.3.15 / Haystack 2.29（Apache-2.0） |
| LangChain + LangGraph | langchain 1.3.15（2026-09-20 前后）；langgraph 1.2.10 | MIT | Agent 编排、状态图、检查点 / 持久化 | v1.0 后稳定性契约明确，LangGraph 提供 durable execution 与 time-travel 调试，是「多步 Agent 状态机」的默认答案 | **仅参考**（本仓库用自研零依赖状态机 + Protocol） | LlamaIndex Workflows / CrewAI 1.15.22 |
| Diffusers | 待验证 | Apache-2.0 | 扩散模型调度器与管线 | 调度器（DDPM / DDIM / Euler / DPM-Solver）实现成熟，自研无收益 | **仅参考**（算法与调度器公式对照） | 自研零依赖调度器（默认演示） |

### 1.3 代码工程层（类型注解、asyncio、pytest、ruff、pre-commit）

| 项目 | 当前版本 | License | 用途 | 为何选它（一句话） | 在本仓库中的角色 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |
| `typing` / `typing_extensions`（Protocol, runtime_checkable, TypeAlias） | stdlib（3.13.14）；typing_extensions 待验证 | PSF-2.0 / PSF-2.0 | 结构化类型与可注入 Protocol 定义 | PEP 544 的 `Protocol` 正是「默认零依赖实现 + 可注入生产后端」这一架构的原生表达方式，无需任何第三方 | **默认实现**（core 必用） | `abc.ABC`（不支持结构化子类型） |
| `asyncio` / `contextlib` / `dataclasses` | stdlib 3.13.14 | PSF-2.0 | 并发原语、资源生命周期、配置数据类 | 零依赖且 3.13 的 asyncio 已足够稳定，避免引入 anyio / trio 造成运行时依赖 | **默认实现**（core 必用） | anyio（可选 extras `[async-anyio]`，待验证） |
| pytest | 9.1.1（本机实测 2026-10-07） | MIT | 单元测试、fixture、参数化 | 本机已装且为事实标准；`pytest.importorskip` 天然适配「无 extras 时跳过后端测试」的分层测试策略 | **默认实现**（`[dev]` 依赖） | unittest（stdlib，作为零依赖兜底） |
| Ruff | 0.16.8（2026-09-16，astral-sh/ruff CHANGELOG） | MIT | Lint + format 二合一 | Rust 实现、单二进制、无 Python 运行时依赖，一条工具替代 flake8 + isort + black，CI 冷启动成本最低 | **默认实现**（`[dev]` + pre-commit + CI） | flake8 + black + isort（体积大、慢） |
| pre-commit | 待验证（4.x 线） | MIT | 本地提交前钩子编排 | 与 ruff-pre-commit hooks 直接对接，把 lint 左移到 commit 阶段 | **默认实现**（`[dev]`，配置文件随仓库分发） | 纯 GitHub Actions（反馈更慢） |
| mypy / Astral ty | 待验证 | MIT | 静态类型检查 | `ty` 与 ruff 同厂、共享配置，渐进式接入成本低；但两者均为 beta 特性较多，先做可选 | **可选** `[dev-strict]`，不作为 CI 门禁 | pyright（Microsoft，待验证） |
| pytest-cov / hypothesis | 待验证 | MIT / MPL-2.0 | 覆盖率与属性测试 | 属性测试对「零依赖实现 vs SciPy 参考实现」的数值等价性验证特别有效 | **可选** `[dev-strict]` | 手写随机对照测试 |

> 注意：`hypothesis` 为 MPL-2.0，与 MIT 兼容但需注意文件级 copyleft，仅在 `[dev-strict]` 中引用、不 vendor。

### 1.4 技术栈层（PyTorch / JAX、HuggingFace、vLLM、LlamaIndex / LangChain、Ray、ONNX / TensorRT）

| 项目 | 当前版本 | License | 用途 | 为何选它（一句话） | 在本仓库中的角色 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch | 2.14.0（2026-09-02） | BSD-3-Clause | 训练 / 推理主框架（见 1.2） | 生态默认，导出与部署链路最全 | **可选后端** `[train-torch]` | JAX 0.11.1 |
| JAX + jaxlib | 0.11.1（2026-08-18，PyPI jax-team） | Apache-2.0 | 可组合函数变换 + XLA 编译 | EffVer 版本策略（0.10.0/0.11.0 为高破坏性）说明升级成本可预期；纯函数式适合做数值 Protocol 的一个实现 | **可选后端** `[autodiff-jax]`，jax / jaxlib 版本必须严格一致 | PyTorch |
| HuggingFace Transformers | v5.18.0（2026-09-30） | Apache-2.0 | 模型与 tokenizer 定义层 | v5 起官方明确定位为「模型定义层」，训练交给第三方 trainer、部署交给 vLLM / llama.cpp | **可选后端** `[model-hf]` | 自研零依赖实现 |
| vLLM | v0.31.0（2026-10-05） | Apache-2.0 | 生产级 LLM 服务 | 0.31 引入 `vllm preload` 权重缓存守护进程，显著削减重启加载时间；OpenAI 兼容 API 便于灰度替换 | **仅参考 / 可选后端** `[llm-vllm]` | SGLang / TGI（TGI 已于 2026 年归档，官方建议改用 vLLM 或 SGLang） |
| Ray | 2.58.0（2026-08-23） | Apache-2.0 | 分布式执行、Train / Data / Serve | 分布式 Python 事实标准；KubeRay v1.4.x 起官方记录的 Ray 最低兼容线为 2.38.0 | **仅参考 / 可选后端** `[serve-ray]`（本机不可实跑） | 自研零依赖进程池（默认）/ Dask（待验证） |
| ONNX Runtime | 1.28.0（2026-07-25） | MIT | 跨硬件推理 EP 抽象（CPU / CUDA / TensorRT / OpenVINO / CoreML） | Execution Provider 抽象正是「同一 Protocol 多后端」的工业级范本，且 MIT 许可无合规负担 | **可选后端** `[infer-onnx]`（CPU EP 本机可用，是 CPU 场景的优选） | 自研零依赖推理 / OpenVINO（待验证） |
| NVIDIA TensorRT | 10.16.0.72（随 Triton 26.03 容器分发，2026-05） | **NVIDIA 专有（非开源）** | GPU 推理图优化与引擎编译 | 性能天花板，但许可为专有且不可再分发 | **仅文档引用**，严禁进入任何 extras 依赖声明 | ONNX Runtime CPU EP / OpenVINO |
| sentence-transformers | 6.1.0（2026-09-18） | Apache-2.0 | 稠密 / 稀疏 / 多向量嵌入与 rerank | v6.0.0 起要求 transformers >= 5.0，模型生态最全（15000+ 预训练） | **可选后端** `[embed-st]`（本机不可装，依赖 torch） | fastembed（CPU 友好） |
| fastembed | 0.8.0（2026-03-23） | Apache-2.0 | 基于 ONNX Runtime 的轻量嵌入 | 绕开 PyTorch、纯 Python wheel，专门为 serverless 与 CPU-only 环境设计，是本机唯一现实可行的嵌入方案 | **可选后端** `[embed-fastembed]`（CPU 场景首选） | 自研零依赖哈希嵌入（默认，用于冒烟） |
| FAISS | 1.15.0（2026-07-31；conda 索引 2026-08-20） | MIT | ANN 向量索引（HNSW / IVF / PQ / CAGRA） | MIT 许可 + `faiss-cpu` 有 Python 3.13 轮子 + 无 torch 依赖，是 CPU 场景唯一能落地的顶级 ANN 库 | **可选后端** `[vector-faiss]`（CPU 场景首选） | hnswlib（待验证）/ 自研零依赖暴力检索（默认） |

### 1.5 系统基础设施层（Docker / compose、K8s、Milvus / Qdrant、对象存储、GPU 调度）

| 项目 | 当前版本 | License | 用途 | 为何选它（一句话） | 在本仓库中的角色 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |
| Docker Engine + Compose | Engine v29.1（GitHub 托管 runner，2026-02 起）；Compose V2 5.5.0（2026-08-17，Chocolatey 索引，待验证与上游对齐） | Apache-2.0 | 本地多服务编排与可复现环境 | compose v2 的 healthcheck 条件（`depends_on` 等待健康）显著降低本地栈启动竞态 | **仅文档引用**（本机沙箱不能实跑 Docker，只交付 compose 文件作为部署资产） | Podman / nerdctl（待验证） |
| Kubernetes | 1.37.1（2026-09-15，最新）；1.34 将于 2026-10-27 EOL | Apache-2.0 | 集群编排、Workload / PodGroup 与 gang scheduling（v1.37 进入 Beta） | v1.37 把 Workload / PodGroup / 工作负载感知抢占推进到 Beta，原生支持 JobSet / LWS 类分布式 AI 负载 | **仅文档引用**（交付 manifest 模板） | K3s / kind（本地验证，待验证） |
| KubeRay | v1.7.1（2026-09-21） | Apache-2.0 | Ray on K8s 的 operator（RayCluster / RayJob / RayService） | v1.7 把 History Server 推进到 Beta 并内置 RocksDB GCS 容错（去掉外部 Redis），运维组件数下降 | **仅文档引用**（配合 `[serve-ray]`） | 裸 Ray 集群（不推荐）/ Volcano / Kueue |
| Qdrant | 1.19.1（2026-09-04） | Apache-2.0 | 向量数据库（Rust，过滤检索强项） | ACORN 把元数据过滤折进 HNSW 遍历，过滤命中率低时也不塌速；Rust 底座内存占用低，4GB 实例可跑百万级 | **可选后端** `[vector-qdrant]`（生产推荐，作为 `VectorStoreProtocol` 实现） | Milvus / pgvector |
| Milvus | 3.0.2（2026-09-20） | Apache-2.0 | 十亿级向量的分布式向量库 | 存算分离架构 + 索引种类最全（HNSW / IVF / SCANN / DiskANN），是超大规模时的唯一选择 | **可选后端** `[vector-milvus]`（仅在生产规模文档中出现） | Qdrant / Weaviate |
| LanceDB | 0.39.0（2026-09-17） | Apache-2.0 | 嵌入式列式向量库 | 进程内嵌入、无需服务端，是「单机可跑 + 可持久化」的中间档 | **可选后端** `[vector-lancedb]` | Chroma 1.5.9（Apache-2.0）/ 自研 JSONL 持久化（默认） |
| pgvector | 0.8.6（2026-07-29） | PostgreSQL License | PostgreSQL 的向量扩展 | 已有 Postgres 的团队零新增组件，事务与 SQL 过滤能力是向量库里最强的 | **仅文档引用** | Qdrant |
| MinIO | 社区仓库已于 2026-04-25 归档（read-only）；2025-10-15 起停止发布社区镜像 | **AGPL-3.0** | S3 兼容对象存储 | 曾是自托管 S3 默认解，但社区版已归档且停止安全修复，新项目不应再押注 | **仅文档引用（且标注已归档，不推荐）** | Garage（AGPL-3.0）/ SeaweedFS（Apache-2.0）/ 本地文件系统（默认） |
| SeaweedFS | 待验证 | Apache-2.0 | S3 兼容对象存储 | Apache-2.0 许可，是 MinIO 归档后在合规上最干净的自托管 S3 替代 | **仅文档引用**（推荐替代 MinIO） | Garage |
| NVIDIA GPU Operator | v26.7.0（CalVer YY.MM.PP，2026-08 文档示例）；验证 K8s 1.33 到 1.36 | Apache-2.0 | K8s 上的 GPU 驱动 / device plugin / DRA 统一管理 | 把 GPU 暴露成 `nvidia.com/gpu` 资源，让 Ray 与 K8s 调度统一；但升级只支持跨一个大版本，不能跳版本 | **仅文档引用**（本机无 GPU） | 手动 device plugin（不推荐） |

### 1.6 软件工具链层（GitHub Actions、MLflow / DVC、SemVer、Dependabot）

| 项目 | 当前版本 | License | 用途 | 为何选它（一句话） | 在本仓库中的角色 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |
| GitHub Actions | 平台服务（runner 镜像 2026-01-30 起预装 Docker v29.1 / Compose v2.40+） | 平台服务（非 OSS 许可） | CI：矩阵测试、lint、构建、发布 | 与仓库同平台、零额外账号，且 runner 已预装 Docker 与 Compose，交付 compose 资产可直接验证 | **默认实现**（`.github/workflows/` 随仓库分发） | GitLab CI / Jenkins（待验证） |
| actions/checkout 等官方 action | 待验证（v4 / v5 线） | MIT | 流水线原子步骤 | 官方维护、有 SHA pin 可固化，是供应链安全基线 | **默认实现**（SHA pin） | 自写脚本（不推荐） |
| MLflow | 3.15.1（2026-08-03） | Apache-2.0 | 实验追踪、模型注册、GenAI trace | OSS 且自带 model registry 与 RBAC，是自托管实验追踪的默认解 | **仅文档引用 / 可选** `[track-mlflow]`（默认走零依赖 JSONL 指标落盘） | W&B / ClearML（均为外部服务） |
| DVC | 3.67.1（2026-03-31） | Apache-2.0 | 数据与模型版本管理（Git 式） | 用 `.dvc` 指针 + `dvc.lock` 把数据指纹纳入 Git 审查，是「可复现」的现成答案 | **仅文档引用**（小仓库默认不引入） | Git LFS（待验证） |
| SemVer 2.0.0 | 规范 2.0.0（2013，长期稳定） | CC-BY-3.0（规范文本） | 版本号语义 | 事实标准；本仓库 extras 分层需要明确的 MAJOR / MINOR / PATCH 语义来隔离破坏性变更 | **默认实现**（发布流程强制） | CalVer（不适合本仓库） |
| Conventional Commits | 1.0.0（规范，长期稳定） | CC-BY-3.0 | 提交信息规范 | 让 CHANGELOG 与版本号可自动生成，是开源仓库被外部贡献者接受的前提 | **默认实现** | 自由格式（不推荐） |
| Dependabot | 平台服务 | 平台服务 | 依赖漏洞与版本自动 PR | GitHub 原生、零配置成本，对 60+ 个可选依赖的仓库尤其重要 | **默认实现**（`.github/dependabot.yml`） | Renovate（待验证） |
| SPDX / REUSE | SPDX 3.0 线（待验证）；REUSE 规范 3.x（待验证） | CC-BY-3.0 | 许可标识与合规自动化 | 六层 60+ 依赖的许可是本仓库最大合规面，机器可读的 SPDX ID 是唯一可审计方式 | **默认实现**（每个源文件头 + `LICENSES/` 目录） | 手写 NOTICE（易漏） |

---

## 2. 「本仓库究竟怎么用上它们」落地表

### 2.1 (a) 默认零依赖自研实现（保证裸 CPython 3.13 可跑，无任何 pip 安装）

| 能力 | 默认实现 | 依赖 | 说明 |
| --- | --- | --- | --- |
| 数值稳定运算 | `core/mathx/stable.py`：logsumexp、stable softmax、safe log | stdlib `math` | 直接对标 SciPy 同名函数，tests 中与 SciPy 对照（无 SciPy 时跳过） |
| 自动微分 | `core/mathx/autodiff.py`：标量与小向量反向模式 | stdlib | 实现 `AutodiffProtocol`，与 JAX `grad` 结果做数值对照 |
| 矩阵分解 | `core/mathx/linalg.py`：幂迭代 / QR（Householder）/ Cholesky | stdlib | 小规模可用；大规模走 `[math-scipy]` |
| 散度与距离 | `core/mathx/divergence.py`：KL / JS / Renyi / Wasserstein-1D | stdlib | 与 `scipy.special.rel_entr` 对照 |
| 贝叶斯推断 | `core/algo/bayes.py`：MH 采样 + 共轭解析解 | stdlib `random` | 小规模可用；生产走 `[bayes-pymc]` |
| 向量检索 | `core/retrieval/brute.py`：归一化余弦暴力检索 + JSONL 持久化 | stdlib | 十万级以内可用；生产走 `[vector-*]` |
| 嵌入 | `core/embed/hashing.py`：确定性哈希嵌入（可复现） | stdlib `hashlib` | 仅用于冒烟与测试；语义质量不达标，必须标注 |
| RAG 检索 | `core/retrieval/hybrid.py`：BM25 + 稠密向量 RRF 融合 | stdlib | 默认可用；生产走 `[embed-fastembed]` + `[vector-faiss]` |
| 编排状态机 | `core/orchestrator/stategraph.py`：显式状态图 + 检查点序列化 | stdlib `asyncio` / `dataclasses` | 对标 LangGraph 语义，但零依赖 |
| 实验追踪 | `core/tracking/jsonl.py`：指标与配置 JSONL 落盘 | stdlib | 生产走 `[track-mlflow]` |
| 测试 | `unittest` 兜底 + `pytest` 主跑 | stdlib / pytest | `pytest.importorskip` 实现「无 extras 自动跳过后端测试」 |

### 2.2 (b) 通过 extras 可切换的生产后端（环境变量注入）

| extras 名称 | 注入的依赖 | 环境变量开关 | 实现的 Protocol | 本机（CPU / 无 GPU）可行性 |
| --- | --- | --- | --- | --- |
| `[math-numpy]` | numpy >= 2.5 | `WORLDFORGE_BACKEND_MATH=numpy` | `MathBackend` | 可行（本机已装 2.5.3） |
| `[math-scipy]` | scipy >= 1.18 | `WORLDFORGE_BACKEND_LINALG=scipy` | `LinAlgBackend` | 可行（本机已装 1.18.1） |
| `[autodiff-jax]` | jax == 0.11.1, jaxlib == 0.11.1 | `WORLDFORGE_BACKEND_AUTODIFF=jax` | `AutodiffProtocol` | 可行（CPU wheel 存在），但体积较大 |
| `[bayes-pymc]` | pymc >= 6.3 | `WORLDFORGE_BACKEND_BAYES=pymc` | `BayesBackend` | 可行（需 Python >= 3.12，本机 3.13 满足）；依赖 pytensor 编译 |
| `[embed-fastembed]` | fastembed >= 0.8 | `WORLDFORGE_BACKEND_EMBED=fastembed` | `Embedder` | **可行，CPU 首选**（依赖 onnxruntime，无 torch） |
| `[embed-st]` | sentence-transformers >= 6.1 | `WORLDFORGE_BACKEND_EMBED=sentence-transformers` | `Embedder` | 不可行（强制 transformers >= 5 且拉入 torch） |
| `[vector-faiss]` | faiss-cpu >= 1.15 | `WORLDFORGE_BACKEND_VECTOR=faiss` | `VectorStore` | **可行，CPU 首选**（MIT，无 torch 依赖） |
| `[vector-lancedb]` | lancedb >= 0.39 | `WORLDFORGE_BACKEND_VECTOR=lancedb` | `VectorStore` | 待验证（需确认 Python 3.13 wheel） |
| `[vector-qdrant]` | qdrant-client >= 1.19 | `WORLDFORGE_BACKEND_VECTOR=qdrant` | `VectorStore` | 可行（客户端轻量），服务端需外部部署 |
| `[vector-milvus]` | pymilvus >= 3.0 | `WORLDFORGE_BACKEND_VECTOR=milvus` | `VectorStore` | 仅文档（服务端需集群） |
| `[infer-onnx]` | onnxruntime >= 1.28 | `WORLDFORGE_BACKEND_INFER=onnxruntime` | `InferenceBackend` | **可行，CPU EP 可用** |
| `[llm-llamacpp]` | llama-cpp-python（版本待验证） | `WORLDFORGE_BACKEND_LLM=llamacpp` | `LLMBackend` | **可行，CPU 首选**（llama.cpp v0.6.0 MIT，有 Windows x64 CPU 预编译包） |
| `[llm-vllm]` | vllm >= 0.31 | `WORLDFORGE_BACKEND_LLM=vllm` | `LLMBackend` | 不可行（强依赖 GPU / CUDA） |
| `[train-torch]` | torch >= 2.14 | `WORLDFORGE_BACKEND_TRAIN=torch` | `TrainBackend` | 理论可行（CPU wheel），但沙箱安装体积与时间成本高，标为「不默认验证」 |
| `[model-hf]` | transformers >= 5.18 | `WORLDFORGE_BACKEND_MODEL=hf` | `ModelDef` | 同上，不默认验证 |
| `[gnn-pyg]` | torch-geometric >= 2.8 | `WORLDFORGE_BACKEND_GNN=pyg` | `GNNLayer` | 不可行（需 torch >= 2.9） |
| `[serve-ray]` | ray >= 2.58 | `WORLDFORGE_BACKEND_EXEC=ray` | `Executor` | 待验证（Windows 支持 historically 较弱） |
| `[track-mlflow]` | mlflow >= 3.15 | `WORLDFORGE_BACKEND_TRACK=mlflow` | `Tracker` | 可行但重（拉入大量依赖），建议仅在 CI 中启用 |
| `[dev]` | pytest, ruff, pre-commit | 无（直接运行） | 无 | **可行，本机已具备 pytest 9.1.1** |
| `[dev-strict]` | mypy 或 ty, pytest-cov, hypothesis | 无 | 无 | 可选 |

### 2.3 (c) 文档 / 运维层引用（不进 Python 依赖，只作为资产文件交付）

| 资产 | 位置 | 引用项目 | 说明 |
| --- | --- | --- | --- |
| compose 栈 | `deploy/docker-compose.yml` | Docker Compose V2, Qdrant, SeaweedFS | 本机不能实跑 Docker，仅交付文件与 README 说明 |
| K8s manifest | `deploy/k8s/` | Kubernetes 1.37, KubeRay 1.7.1, GPU Operator 26.7 | 标注 `ray.io/v1` API（v1alpha1 已废弃） |
| CI 工作流 | `.github/workflows/` | GitHub Actions, ruff, pytest | SHA pin 所有第三方 action |
| 依赖更新 | `.github/dependabot.yml` | Dependabot | 按 extras 分组，避免可选依赖噪声 |
| 发布流程 | `docs/RELEASE.md`（仅本文件提及，不新建） | SemVer 2.0.0, Conventional Commits | 版本号与 CHANGELOG 自动生成 |
| 合规清单 | 本文件第 3 节 | SPDX / REUSE | 逐项标注 SPDX ID |
| 实验 / 数据管理 | 文档引用 | MLflow 3.15.1, DVC 3.67.1 | 小仓库默认不引入，仅在规模化章节给出迁移路径 |

---

## 3. License 合规小节

本仓库主许可为 MIT。逐项合规标注如下。

### 3.1 许可分布与兼容性总表

| License | SPDX ID | 涉及项目 | 与 MIT 主许可兼容性 | 风险点 |
| --- | --- | --- | --- | --- |
| MIT | `MIT` | FAISS, llama.cpp, PyTorch Geometric, LlamaIndex, LangChain / LangGraph, ONNX Runtime, ruff, pytest, pre-commit, mypy / ty, pytest-cov | 完全兼容 | 无实质风险；需保留版权声明 |
| BSD-3-Clause | `BSD-3-Clause` | PyTorch, NumPy, SciPy, scikit-learn, FlashAttention, mpmath | 兼容 | 无专利授予条款（弱于 Apache-2.0）；禁止用贡献者名义背书（第 3 条） |
| Apache-2.0 | `Apache-2.0` | JAX / jaxlib, transformers, TRL, vLLM, Ray, KubeRay, Qdrant, Milvus, LanceDB, Kubernetes, MLflow, DVC, PyMC, sentence-transformers, fastembed, SeaweedFS, Docker Engine / Compose, GPU Operator, Diffusers | 兼容 | **必须保留 NOTICE 文件并随分发传递**；含专利授予与专利报复条款（触发即终止授权） |
| PostgreSQL License | `PostgreSQL` | pgvector | 兼容（ permissive，类 BSD） | 需注意与具体 Postgres 版本的绑定关系 |
| PSF-2.0 | `PSF-2.0` | CPython 标准库（`math` / `asyncio` / `typing` / `dataclasses`） | 兼容 | 无 |
| MPL-2.0 | `MPL-2.0` | hypothesis | 兼容但为**文件级弱 copyleft** | 若修改 hypothesis 源文件须公开；仅作 dev 依赖、不 vendor 即无风险 |
| **AGPL-3.0** | `AGPL-3.0` | **MinIO, Garage** | **硬冲突** | 网络交互即触发源码公开义务；**严禁 vendor、严禁静态 / 动态链接、严禁随镜像分发**；只能作为独立部署的外部服务通过 HTTP/S3 API 调用 |
| NVIDIA 专有 | `LicenseRef-NVIDIA-Proprietary` | **TensorRT** | **不兼容（非开源）** | 禁止进入 `pyproject.toml` 任何 extras；禁止在 CI 中自动安装；文档中只能给「用户自行安装后可选启用」的说明 |
| 平台服务 | 非 OSS | GitHub Actions, Dependabot | 不涉及（服务条款约束） | 关注 runner 镜像预装组件版本漂移 |
| CC-BY-3.0 | `CC-BY-3.0` | SemVer 规范文本, Conventional Commits 规范文本 | 仅规范文本，不涉及代码 | 引用规范需署名；仓库内复述规范内容时应注明出处 |

### 3.2 三个必须守住的红线

| 风险 | 触发条件 | 处置 |
| --- | --- | --- |
| AGPL 传染 | 把 MinIO / Garage 打进镜像、或在代码中 import 其 AGPL 库 | 对象存储统一抽象为 `ObjectStoreProtocol`，默认实现为本地文件系统；MinIO / Garage / SeaweedFS 均以「外部服务端点 + 环境变量」方式对接，仓库内不出现任何 AGPL 代码 |
| 专有许可污染 | `pyproject.toml` 中声明 TensorRT 依赖，或 CI 自动 `pip install tensorrt` | TensorRT 只出现在文档的「GPU 部署（用户自备）」章节；`[infer-onnx]` 的 TensorRT EP 由用户在本地自行配置 ONNX Runtime 后启用 |
| Apache-2.0 NOTICE 遗漏 | 分发 wheel / sdist 时未附带依赖的 NOTICE | 构建产物根目录附 `NOTICE` 文件，汇总所有 Apache-2.0 依赖的版权与 NOTICE 原文；建议用 REUSE / SPDX 工具在 CI 中校验 |

### 3.3 建议的 CI 合规门禁

| 门禁 | 工具 | 阈值 |
| --- | --- | --- |
| 许可扫描 | `pip-licenses` 或 SPDX SBOM 生成（待验证） | 出现 `AGPL-*` / `GPL-*` / `LicenseRef-*Proprietary` 即 fail |
| 依赖清单快照 | `pip freeze` 到 `requirements.lock` | 每次 release 冻结，配合 Dependabot 分组 PR |
| Action SHA pin | 人工 + 脚本校验 | 禁止 `@v1` 浮动引用 |

---

## 4. CPU / 无 GPU 场景的最优选型建议

### 4.1 本机不可行的重型依赖与替代方案

| 不可行项 | 原因 | 替代方案 |
| --- | --- | --- |
| `torch` / `torchvision` | CPU wheel 体积巨大，沙箱 `pip install --target` 安装耗时与磁盘成本高；且无 GPU 时训练无意义 | 默认零依赖实现；确需时走 `[train-torch]` 并明确标注「不默认验证」 |
| `vllm` | 强依赖 CUDA / GPU，设计目标就是高吞吐 GPU 服务 | **llama.cpp v0.6.0**（MIT，有 Windows x64 CPU 预编译包，GGUF 量化） |
| FlashAttention（FA2 / FA3 / FA4） | 全部为 CUDA 内核，FA4 仅支持 Hopper / Blackwell | PyTorch `scaled_dot_product_attention` 自动派发；本仓库自研分块 online-softmax 参考其算法 |
| Milvus 集群 | 需要多节点与对象存储，本机无 Docker 实跑能力 | **Qdrant**（单节点可跑，4GB 内存百万级）或 **LanceDB**（进程内嵌入）；本机开发直接用自研 JSONL 持久化 |
| MinIO | 社区仓库 2026-04-25 已归档，2025-10-15 起停止发布镜像 | 默认走本地文件系统；文档推荐 **SeaweedFS**（Apache-2.0） |
| TensorRT / GPU Operator / NVIDIA 驱动栈 | 本机无 GPU | 全部降级为「文档引用」，不出现在任何依赖声明 |
| K8s / KubeRay 实跑 | 本机无集群 | 交付 manifest 资产，CI 中用 `kubectl --dry-run` 校验（待验证 runner 是否可用） |
| `sentence-transformers` 6.x | 强制 `transformers >= 5.0`，必然拉入 torch | **fastembed 0.8.0**（ONNX Runtime，无 torch，专为 CPU / serverless 设计） |
| `pymc` 完整编译链 | PyTensor 默认后端已改为 numba，CPU 编译可行但重 | 小规模用自研 MH 采样；仅当需要 NUTS 时启用 `[bayes-pymc]` 并显式 `pytensor.config.linker = "cvm"` |

### 4.2 CPU 场景推荐启用组合

| 场景 | 命令 | 预期能力 |
| --- | --- | --- |
| 最小可跑（默认） | `python -m pytest`（无需任何 pip 安装） | 全部核心算法零依赖跑通 |
| 开发态 | `pip install --target .vendor pytest ruff pre-commit`（`[dev]`） | lint + 测试闭环 |
| CPU 完整栈 | `pip install --target .vendor ".[math-numpy,math-scipy,infer-onnx,embed-fastembed,vector-faiss]"` | 嵌入 + 向量检索 + ONNX 推理 + 数值加速，全部 CPU 可跑 |
| CPU 本地 LLM | 追加 `[llm-llamacpp]` | GGUF 量化模型本地推理（llama.cpp v0.6.0） |
| 不推荐的组合 | `[train-torch]` / `[model-hf]` / `[llm-vllm]` / `[gnn-pyg]` | 沙箱内安装成本高且无 GPU，仅在有 GPU 的机器上验证 |

### 4.3 三条架构纪律（与选型直接相关）

1. **默认实现必须是真实实现而非 stub**：零依赖 BM25 + 哈希嵌入 + 暴力检索必须能返回真实结果，只是规模与质量有上限，并在文档中明确标注上限。
2. **后端切换只能走环境变量**：`WORLDFORGE_BACKEND_*`，禁止在代码中 `try: import torch except ImportError` 式隐式降级，否则行为不可预测。
3. **测试分层**：`tests/core/` 必须裸环境可跑（无第三方依赖）；`tests/backend/` 用 `pytest.importorskip` 挂到对应 extras，CI 矩阵中单独一个 job 装全 extras 跑。

---

## 5. 数据来源与时效

| 项目 | 版本 | 检索日期 | 来源 |
| --- | --- | --- | --- |
| PyTorch | 2.14.0 GA | 2026-10-07 | https://dev-discuss.pytorch.org/t/pytorch-2-14-0-general-availability/3431/1 ；https://www.eol.wiki/pytorch |
| JAX / jaxlib | 0.11.1 | 2026-10-07 | https://pypi.org/user/jax-team/ ；https://implicit-none.com/en/jax-release-notes-2026/ |
| FlashAttention | FA2 v2.8.3.post1 / FA4 fa4-v4.0.0.beta33 | 2026-10-07 | https://hysenlabs.com/en/projects/dao-ailab-flash-attention ；https://handbook.modular.com/kernel-optimization/flashattention/ |
| vLLM | v0.31.0 | 2026-10-07 | https://www.change8.dev/package/vllm ；https://releasealert.dev/pypi/vllm |
| HuggingFace Transformers | v5.18.0 | 2026-10-07 | https://releasealert.dev/github/huggingface/transformers ；https://releases.sh/hugging-face |
| TRL | v1.14.0（最新）/ v1.13.0 | 2026-10-07 | https://releases.sh/hugging-face/trl/highlights ；http://ai-tldr.dev/tools/trl |
| llama.cpp | v0.6.0（ggml v0.26.0） | 2026-10-07 | https://github.com/ggml-org/llama.cpp/releases/ ；https://localmodelwatch.tsuchitsuchi.com/en/2026/10/06/llamacpp-v0-6-0-released |
| PyTorch Geometric | 2.8.0.post1 | 2026-10-07 | https://github.com/pyg-team/pytorch_geometric/releases ；https://anaconda.org/conda-forge/pytorch_geometric |
| PyMC | 6.3.1 | 2026-10-07 | https://discourse.pymc.io/t/release-v6-0-0/17710 ；https://www.wikiwand.com/en/PyMC |
| Ray / KubeRay | 2.58.0 / v1.7.1 | 2026-10-07 | https://www.anyscale.com/blog/kuberay-v1-7 ；https://github.com/ray-project/kuberay/releases |
| ONNX Runtime | 1.28.0 | 2026-10-07 | https://github.com/ImChong/Robotics_Notebooks/blob/main/sources/repos/onnxruntime-official.md ；https://baike.baidu.com/item/ONNX%20Runtime/67914822 |
| TensorRT | 10.16.0.72 | 2026-10-07 | https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-03.html |
| FAISS | 1.15.0 | 2026-10-07 | https://github.com/facebookresearch/faiss （via CSDN 版本演进表）；https://anaconda.org/pytorch/repo |
| Qdrant / Milvus / LanceDB / Weaviate / pgvector / Chroma | 1.19.1 / 3.0.2 / 0.39.0 / 1.39.6 / 0.8.6 / 1.5.9 | 2026-10-07 | https://www.bestfreewebresources.com/best-vector-database-local-rag-2026 （GitHub API 读取于 2026-09-25） |
| MinIO | 社区仓库 2026-04-25 归档 | 2026-10-07 | https://cloudraptor.io/minio-docker-compose |
| Kubernetes | 1.37.1 | 2026-10-07 | https://kubernetes.io/releases/ |
| Docker Engine / Compose | v29.1 / Compose V2 5.5.0 | 2026-10-07 | https://github.blog/changelog/2026-01-30-docker-and-docker-compose-version-upgrades-on-hosted-runners/ ；https://push.chocolatey.org/packages/docker-compose/5.5.0 |
| MLflow / DVC | 3.15.1 / 3.67.1 | 2026-10-07 | https://aib2.io/2026/08/mlflow-started-as-a-databricks-tracking-tool-in-20 |
| LangChain / LangGraph / LlamaIndex | 1.3.15 / 1.2.10 / 0.14.25 | 2026-10-07 | https://tech-insider.org/langchain-vs-llamaindex-vs-crewai-2026 ；https://home.forage.ai/blog/langchain-vs-llamaindex/ |
| Ruff | 0.16.8 | 2026-10-07 | https://newreleases.io/project/github/astral-sh/ruff/release/0.16.7 ；https://awesome-repositories.com/r/astral-sh-ruff |
| sentence-transformers / fastembed | 6.1.0 / 0.8.0 | 2026-10-07 | https://www.hivebook.wiki/wiki/sentence-transformers-text-embeddings-retrieval-reranking ；https://skillfed.io/packages/fastembed |
| 本机环境（实测） | Python 3.13.14 / numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1 / pytest 9.1.1 | 2026-10-07 | 本机 `python -c "import ..."` 实测输出 |

### 待验证项清单

| 项目 | 状态 | 建议动作 |
| --- | --- | --- |
| PyTorch 2.14.1 补丁线 | 待验证 | 发布前查 PyPI `torch` 最新 patch |
| PyTorch Geometric 2.9.0 | 待验证（滚动文档标注） | 以 PyPI 实际 tag 为准 |
| `llama-cpp-python` 绑定版本 | 待验证 | 决定 `[llm-llamacpp]` 的下界 |
| Docker Compose 上游版本与 Chocolatey 5.5.0 的对应关系 | 待验证 | 以 `docker compose version` 官方 release 为准 |
| pre-commit / mypy / ty / pytest-cov / hypothesis 版本 | 待验证 | `[dev]` 与 `[dev-strict]` 定版时统一检索 |
| SeaweedFS / K3s / kind / Podman / Renovate 版本 | 待验证 | 仅文档引用，可在交付文档时补全 |
| SPDX 3.0 与 REUSE 规范版本 | 待验证 | 合规门禁落地时确认 |
| actions/checkout 等官方 action 的最新 tag | 待验证 | CI 配置时以 SHA pin 固化 |
| `[serve-ray]` 在 Windows 上的可用性 | 待验证 | 建议标为 Linux-only extras |

---

## 免责声明

本文为技术选型研究观点，不构成投资建议。版本号与许可信息以各项目官方仓库与 PyPI 的实际状态为准；标注「待验证」的条目在落地前必须重新核实。
