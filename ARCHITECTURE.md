# WorldForge AI 系统架构设计文档

- 仓库: `worldforge-ai`
- 作者: 晨星 (CJX0712)
- 定位: 可插拔端到端 AI 平台 -- 数据接入 -> 嵌入 + 向量检索 (RAG) -> 推理服务 -> Agent 编排 -> HTTP 网关 -> 评估与可观测
- 本文档是工程实现的唯一编码依据, 接口签名以本文档为准, 实现不得擅自增删 Port 方法
- 时效声明: 文档中出现的生产后端 (FAISS / Qdrant / fastembed / llama-cpp / Ollama / OpenAI) 均为可选 extras, 版本与单价以各厂商官方渠道当日信息为准, 代码内禁止硬编码单价

---

## 0. 结论先行

| 项 | 结论 |
| --- | --- |
| 核心铁律 | `core/` + `ports/` + 默认 adapters 在裸 CPython 3.11+ 下零第三方依赖, `python scripts/verify.py` 必过 |
| 依赖方向 | 单向 `serving -> pipeline -> agents/retrieval -> adapters -> ports -> core`, 反向 import 一律禁止 |
| 可插拔机制 | 能力抽象为 `typing.Protocol`, 实现在 `adapters/`, 由 `pipeline.py` 按环境变量注入 |
| 默认实现 | 必须是真实现, 不是 stub: 精确余弦内存索引 / BLAKE2b 有符号散列嵌入 / Robertson-IDF 的 BM25 / IDF 加权词法重排器 / 抽取式 Mock LLM |
| 降级策略 | 生产后端 import 失败或连接失败时静默回退默认实现, 只 warn 一次, 并在 `/health.degraded` 暴露 |
| 确定性 | 固定种子 + `deterministic_view()` + 评估必须在全新管道上跑; 延迟单独统计, 不进确定性断言 |
| 交付门禁 | `scripts/verify.py` (零依赖冒烟) 与 `pytest` (全量) 双门禁, CI 在 GitHub Actions 上实跑, 本机不实跑 Docker |

---

## 1. 运行环境与硬约束

| 约束 | 值 | 对架构的影响 |
| --- | --- | --- |
| OS | Windows 11 沙箱 | 路径分隔符统一 `pathlib`; 禁止依赖 fork; 禁止 Unix-only 系统调用 |
| GPU | 无 | 默认栈全部 CPU; 生产向量库一律 CPU 版 (`faiss-cpu`) |
| Python | 3.13.14 (兼容下限 3.11) | 使用 `from __future__ import annotations` 保证 3.11 可解析新语法 |
| pip | 仅 `pip install --target <dir>` | 依赖装到 `_deps/`, 通过 `PYTHONPATH` 注入; `_deps/` 已 gitignore |
| 已装 | numpy 2.5.3, pytest 9.1.1 | numpy 只能作**可选加速**, 不得成为默认路径的硬依赖 |
| Docker/CI | 仅作交付物 | `Dockerfile` 与 `.github/workflows/ci.yml` 只写不跑 |
| 第三方库风险 | 高 (沙箱编译链缺失) | 默认路径零第三方; 任何 extras 必须在 `try: import ... except ImportError:` 中加载 |

---

## 2. 分层架构图

```
                    +----------------------------------------------+
                    |  客户端: curl / OpenAI SDK / scripts/smoke.py |
                    +----------------------------------------------+
                                       |  HTTP/1.1 + JSON
                                       v
+------------------------------------------------------------------------------+
| L6  serving/                                                                 |
|   /health   /v1/ingest   /v1/query   /v1/chat/completions                    |
|   /v1/evaluate   /v1/metrics   /v1/traces/{trace_id}                         |
|   实现: stdlib http.server (零依赖) ; 可选 uvicorn/FastAPI 作为 extras        |
+-------------------------------+----------------------------------------------+
                                | Pipeline 门面 (进程内函数调用, 不走网络)
                                v
+------------------------------------------------------------------------------+
| L5  pipeline.py   装配根: build_pipeline(env) -> Pipeline                     |
|   Config.from_env -> pick(...) 逐个选后端 -> 失败回退 -> 冻结对象图            |
+-------+---------------------------+---------------------------+--------------+
        |                           |                           |
        v                           v                           v
+--------------------+  +-------------------------+  +------------------------+
| L4  agents/        |  | L3  retrieval/          |  | L6b eval/              |
|  ReAct 循环        |  |  分块 -> BM25 -> Dense  |  |  指标/scorecard/门限   |
|  规则规划器        |  |  -> RRF 融合 -> 重排    |  | L6c observability/     |
|  工具注册与调用    |  |  -> 证据单行化渲染      |  |  trace/metrics/exporter|
|  多 Agent DAG      |  |                         |  |                        |
+--------------------+  +-------------------------+  +------------------------+
        |                           |
        |  (只依赖 Port, 不依赖具体库)
        v                           v
+------------------------------------------------------------------------------+
| L2  adapters/                                                                |
|   默认零依赖实现  : hashing / memory / bm25 / lexical / mock / recursive      |
|   生产实现(extras): faiss / qdrant / fastembed / elastic / llama-cpp /        |
|                     ollama / openai / cross-encoder / otlp / sqlite           |
+-------------------------------+----------------------------------------------+
                                | implements
                                v
+------------------------------------------------------------------------------+
| L1  ports/    10 个 typing.Protocol 契约 (LLM/Embedder/VectorStore/           |
|               LexicalIndex/Reranker/DocStore/Observer/Tool/Planner/Chunker)   |
+-------------------------------+----------------------------------------------+
                                | depends on (纯函数与数据结构)
                                v
+------------------------------------------------------------------------------+
| L0  core/     数学与文本算法内核: hash_embed / bm25 / rrf / rank_corr /       |
|               tokenizer / safe_eval / text / mathx / seeding                  |
|               零第三方依赖, 纯函数, 无 I/O, 无全局状态                        |
+------------------------------------------------------------------------------+
```

依赖规则 (CI 静态检查, `tests/test_layering.py`):

| 允许 | 禁止 |
| --- | --- |
| `serving` -> `pipeline`, `types`, `ports` | `core` import 任何上层 |
| `agents` -> `ports`, `types`, `errors` | `ports` import `adapters` |
| `retrieval` -> `core`, `ports`, `types` | `adapters` import `serving` / `agents` |
| `adapters` -> `core`, `ports`, `types` | 默认 adapters import 任何第三方库 |
| `core` -> 仅标准库 | `core` 内出现 `import numpy` (只能在 `core/accel.py` 可选分支) |

---

## 3. 每层职责

| 层 | 目录 | 职责 | 禁止事项 |
| --- | --- | --- | --- |
| L0 内核 | `core/` | 纯算法: 向量运算、散列嵌入、BM25、RRF、秩相关、分词、安全求值、文本规范化 | I/O、网络、全局状态、第三方依赖 |
| L1 契约 | `ports/` | 只定义 `typing.Protocol` 与配套数据类, 无实现代码 | 任何可执行逻辑 |
| L2 实现 | `adapters/` | 默认零依赖实现 + 生产实现 + `registry.py` 选择工厂 | 默认实现里出现 `pass`/`NotImplementedError`/假数据 |
| L3 检索 | `retrieval/` | 分块、稀疏/稠密双路召回、RRF 融合、重排编排、证据渲染 | 直接 import 具体向量库 |
| L4 智能体 | `agents/` | ReAct 循环、规则规划、工具注册与调用、多 Agent DAG、接地核验 | 直接 import 具体 LLM SDK |
| L5 装配 | `pipeline.py` | 读环境变量、选后端、注入依赖、暴露门面方法 | 业务逻辑 |
| L6 服务 | `serving/` | HTTP 网关、OpenAI 兼容协议、错误信封、请求校验 | 绕过 Pipeline 直连 adapters |
| L6b 评估 | `eval/` | 指标、scorecard、门限、基线记录 | 复用污染的单例管道 |
| L6c 可观测 | `observability/` | trace span、metrics 聚合、导出器 | 把墙钟时间写进业务对象 |

分层理由 (与任务书建议的差异与说明):

| 决策 | 理由 |
| --- | --- |
| 新增 `LexicalIndex` 端口 | 稀疏分支需要与稠密分支同等的替换能力 (Elastic/Pyserini), 若硬编码进 `retrieval/` 则违反"规则强制、框架可选" |
| 新增 `Chunker` 端口 | 分块策略是检索质量第一变量, 需可替换 (递归字符 / 语义分块 / Markdown 结构分块) |
| 新增 `agents/verify.py` | 接地核验在运行时与评估时共用一套剥离与打分逻辑, 放在 `agents/` 保证线上与离线口径一致, `eval/` 只做聚合 |
| `serving/` 用 stdlib `http.server` | 沙箱约束; 同时保留 `app.py` 中的 WSGI/ASGI 适配点, extras 装了 uvicorn 才走 |
| `sqlite` 归为"零依赖但非默认" | `sqlite3` 是标准库, 但默认仍用 `MemoryDocStore` 以保证评估确定性 |

---

## 4. Ports 接口契约

所有 Port 位于 `src/worldforge/ports/`, 均为 `typing.Protocol` (结构化子类型, 实现方无需继承)。
实现必须满足: 相同输入 -> 相同输出 (确定性); 不吞异常 (统一转 `errors.py` 中的类型); 不在返回对象里塞墙钟时间。

### 4.0 共享数据类型 (`src/worldforge/types.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

Role = str  # "system" | "user" | "assistant" | "tool"


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str            # f"{doc_id}#{ordinal:04d}", 全局唯一且稳定
    doc_id: str
    ordinal: int             # 文档内序号, 从 0 开始
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float             # 越大越相关; 融合/重排后必须仍单调递增可比
    source: str              # "dense" | "sparse" | "fused" | "rerank"
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    name: str | None = None  # 工具回灌时填工具名


@dataclass(frozen=True)
class Completion:
    text: str
    model_id: str
    finish_reason: str        # "stop" | "length" | "tool_calls" | "error"
    usage: Mapping[str, int] = field(default_factory=dict)  # prompt_tokens/completion_tokens


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    doc_id: str
    text: str                 # 已单行化
    score: float


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: Mapping[str, Any]   # JSON Schema (draft 2020-12 子集)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    error: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    step_id: str
    kind: str                 # "retrieve" | "tool" | "llm" | "answer"
    payload: Mapping[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    goal: str
    steps: tuple[Step, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolContext:
    trace_id: str
    seed: int
    observer: Any             # ports.Observer, 避免循环 import 用 Any
    budget: Mapping[str, int] = field(default_factory=dict)
```

### 4.1 异常体系 (`src/worldforge/errors.py`)

```python
class WorldforgeError(Exception): ...
class ConfigError(WorldforgeError): ...
class BackendUnavailable(WorldforgeError): ...      # 生产后端 import/连接失败
class DimensionMismatch(WorldforgeError): ...       # 嵌入维度与索引指纹不符
class VectorStoreError(WorldforgeError): ...
class DocNotFound(WorldforgeError): ...
class IngestError(WorldforgeError): ...
class ToolError(WorldforgeError): ...               # safe_eval 归一化的统一出口
class SafeEvalError(ToolError): ...
class LLMError(WorldforgeError): ...
class PlanError(WorldforgeError): ...
class EvalGateError(WorldforgeError): ...           # scorecard 未过门限
```

### 4.2 LLM

```python
# ports/llm.py
from __future__ import annotations
from typing import Iterator, Mapping, Protocol, Sequence, runtime_checkable
from worldforge.types import Completion, Message

@runtime_checkable
class LLM(Protocol):
    @property
    def model_id(self) -> str:
        """稳定标识, 写入 trace 与响应体; 不含版本号中的时间戳。"""
        ...

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        """确定性采样: temperature=0 且 seed 固定时, 同输入必须逐字节一致。
        Raises:
            LLMError: 后端不可用、超时、返回体无法解析。
        """
        ...

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        """messages 为空或 role 非法时抛 ValueError; 其余同 complete。"""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Iterator[str]:
        """逐块产出文本增量。不保证跨后端字节一致, 禁止进确定性断言。"""
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/llm_mock.py` | `ExtractiveMockLLM`: 从证据里按 IDF 权重抽取得分最高的原句拼接, 温度/seed 被忽略但接口兼容, 输出必带 `[id#n]` 引用标记 |
| 生产 | `adapters/llm_openai.py` / `llm_ollama.py` / `llm_llamacpp.py` | 缺 SDK 时 `BackendUnavailable` -> 回退 mock |

### 4.3 Embedder

```python
# ports/embedder.py
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable

@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int:
        """向量维度, 一经写入索引即成为指纹, 变更必须触发重建。"""
        ...

    @property
    def model_id(self) -> str:
        """写入索引指纹; 不同 model_id 的向量禁止混存。"""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """texts 为空返回 []; 返回向量必须已 L2 归一化。
        相同文本 -> 位级相同的 float。
        Raises:
            ValueError: 单个文本长度超过实现上限。
            LLMError: 远端嵌入服务失败。
        """
        ...

    def embed_one(self, text: str) -> list[float]:
        """等价于 embed([text])[0]。"""
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/embedder_hashing.py` | `HashingEmbedder`: BLAKE2b 有符号散列嵌入, `dim` 默认 256, 中英双语 token 加权求和后 L2 归一化 |
| 生产 | `adapters/embedder_fastembed.py` / `embedder_openai.py` | extras |

### 4.4 VectorStore

```python
# ports/vectorstore.py
from __future__ import annotations
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from worldforge.types import Hit

@runtime_checkable
class VectorStore(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def fingerprint(self) -> str:
        """f"{model_id}:{dim}", upsert 时校验, 不符抛 DimensionMismatch。"""
        ...

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metas: Sequence[Mapping[str, Any]],
    ) -> int:
        """按 id 覆盖写入。三个入参长度必须相等, 否则 ValueError。
        返回写入条数。
        Raises:
            DimensionMismatch: len(v) != self.dim 或 fingerprint 不匹配。
        """
        ...

    def delete(self, ids: Sequence[str]) -> int:
        """删除指定 chunk_id, 返回实际删除条数 (不存在的不报错)。"""
        ...

    def drop_document(self, doc_id: str) -> int:
        """按 doc_id 前缀删除该文档全部 chunk, 返回删除条数。
        ingest 必须在写入前调用本方法 (见失败模式 FM-02)。
        """
        ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[Hit]:
        """精确或近似余弦检索; where 为等值过滤 (AND 语义)。
        返回按 (-score, chunk_id) 稳定排序, score 降序。
        Raises:
            DimensionMismatch: 查询向量维度不符。
            VectorStoreError: 索引损坏或后端不可达。
        """
        ...

    def count(self) -> int: ...

    def clear(self) -> None:
        """清空索引与指纹, 仅供测试与评估新建管道使用。"""
        ...

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None:
        """load 时校验 fingerprint, 不符抛 DimensionMismatch。"""
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/vectorstore_memory.py` | `InMemoryExactIndex`: 暴力精确余弦, 结果等于定义本身; numpy 存在时走 `core/accel.py` 加速且必须通过 atol=1e-9 对齐测试 |
| 生产 | `adapters/vectorstore_faiss.py` / `vectorstore_qdrant.py` | extras |

### 4.5 LexicalIndex (稀疏分支)

```python
# ports/lexical.py
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable
from worldforge.types import Chunk, Hit

@runtime_checkable
class LexicalIndex(Protocol):
    def index(self, chunks: Sequence[Chunk]) -> int:
        """建立/增量更新倒排表, 返回新增条数。"""
        ...

    def drop_document(self, doc_id: str) -> int:
        """移除该文档全部 posting, 返回删除条数。"""
        ...

    def search(self, query: str, top_k: int) -> list[Hit]:
        """返回 source="sparse" 的 Hit, 按 (-score, chunk_id) 稳定排序。
        未命中任何词时返回 [] (不得返回全 0 分数的噪声结果)。
        """
        ...

    def stats(self) -> dict[str, float]:
        """{"docs": N, "avgdl": float, "vocab": int}, 用于 IDF 与调试。"""
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/lexical_bm25.py` | 自写 Robertson-IDF BM25: `idf = ln(1 + (N - n + 0.5) / (n + 0.5))`, 恒非负 (FM-01) |
| 生产 | `adapters/lexical_elastic.py` | extras |

### 4.6 Reranker

```python
# ports/reranker.py
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable
from worldforge.types import Hit

@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Hit], top_k: int) -> list[Hit]:
        """输入候选重排, 返回 top_k 条 source="rerank" 的 Hit。
        candidates 为空返回 []; top_k <= 0 抛 ValueError。
        必须保持稳定排序 (-score, chunk_id)。
        """
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/reranker_lexical.py` | IDF 加权词法重排: 查询词 IDF 加权覆盖率 + 覆盖率平滑 + 位置先验 |
| 生产 | `adapters/reranker_crossencoder.py` | extras (bge-reranker / Cohere) |

### 4.7 DocStore

```python
# ports/docstore.py
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable
from worldforge.types import Chunk, Document

@runtime_checkable
class DocStore(Protocol):
    def put_document(self, doc: Document) -> None:
        """覆盖写入文档元数据; 不负责分块 (分块由 Chunker 完成)。"""
        ...

    def get_document(self, doc_id: str) -> Document | None:
        """不存在返回 None, 不抛异常。"""
        ...

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[str]:
        """返回 doc_id 列表, 按字典序稳定排序。"""
        ...

    def put_chunks(self, chunks: Sequence[Chunk]) -> int:
        """按 chunk_id 覆盖写入, 返回条数。"""
        ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """缺失的 id 静默跳过; 返回顺序与入参一致。
        评估与提示词渲染只走本方法, 禁止读任何 last_* 缓存 (FM-08)。
        """
        ...

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        """按 ordinal 升序返回。"""
        ...

    def drop_document(self, doc_id: str) -> int:
        """删除文档及其全部 chunk, 返回删除条数。"""
        ...

    def count_chunks(self) -> int: ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/docstore_memory.py` | `MemoryDocStore`: 进程内 dict, 确定性最佳 |
| 零依赖可选 | `adapters/docstore_sqlite.py` | `sqlite3` 标准库, 持久化但评估默认不用 |
| 生产 | `adapters/docstore_postgres.py` | extras |

### 4.8 Observer

```python
# ports/observer.py
from __future__ import annotations
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable
from contextlib import contextmanager

@runtime_checkable
class Observer(Protocol):
    def span(self, name: str, **attrs: Any) -> Any:
        """返回上下文管理器; 退出时自动记录 wall_ms (仅此处允许墙钟)。
        用法: with obs.span("retrieve.dense", top_k=20): ...
        """
        ...

    def emit(self, event: str, **fields: Any) -> None:
        """记录一条结构化事件; fields 中禁止放墙钟时间。"""
        ...

    def metric(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        """累加/观测一个指标 (counter / gauge / histogram 由 name 前缀决定)。"""
        ...

    def trace_id(self) -> str:
        """当前 trace_id; 无活动 span 时返回 "unknown"。"""
        ...

    def flush(self) -> dict[str, Any]:
        """导出并清空缓冲, 返回 {"spans": [...], "metrics": {...}}。"""
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/observer_memory.py` | `MemoryObserver`: 环形缓冲 (上限 `WF_TRACE_BUFFER`, 默认 1000), `/v1/metrics` 直接读 |
| 生产 | `adapters/observer_otlp.py` | extras (OpenTelemetry / Prometheus) |

### 4.9 Tool

```python
# ports/tool.py
from __future__ import annotations
from typing import Protocol, Mapping, runtime_checkable
from worldforge.types import ToolContext, ToolResult, ToolSpec

@runtime_checkable
class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec:
        """name 必须匹配 ^[a-z][a-z0-9_]{0,31}$; description 用于路由与提示词。"""
        ...

    def invoke(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        """执行工具。永不向上抛非受控异常:
        所有内部异常归一化为 ToolResult(ok=False, error=str(exc)) (FM-04)。
        禁止读/写 ctx 之外的全局状态。
        """
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/builtin_tools.py` -> `agents/builtin_tools.py` | `calculator` (safe_eval) / `retriever` / `dictionary` / `echo` |
| 生产 | `adapters/tool_mcp.py` | extras (MCP 工具服务器) |

### 4.10 Planner

```python
# ports/planner.py
from __future__ import annotations
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from worldforge.types import Plan, Step, ToolSpec

@runtime_checkable
class Planner(Protocol):
    def plan(self, goal: str, tools: Sequence[ToolSpec], ctx: Mapping[str, Any]) -> Plan:
        """产出 DAG。step_id 必须唯一; depends_on 引用不存在的 step_id 抛 PlanError。
        不允许环, 装配时由 agents/dag.py 校验。
        """
        ...

    def replan(
        self,
        plan: Plan,
        done: Sequence[Step],
        observation: str,
        ctx: Mapping[str, Any],
    ) -> Plan:
        """ReAct 单步再规划, 返回新 Plan (不可变, 不得原地修改入参)。"""
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `agents/planner_rule.py` | `RuleBasedPlanner`: 意图规则表 (算术/查询/多跳), 零 LLM 依赖 |
| 备选 | `agents/react.py` | `ReActPlanner`: 用 LLM 产生 Thought/Action/Observation 循环 |
| 生产 | `agents/planner_llm.py` | `LLMPlanner`: 需要真 LLM, 缺失时回退 `RuleBasedPlanner` |

### 4.11 Chunker

```python
# ports/chunker.py
from __future__ import annotations
from typing import Protocol, runtime_checkable
from worldforge.types import Chunk, Document

@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]:
        """返回非空则 ordinal 从 0 连续递增, chunk_id = f"{doc.doc_id}#{ordinal:04d}"。
        空文本返回 []。重叠窗口大小由实现自持, 但必须确定性。
        """
        ...
```

| 实现 | 文件 | 说明 |
| --- | --- | --- |
| 默认 | `adapters/chunker_recursive.py` | `RecursiveCharChunker`: 按段落 -> 句子 -> 字符递归切分, `chunk_size=512`, `overlap=64` |
| 生产 | `adapters/chunker_semantic.py` | extras (依赖嵌入相似度) |

---

## 5. 可插拔矩阵

| 能力 | 默认零依赖实现 | 生产实现 (extras) | 环境变量开关 | 缺失时行为 |
| --- | --- | --- | --- | --- |
| Embedder | `HashingEmbedder` (BLAKE2b 有符号散列, dim=256) | `FastEmbedEmbedder` / `OpenAIEmbedder` | `WF_EMBEDDER=hashing|fastembed|openai` | 静默回退 `hashing`, warn 一次, `degraded+=embedder` |
| VectorStore | `InMemoryExactIndex` (精确余弦, numpy 可选加速) | `FaissStore` / `QdrantStore` | `WF_VECTORSTORE=memory|faiss|qdrant` | 回退 `memory` |
| LexicalIndex | `BM25Index` (Robertson IDF, 自写) | `ElasticLexical` | `WF_LEXICAL=bm25|elastic` | 回退 `bm25` |
| Reranker | `LexicalReranker` (IDF 加权词法) | `CrossEncoderReranker` / Cohere | `WF_RERANKER=lexical|crossencoder|cohere` | 回退 `lexical` |
| LLM | `ExtractiveMockLLM` (抽取式, 真实现) | `OpenAILLM` / `OllamaLLM` / `LlamaCppLLM` | `WF_LLM=mock|openai|ollama|llamacpp` | 回退 `mock` |
| DocStore | `MemoryDocStore` | `SqliteDocStore` (stdlib) / `PostgresDocStore` | `WF_DOCSTORE=memory|sqlite|postgres` | 回退 `memory` |
| Observer | `MemoryObserver` (环形缓冲) | `OtlpObserver` / Prometheus exporter | `WF_OBSERVER=memory|otlp` | 回退 `memory` |
| Tool 源 | `builtin` (calculator/retriever/dictionary/echo) | MCP 工具服务器 | `WF_TOOLS=builtin|mcp` | 回退 `builtin` |
| Planner | `RuleBasedPlanner` | `ReActPlanner` / `LLMPlanner` | `WF_PLANNER=rule|react|llm` | `llm` 缺 LLM 时回退 `rule` |
| Chunker | `RecursiveCharChunker` | `SemanticChunker` | `WF_CHUNKER=recursive|semantic` | 回退 `recursive` |
| HTTP 服务器 | `stdlib http.server` (ThreadingHTTPServer) | `uvicorn` + FastAPI | `WF_SERVER=stdlib|uvicorn` | 回退 `stdlib` |

完整环境变量表 (`config.py` 解析, 未知前缀 `WF_` 的变量报 `ConfigError`):

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `WF_SEED` | `1337` | 全局随机种子 |
| `WF_EMBED_DIM` | `256` | 仅对 hashing 生效 |
| `WF_CHUNK_SIZE` | `512` | 分块字符数 |
| `WF_CHUNK_OVERLAP` | `64` | 分块重叠 |
| `WF_TOP_K` | `20` | 单路召回条数 |
| `WF_RERANK_TOP_K` | `5` | 重排后送入 LLM 的条数 |
| `WF_FUSION` | `rrf` | `rrf|weighted`, RRF `k=60` |
| `WF_RRF_K` | `60` | RRF 平滑常数 |
| `WF_HYBRID` | `on` | `on|off`, off 时只走 dense |
| `WF_VERIFY` | `on` | 接地核验开关 |
| `WF_BILINGUAL` | `on` | 双语词典扩展, **仅作用于稀疏分支** |
| `WF_LOG_LEVEL` | `INFO` | |
| `WF_TRACE_BUFFER` | `1000` | MemoryObserver 环形缓冲 |
| `WF_HOST` / `WF_PORT` | `127.0.0.1` / `8000` | |
| `WF_STRICT` | `off` | `on` 时禁止静默回退, 直接抛 `BackendUnavailable` |

---

## 6. 依赖注入与装配

### 6.1 三条装配路径

| 路径 | 用途 | 入口 |
| --- | --- | --- |
| 生产 | `build_pipeline(env=None)` 读 `os.environ` | `pipeline.py` |
| 测试 | `Pipeline.from_parts(**overrides)` 显式注入 fake | 单测 |
| 评估 | 每个 case 调 `build_pipeline(env)` 建**全新**管道 | `eval/runner.py` |

### 6.2 `build_pipeline` 伪代码

```python
# src/worldforge/pipeline.py
from __future__ import annotations
from typing import Any, Mapping
from worldforge import config, ports
from worldforge.adapters import registry
from worldforge.retrieval import hybrid as hybrid_mod
from worldforge.agents import react, planner_rule, planner_llm, tools as tools_mod
from worldforge.core import seeding


def build_pipeline(env: Mapping[str, str] | None = None, **overrides: Any) -> "Pipeline":
    cfg = config.Config.from_env(env if env is not None else os.environ)
    degraded: list[str] = []
    seeding.seed_everything(cfg.seed)          # random / hash seed / dict order 固定

    def pick(kind: str, table: Mapping[str, Any], default_key: str, tag: str):
        """选后端: 命中生产实现但 import 失败 -> warn 一次 -> 回退 default_key。"""
        factory = table.get(kind)
        if factory is None:
            raise config.ConfigError(f"unknown backend: {kind}")
        if kind == default_key:
            return factory(cfg)
        try:
            return factory(cfg)
        except (ImportError, ports.BackendUnavailable) as exc:
            if cfg.strict:
                raise
            warn_once(f"[{tag}] {kind} unavailable ({exc}); fallback -> {default_key}")
            degraded.append(tag)
            return table[default_key](cfg)

    observer   = overrides.pop("observer",   None) or pick(cfg.observer,   registry.OBSERVERS,   "memory", "observer")
    embedder   = overrides.pop("embedder",   None) or pick(cfg.embedder,   registry.EMBEDDERS,   "hashing","embedder")
    vectorstore= overrides.pop("vectorstore",None) or pick(cfg.vectorstore,registry.VECTORSTORES,"memory", "vectorstore")
    lexical    = overrides.pop("lexical",    None) or pick(cfg.lexical,    registry.LEXICALS,    "bm25",   "lexical")
    reranker   = overrides.pop("reranker",   None) or pick(cfg.reranker,   registry.RERANKERS,   "lexical","reranker")
    docstore   = overrides.pop("docstore",   None) or pick(cfg.docstore,   registry.DOCSTORES,   "memory", "docstore")
    llm        = overrides.pop("llm",        None) or pick(cfg.llm,        registry.LLMS,        "mock",   "llm")
    chunker    = overrides.pop("chunker",    None) or pick(cfg.chunker,    registry.CHUNKERS,    "recursive","chunker")

    # Planner 需 LLM, 缺失时降级为规则规划器
    planner = overrides.pop("planner", None)
    if planner is None:
        planner = {"rule": planner_rule.RuleBasedPlanner,
                   "react": react.ReActPlanner,
                   "llm": planner_llm.LLMPlanner}[cfg.planner](llm=llm, cfg=cfg)
        if cfg.planner == "llm" and isinstance(llm, registry.MOCK_LLM_TYPE):
            warn_once("[planner] llm backend is mock; downgrade -> rule")
            degraded.append("planner")
            planner = planner_rule.RuleBasedPlanner(llm=llm, cfg=cfg)

    tool_registry = overrides.pop("tools", None) or tools_mod.build_registry(cfg, docstore=docstore,
                                                                            vectorstore=vectorstore,
                                                                            embedder=embedder,
                                                                            lexical=lexical)
    retriever = hybrid_mod.HybridRetriever(embedder=embedder, vectorstore=vectorstore,
                                           lexical=lexical, reranker=reranker,
                                           docstore=docstore, cfg=cfg, observer=observer)
    return Pipeline.from_parts(cfg=cfg, embedder=embedder, vectorstore=vectorstore,
                               lexical=lexical, reranker=reranker, docstore=docstore,
                               llm=llm, planner=planner, chunker=chunker,
                               tools=tool_registry, retriever=retriever,
                               observer=observer, degraded=tuple(degraded), **overrides)
```

### 6.3 `Pipeline` 门面

```python
class Pipeline:
    @classmethod
    def from_parts(cls, **parts: Any) -> "Pipeline":
        """唯一构造入口, 字段缺失抛 ConfigError; 对象构造后冻结 (__slots__ + frozen)。"""
        ...

    def ingest(self, docs: Sequence[Document]) -> dict[str, Any]:
        """先 drop_document 再写入 (FM-02); 返回 {"docs": n, "chunks": m, "skipped": k}。"""
        ...

    def query(self, query: str, top_k: int | None = None,
              mode: str = "hybrid", where: Mapping[str, Any] | None = None) -> list[Hit]: ...

    def answer(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """检索 -> 重排 -> 生成 -> 核验; 返回 {"answer","evidence","hits","verification","trace_id"}。"""
        ...

    def run_agent(self, goal: str, max_steps: int = 8) -> dict[str, Any]:
        """规划 -> DAG 执行 -> 汇总; 返回 {"answer","steps","tool_calls","trace_id"}。"""
        ...

    def evaluate(self, suite: str = "default") -> dict[str, Any]:
        """在调用方新建的管道上跑; 返回 scorecard dict (不含墙钟字段)。"""
        ...

    def health(self) -> dict[str, Any]:
        """{"status","version","backends": {...},"degraded": [...]}。"""
        ...
```

### 6.4 单测 fake 注入

| 规则 | 说明 |
| --- | --- |
| 结构化子类型 | 所有 Port 用 `typing.Protocol`, fake 只需方法签名一致, 无需继承, 无需注册 |
| 显式注入 | 测试一律走 `Pipeline.from_parts(...)` 或 `build_pipeline(env, llm=FakeLLM(...))`, 禁止 monkeypatch 模块级单例 |
| 禁止单例 | `pipeline.py` 不保存全局 `_PIPELINE`, 每次调用产生新对象 |
| Fake 清单 | `tests/fakes.py`: `FakeLLM` (脚本文本映射) / `FakeEmbedder` (确定性 one-hot) / `FakeVectorStore` / `RecordingObserver` / `FakeTool` (固定返回) / `FakePlanner` |
| 断言方式 | 对象比较一律经 `core.seeding.deterministic_view()`, 禁止直接 `assert obj == obj2` |

```python
# tests/fakes.py 片段
class FakeLLM:
    model_id = "fake-llm-0"
    def __init__(self, script: Mapping[str, str] | None = None, default: str = "FINAL: ok"):
        self._script, self._default = dict(script or {}), default
        self.calls: list[str] = []
    def complete(self, prompt, *, temperature=0.0, max_tokens=512, stop=None, seed=None):
        self.calls.append(prompt)
        for key, val in self._script.items():
            if key in prompt:
                return Completion(text=val, model_id=self.model_id, finish_reason="stop",
                                  usage={"prompt_tokens": len(prompt), "completion_tokens": len(val)})
        return Completion(text=self._default, model_id=self.model_id, finish_reason="stop", usage={})
    def chat(self, messages, **kw):
        return self.complete("\n".join(m.content for m in messages), **kw)
    def stream(self, messages, **kw):
        yield self.chat(messages, **kw).text
```

---

## 7. 端到端请求数据流

一次「提问 -> 检索 -> 重排 -> 生成 -> 核验 -> 响应 -> 埋点」的完整链路。

```
用户提问
  -> [1]  HTTP 网关校验
  -> [2]  归一化 + 意图判定
  -> [3]  查询嵌入
  -> [4]  稠密召回
  -> [5]  稀疏召回 (BM25, 含双语扩展)
  -> [6]  RRF 融合
  -> [7]  重排
  -> [8]  证据取回 + 单行化渲染
  -> [9]  提示词组装
  -> [10] LLM 生成
  -> [11] 接地/忠实度核验
  -> [12] 响应组装 (OpenAI 兼容)
  -> [13] 埋点落盘
  -> [14] 返回客户端
```

| # | 步骤 | 组件 | 输入 -> 输出 | 可观测点 |
| --- | --- | --- | --- | --- |
| 1 | 请求校验 | `serving/routes_query.py` | JSON -> `QueryRequest` | span `http.query`; metric `http_requests_total`; 失败 metric `http_errors_total{code}` |
| 2 | 归一化 + 意图判定 | `core/tokenizer.py`, `agents/planner_rule.py` | query -> normalized query + intent | span `plan.intent`; emit `intent.detected{intent}` |
| 3 | 查询嵌入 | `Embedder.embed_one` | str -> `list[float]` (dim=256) | span `embed.query`; metric `embed_calls_total`; 此步结果进确定性断言 |
| 4 | 稠密召回 | `VectorStore.search` | vector, top_k -> `list[Hit]` (source=dense) | span `retrieve.dense{top_k}`; metric `dense_hits`; 确定性 |
| 5 | 稀疏召回 | `LexicalIndex.search` | query, top_k -> `list[Hit]` (source=sparse) | span `retrieve.sparse{top_k}`; metric `sparse_hits`; emit `bm25.stats` |
| 6 | RRF 融合 | `retrieval/fusion.py` | 两路 Hit -> `list[Hit]` (source=fused) | span `fusion.rrf`; metric `fusion_overlap` (两路交集大小) |
| 7 | 重排 | `Reranker.rerank` | fused -> top N (source=rerank) | span `rerank`; metric `rerank_delta` (重排前后 rank 变化, 用 `core/rank_corr.py`) |
| 8 | 证据渲染 | `retrieval/evidence.py` + `core/text.py` | chunk_ids -> 单行化Evidence 列表 | span `evidence.render`; emit `evidence.lines` (必须等于块数, FM-03) |
| 9 | 提示词组装 | `agents/react.py` / `retrieval/evidence.py` | query + evidence -> prompt | span `prompt.build`; emit `prompt.tokens_est` |
| 10 | 生成 | `LLM.chat` | messages -> `Completion` | span `llm.generate`; metric `llm_tokens_in/out`, `llm_latency_ms` (**仅延迟, 不进确定性断言**) |
| 11 | 核验 | `agents/verify.py` | answer + evidence -> `{"grounding","faithfulness","unsupported"}` | span `verify.grounding`; metric `grounding_score`; 剥离引用标记后计算 (FM-10) |
| 12 | 响应组装 | `serving/openai_compat.py` | 内部结果 -> OpenAI 兼容 JSON | span `response.build`; emit `response.chars` |
| 13 | 埋点落盘 | `Observer.flush` | spans + metrics -> 缓冲 | span `obs.flush`; 导出 JSON 供 `/v1/metrics` 与 `/v1/traces/{id}` |
| 14 | 返回 | `serving/app.py` | HTTP 200 + JSON | metric `http_latency_ms`, `http_requests_total{code=200}` |

Agent 模式分支 (`/v1/chat/completions` 且 `WF_PLANNER != "rule"`):

| # | 步骤 | 说明 | 可观测点 |
| --- | --- | --- | --- |
| A1 | 规划 | `Planner.plan` -> DAG | span `plan.build`; emit `plan.steps` |
| A2 | DAG 调度 | `agents/dag.py` 拓扑排序, 环检测失败抛 `PlanError` | span `dag.schedule` |
| A3 | 工具调用 | `Tool.invoke` -> `ToolResult`, 异常归一为 `ok=False` | span `tool.invoke{name}`; metric `tool_calls_total{name,ok}`; emit `tool.route` |
| A4 | 回灌观察 | Observation 追加到 messages, 继续 ReAct | span `react.step{index}` |
| A5 | 终止判定 | `max_steps` 用尽或出现 Final 行; 兜底排除 `action:`/`thought:`/`observation:` 前缀 (FM-07) | emit `react.termination{reason}` |
| A6 | 汇总 | 最终答案 + 工具调用轨迹 | span `agent.summarize` |

---

## 8. HTTP API 设计

通用约定:

| 项 | 约定 |
| --- | --- |
| 前缀 | `/v1` |
| Content-Type | `application/json; charset=utf-8` |
| 错误信封 | `{"error": {"code": "<snake_case>", "message": "...", "trace_id": "..."}}` |
| 幂等 | `POST /v1/ingest` 幂等 (同 `doc_id` 覆盖, 内部先 drop) |
| 认证 | 默认无; `WF_API_KEY` 设置时校验 `Authorization: Bearer` |

| # | 方法 | 路径 | 请求体 | 响应体 | 错误码 |
| --- | --- | --- | --- | --- | --- |
| 1 | GET | `/health` | - | `{"status":"ok|degraded","version":"0.1.0","backends":{"embedder":"hashing","vectorstore":"memory","llm":"mock","lexical":"bm25","reranker":"lexical","docstore":"memory","planner":"rule"},"degraded":[],"index":{"docs":0,"chunks":0}}` | 200 / 500 |
| 2 | POST | `/v1/ingest` | `{"documents":[{"doc_id":"d1","text":"...","meta":{}}],"chunk":true}` | `{"docs":1,"chunks":7,"skipped":0,"trace_id":"..."}` | 400 参数缺失 / 413 超 `WF_MAX_DOCS` / 422 schema 不符 / 500 |
| 3 | DELETE | `/v1/documents/{doc_id}` | - | `{"doc_id":"d1","deleted_chunks":7}` | 404 无此文档 / 500 |
| 4 | GET | `/v1/documents` | query `?limit=100&offset=0` | `{"doc_ids":["d1"],"total":1}` | 400 limit/offset 非法 |
| 5 | POST | `/v1/query` | `{"query":"...","top_k":20,"mode":"hybrid|dense|sparse","where":{"lang":"zh"}}` | `{"hits":[{"chunk_id":"d1#0003","doc_id":"d1","score":0.83,"source":"rerank","text":"...","meta":{}}],"trace_id":"..."}` | 400 / 422 / 500 |
| 6 | POST | `/v1/chat/completions` | OpenAI 兼容: `{"model":"worldforge-1","messages":[{"role":"user","content":"..."}],"temperature":0,"stream":false,"worldforge":{"mode":"rag|auto","top_k":5}}` | `{"id":"chatcmpl-...","object":"chat.completion","created":<int>,"model":"worldforge-1","choices":[{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],"usage":{"prompt_tokens":N,"completion_tokens":M,"total_tokens":N+M},"worldforge":{"evidence":[...],"verification":{...},"trace_id":"..."}}` | 400 / 422 / 429 限流 / 500 / 503 后端全降级 |
| 7 | POST | `/v1/evaluate` | `{"suite":"default|retrieval|agent|rag","cases":null,"record_baseline":false}` | `{"suite":"default","passed":true,"metrics":{...},"gates":[{"metric":"grounding","value":0.87,"threshold":0.84,"ok":true}],"trace_id":"..."}` | 400 未知 suite / 500 / 注: 门限失败时**仍返回 200**, `passed=false`; 仅进程内 `EvalGateError` 抛给 CLI |
| 8 | GET | `/v1/metrics` | query `?format=json|prom` | JSON: `{"counters":{...},"gauges":{...},"histograms":{"http_latency_ms":{"p50":..,"p95":..,"n":..}}}` / Prometheus 文本 | 500 |
| 9 | GET | `/v1/traces/{trace_id}` | - | `{"trace_id":"...","spans":[{"name":"...","attrs":{},"order":0}],"events":[]}` (墙钟字段仅在 `?wall=1` 时返回) | 404 trace 不存在 |
| 10 | POST | `/v1/admin/reset` | `{"confirm":true}` | `{"cleared":{"docs":1,"chunks":7}}` | 400 未确认 / 403 `WF_ALLOW_RESET=off` |

流式: `POST /v1/chat/completions` 带 `"stream": true` 时返回 `text/event-stream`, 每帧 `data: {"choices":[{"delta":{"content":"..."}}]}`, 结束帧 `data: [DONE]`。流式输出**不进**确定性断言。

---

## 9. 评估方案

### 9.1 指标清单

| 指标 | 定义 | 计算位置 | 门限类型 |
| --- | --- | --- | --- |
| `doc_hit_rate@k` | 检索结果中含至少一个黄金 chunk 的查询占比 | `eval/metrics.py` | 硬 |
| `recall@k` | 命中黄金 chunk 数 / 黄金 chunk 总数 (按查询平均) | `eval/metrics.py` | 硬 |
| `mrr` | 首个相关 chunk 秩的倒数平均 | `eval/metrics.py` | 硬 |
| `ndcg@k` | 折扣累计增益 (可选) | `eval/metrics.py` | 软 |
| `grounding` | 答案句子中可追溯到证据的比例 (核验前剥离引用标记) | `eval/metrics.py` + `agents/verify.py` | 硬 |
| `faithfulness` | 答案内容 token 出现在证据中的 IDF 加权占比 | `eval/metrics.py` | 硬 |
| `tool_route_accuracy` | 工具选择正确的比例 | `eval/suites/agent_suite.py` | 硬 |
| `tool_call_success` | 工具执行 `ok=True` 的比例 | `eval/suites/agent_suite.py` | 硬 |
| `rank_corr` | 重排前后序与黄金序的 Spearman 相关 (常量输入返回 0.0) | `core/rank_corr.py` | 软 |
| `latency_p50_ms` / `latency_p95_ms` | 端到端延迟分位 | `observability/metrics.py` | 软, 且**不进确定性断言** |

### 9.2 门限制定流程 (贴基线, 真回归必挂, 抖动不挂)

| 步骤 | 动作 |
| --- | --- |
| S1 | 首次执行 `python scripts/record_baseline.py --runs 5`, 得到每个指标的均值 `mu` 与样本标准差 `sigma`, 写入 `src/worldforge/eval/baseline.json` |
| S2 | 硬门限 `threshold = max(floor, mu - max(0.02, 3 * sigma))`, `floor` 为人工绝对下限 (见下表), 防止基线本身偏低时门限退化 |
| S3 | 软告警: 相对基线下降 `> 2%` 即 warn, 不挂 CI |
| S4 | 抖动型指标 (延迟) 只走软告警, 门限一律 `-1` 表示不校验 |
| S5 | 基线文件随代码入库; 有意的行为变更必须同步改 `baseline.json` 并在 PR 说明原因 |

门限表模板 (数值为占位, 由 S1 实测填入, 实现不得直接抄):

| 指标 | floor (绝对下限) | baseline (实测填入) | threshold = max(floor, mu - max(0.02, 3sigma)) | 类型 |
| --- | --- | --- | --- | --- |
| `doc_hit_rate@5` | 0.80 | `BASELINE.doc_hit_rate@5` | 计算 | 硬 |
| `recall@10` | 0.70 | `BASELINE.recall@10` | 计算 | 硬 |
| `mrr` | 0.60 | `BASELINE.mrr` | 计算 | 硬 |
| `grounding` | 0.60 | `BASELINE.grounding` | 计算 | 硬 |
| `faithfulness` | 0.55 | `BASELINE.faithfulness` | 计算 | 硬 |
| `tool_route_accuracy` | 0.85 | `BASELINE.tool_route_accuracy` | 计算 | 硬 |
| `tool_call_success` | 0.90 | `BASELINE.tool_call_success` | 计算 | 硬 |
| `rank_corr` | -1 | `BASELINE.rank_corr` | 计算 | 软 |
| `latency_p95_ms` | -1 | `BASELINE.latency_p95_ms` | 不校验 | 软 |

### 9.3 评估执行的强制约束

| 约束 | 原因 |
| --- | --- |
| 每个 case 新建管道 `build_pipeline(env)` | 复用管道会残留索引/缓存, 第二次跑分数漂移 (FM-18) |
| 评估器显式捕获本次返回值, 禁止读 `last_*` 字段 | 读到上一次结果导致指标系统性偏低 (FM-08) |
| 结果比较走 `deterministic_view()` | 排除墙钟字段 |
| 数据集固定顺序, 不打乱, 不并行 | 并行会引入非确定性 |
| 黄金集随仓库入库, 版本化 | 指标可比的前提 |

```python
# eval/runner.py 骨架
def run_suite(suite: str, env: Mapping[str, str]) -> dict[str, Any]:
    cfg = config.Config.from_env(env)
    rows, perf = [], []
    for case in load_cases(suite):                 # 固定顺序
        pipe = pipeline.build_pipeline(env)        # 每个 case 全新管道 (FM-18)
        pipe.ingest(case.documents)
        result = pipe.answer(case.query)           # 显式捕获本次结果 (FM-08)
        rows.append(score_case(case, result))
        perf.append(collect_latency(pipe))         # 延迟单独收集
    scorecard = aggregate(rows)                    # 不含墙钟字段
    gates = apply_gates(scorecard, load_baseline())
    return {"suite": suite,
            "metrics": scorecard,
            "gates": gates,
            "passed": all(g["ok"] for g in gates if g["hard"]),
            "perf": summarize_latency(perf)}       # perf 单独键, 不参与 gates
```

---

## 10. 确定性策略

| 策略 | 实现位置 | 说明 |
| --- | --- | --- |
| 固定种子 | `core/seeding.py` -> `seed_everything(seed)` | 设置 `random.seed`, `PYTHONHASHSEED`, 所有采样 `temperature=0` |
| 稳定排序 | 全局约定 | 任何排序主键一律 `(-score, id)`, 禁止只按分数排序 (FM-14) |
| 浮点收敛 | `deterministic_view(ndims=6)` | 统一 `round(x, 6)`, `-0.0 -> 0.0` |
| 字典序稳定 | 所有 dict 输出前排序 | 避免 hash 序差异 |
| 无墙钟 | 类型定义不放时间字段 | 延迟只经 `Observer` 走 span/metric |
| 排除视图 | `deterministic_view()` | 见下 |
| 双跑校验 | `assert_deterministic(fn, runs=2)` | 跑两次比对视图, 不等即失败 |
| 全新管道 | 评估与关键单测 | 见 9.3 |

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
    ...

def assert_deterministic(fn, *args, runs: int = 2, ndims: int = 6, **kwargs) -> Any:
    """连续执行 fn 多次, deterministic_view 必须完全相等, 否则抛 AssertionError。
    延迟不参与: 调用方需自行从 perf 通道取。
    """
    ...
```

延迟统计约定:

| 规则 | 说明 |
| --- | --- |
| 延迟只从 `Observer` 的 span 出 | 业务对象 (`Hit`/`Completion`/`ToolResult`/`Plan`) 不得含时间字段 |
| 延迟进 `perf` 通道 | `run_suite` 返回体的 `perf` 键, 或 `/v1/metrics` 的 histograms |
| 断言排除 | 所有 `assert_deterministic` 与 `deterministic_view` 自动剔除墙钟键 |
| P95 只告警 | 阈值 `-1`, 不挂 CI |

---

## 11. 失败模式表

| ID | 症状 | 根因 | 修法 | 守护测试 |
| --- | --- | --- | --- | --- |
| FM-01 | BM25 检索全 0 分或排序反转 | 直接套用 `rank_bm25` 的 IDF 定义, 当词出现在半数及以上文档时 IDF 为 0 或负数 (小语料尤其明显) | 自写 Robertson IDF: `idf = ln(1 + (N - n + 0.5) / (n + 0.5))`, 恒非负; 未命中词直接不入结果 | `test_bm25_idf_non_negative`: 构造 `n = N/2` 与 `n = N` 两种语料, 断言 `idf > 0` 且 top1 为含该词更密的文档 |
| FM-02 | 重复摄入同一 `doc_id` 后, 旧的、更长版本的分块仍在检索结果里 (孤儿块) | 摄入按 upsert/追加语义实现, 短文档覆盖后旧 chunk 未被删除 | ingest 入口先调 `drop_document(doc_id)` (DocStore + VectorStore + LexicalIndex 三处都要), 再写入新分块 | `test_reingest_shorter_doc_no_orphan`: 先摄入 10 段再摄入 3 段, 断言 `count_chunks() == 3` 且旧文本不可检索 |
| FM-03 | 模型只看到第一条证据, 答案遗漏后几条 | 证据块按多行渲染 (含标题行/空行), 消费方按行解析只取首行 | 渲染时把每个证据块压缩为一行: 去掉换行/空行, 多空格折叠为一个; 块间用 `\n` 分隔, 块内禁止 `\n` | `test_evidence_single_line`: 渲染结果按 `\n` 切分后行数 == 证据块数, 且每行不含 `\n` |
| FM-04 | 计算器工具遇到 `1/0` 或 `9**9**9` 直接 500 / 进程崩溃 | 安全求值只归一化 `SyntaxError`, 运行时异常 (`ZeroDivisionError` / `OverflowError` / `ValueError` / `TypeError` / `NameError`) 逃逸 | 求值外层统一 `except Exception` 归一化为 `SafeEvalError` -> `ToolResult(ok=False, error=...)` | `test_safe_eval_normalizes_all`: 参数化 `1/0`, `9**9**9`, `int("x")`, `undefined_name`, `1 +`, 断言全部返回 `ok=False` 且不抛 |
| FM-05 | 中文问句「1+1 等于多少?」路由不到 calculator | 全角标点污染关键字/正则匹配 | 归一化时剥离全角标点 (U+FF1F 全角问号、U+FF01 全角叹号、U+3002 句号、U+FF0C 全角逗号、U+FF1A 全角冒号、U+FF1B 全角分号); **裁剪字符集合严禁包含 `+ - * /`** | `test_fullwidth_punct_normalized` + `test_calc_operators_preserved` (双测试, 后者防止误删运算符) |
| FM-06 | 中文 query 检索不到英文段落 (或反之), 跨语言召回为 0 | 两种语言 token 零交集, 稠密与稀疏双路都失配 | 双语词典扩展**只作用于稀疏分支** (`WF_BILINGUAL` 仅影响 `LexicalIndex` 的查询改写); 不得改写稠密查询向量, 不得改变最终融合权重 | `test_bilingual_sparse_only`: 开启双语扩展后断言 dense 分支 `search` 结果与关闭时完全一致 |
| FM-07 | ReAct 的最终答案输出成 `Action: search(...)` | 提取"最后一行"作为最终答案时, 未排除控制行前缀, 最后一行恰是 Action 行 | 单行兜底必须跳过以 `action:` / `thought:` / `observation:` (大小写不敏感、前后空白剥离、允许中文冒号) 开头的行; 若无合法行则回退到"最近一条 Observation 摘要" | `test_react_final_answer_not_action`: 脚本化 LLM 最后输出 Action 行, 断言 `run_agent()["answer"]` 不以 `Action` 开头 |
| FM-08 | 评估分数系统性偏低, 且与手工抽查结果不一致 | 评估适配器读的是对象上的 `last_answer` / `last_hits` 等缓存字段, 拿到的是上一轮结果 | 评估必须显式捕获本次调用的返回值并传入评估器; 类型上删除所有 `last_*` 字段 | `test_eval_uses_current_result`: 连续两个 case, 断言第二个 case 的评分输入 == 第二个 case 的输出 |
| FM-09 | 常量输入 (全相同分数) 的秩相关被算成 `+1.0`, 掩盖真实退化 | Pearson/Spearman 在标准差为 0 时数学上未定义, 实现里 `0/0` 被兜底成了 1.0 | 常量输入 (任一方 std == 0) 直接返回 `0.0`, 不参与门限均值 | `test_rank_corr_constant_returns_zero`: 输入 `[0.5,0.5,0.5]` 与 `[1,2,3]`, 断言 `== 0.0` |
| FM-10 | grounding / faithfulness 分数莫名被拉低 | 自家答案里的引用标记 `[id#n]`、`(依据: ...)`、`(来源: ...)` 被当成未接地 token 计入分母 | 核验前先用 `core/text.py:strip_citations()` 剥离引用标记 (正则覆盖 `[id#n]`、`[n]`、`(依据: ...)`、`(来源: ...)`、`【n】`) | `test_grounding_strips_citations`: 同一答案带/不带引用标记, 断言 grounding 分数相等 |
| FM-11 | 冒烟脚本 httpx 请求 localhost 报 `WinError 10054` (连接被远程主机强制关闭) | httpx 默认 `trust_env=True`, 走系统代理访问 `127.0.0.1` | 客户端一律 `httpx.Client(..., trust_env=False)`, 或在进程环境设 `NO_PROXY=localhost,127.0.0.1`; stdlib `urllib` 同样检查代理 | `test_smoke_client_disables_env_proxy`: 断言构造参数 `trust_env is False`; 冒烟脚本必须带此参数 |
| FM-12 | clone 后包导入失败, `src/worldforge/core/__init__.py` 缺失 | `.gitignore` 里的 `_*.py` 规则命中了 `__init__.py` | 用根锚定 `/_*` 限定到仓库根的临时脚本, 并补白名单 `!/**/__init__.py` | `test_repo_layout_init_files`: 遍历所有包目录断言存在 `__init__.py`, 并对每个断言 `git check-ignore` 不命中 |
| FM-13 | 切换到生产 Embedder 后检索结果全乱, 或静默出错 | 换了 `dim`/`model_id`, 旧索引向量与新向量混存 | `VectorStore.fingerprint` = `f"{model_id}:{dim}"`, `upsert`/`load` 时校验, 不符抛 `DimensionMismatch` 并提示 `python -m worldforge.admin reindex` | `test_dim_fingerprint_mismatch`: 用 dim=8 的 embedder 写, 用 dim=16 查询, 断言抛 `DimensionMismatch` |
| FM-14 | 混合检索同分数结果顺序在两次运行间漂移 | 融合/重排后只按 `score` 排序, 同分时顺序随底层字典序或输入顺序变化 | 统一稳定排序主键 `(-score, chunk_id)`; 所有 `sorted` 调用走 `core/sortutil.py` 的 `stable_sort` | `test_fusion_tie_break_stable`: 构造 5 条同分候选, 打乱输入顺序跑两次, 断言输出顺序完全一致 |
| FM-15 | 摄入空文本产生 0 分块且不报错; 超大文档把内存打满 | 缺少输入边界校验 | 空/纯空白文本拒绝 (`IngestError`, HTTP 400); 单文档分块数超 `WF_MAX_CHUNKS` (默认 5000) 截断并 emit `ingest.truncated` 告警 | `test_ingest_bounds`: 空文本抛错; 5 万字符文档断言 `chunks <= WF_MAX_CHUNKS` 且 `skipped > 0` |
| FM-16 | 装了 extras 的机器上生产后端不可用, 或直接启动失败 | adapter 工厂在模块 import 期就 `import faiss`, 失败即崩 | 工厂函数内部 `try: import ... except ImportError: raise BackendUnavailable`; `pick()` 捕获后 warn 一次回退默认实现, 并写入 `degraded` 供 `/health` 暴露 | `test_missing_backend_silent_fallback`: monkeypatch 让 import 失败, 断言 `build_pipeline` 成功且 `health()["degraded"]` 含该 tag; `WF_STRICT=on` 时断言抛 `BackendUnavailable` |
| FM-17 | 开 numpy 加速与不开的检索结果有微小差异, 导致评估偶发失败 | 两套实现独立演进, 浮点累加顺序不同 | numpy 只做**加速**不新起一套逻辑: `core/accel.py` 提供同签名函数, 必须过与纯 Python 实现的逐元素对齐测试 (`atol=1e-9`) | `test_numpy_accel_parity`: 随机 200 组向量, 断言 `np.allclose(a, b, atol=1e-9)`; numpy 缺失时测试 skip |
| FM-18 | 评估第二次跑分数变了 (变低), 非确定性回归 | 复用同一个 Pipeline 实例, 索引/缓存/计数器残留 | 每个 case 用 `build_pipeline(env)` 新建; Pipeline 禁止模块级单例; 阶段之间 `clear()` | `test_eval_determinism_two_runs_equal`: 同 suite 跑两次, 断言 `deterministic_view(scorecard)` 完全相等 |

---

## 12. 目录树

```
worldforge-ai/
+-- README.md                        项目说明、30 秒上手、架构一图流
+-- LICENSE                          Apache-2.0
+-- pyproject.toml                   打包元数据, 运行时依赖为空, extras 分组: prod/dev
+-- .gitignore                       根锚定 /_* + !/**/__init__.py (FM-12)
+-- .env.example                     全部 WF_* 变量示例与默认值
+-- Dockerfile                       多阶段构建交付物, 本机不实跑
+-- Makefile                         make verify / make test / make smoke / make baseline
+-- docker-compose.yml               可选: qdrant + ollama 侧车, 仅交付
+-- .github/
|   +-- workflows/
|       +-- ci.yml                   matrix: ubuntu/windows x py3.11/3.12/3.13, 不装 extras 跑 verify
+-- scripts/
|   +-- verify.py                    零依赖冒烟入口, 裸 CPython 直接跑, 退出码 0/1
|   +-- smoke.py                     起服务 + httpx(trust_env=False) 打全流程 (FM-11)
|   +-- bench.py                     吞吐/延迟基准, 输出 perf.json
|   +-- record_baseline.py           生成 eval/baseline.json (runs=5)
|   +-- reindex.py                   指纹变更后重建索引
+-- src/
|   +-- worldforge/
|       +-- __init__.py              暴露 __version__ 与 build_pipeline
|       +-- pipeline.py              L5 装配根: build_pipeline / Pipeline 门面
|       +-- config.py                Config.from_env, WF_* 解析与校验
|       +-- types.py                 Document/Chunk/Hit/Message/Completion/Evidence/Plan/Step 等
|       +-- errors.py                异常体系, 所有 Port 的统一出口
|       +-- core/
|       |   +-- __init__.py
|       |   +-- mathx.py             L2 归一化、余弦、点积、安全除 (零依赖)
|       |   +-- accel.py             numpy 可选加速, 缺失时纯 Python 等价实现 (FM-17)
|       |   +-- hash_embed.py        BLAKE2b 有符号散列嵌入
|       |   +-- bm25.py              Robertson-IDF BM25 打分 (FM-01)
|       |   +-- rrf.py               RRF 融合, k=60
|       |   +-- rank_corr.py         Pearson/Spearman, 常量输入返回 0.0 (FM-09)
|       |   +-- tokenizer.py         中英混合分词 + 全角标点归一 (FM-05)
|       |   +-- safe_eval.py         AST 白名单安全求值, 全异常归一 (FM-04)
|       |   +-- text.py              证据单行化 + 引用标记剥离 (FM-03/FM-10)
|       |   +-- bilingual.py         双语词典扩展, 仅供稀疏分支 (FM-06)
|       |   +-- sortutil.py          stable_sort: (-score, id) 统一排序 (FM-14)
|       |   +-- seeding.py           seed_everything / deterministic_view / assert_deterministic
|       +-- ports/
|       |   +-- __init__.py          汇总导出 10 个 Protocol
|       |   +-- llm.py               LLM Protocol
|       |   +-- embedder.py          Embedder Protocol
|       |   +-- vectorstore.py       VectorStore Protocol
|       |   +-- lexical.py           LexicalIndex Protocol
|       |   +-- reranker.py          Reranker Protocol
|       |   +-- docstore.py          DocStore Protocol
|       |   +-- observer.py          Observer Protocol
|       |   +-- tool.py              Tool Protocol
|       |   +-- planner.py           Planner Protocol
|       |   +-- chunker.py           Chunker Protocol
|       +-- adapters/
|       |   +-- __init__.py
|       |   +-- registry.py          后端名 -> 工厂 映射表 + warn_once + 降级标记
|       |   +-- embedder_hashing.py  默认: HashingEmbedder (BLAKE2b 有符号)
|       |   +-- embedder_fastembed.py  extras: fastembed
|       |   +-- embedder_openai.py   extras: OpenAI embeddings
|       |   +-- vectorstore_memory.py 默认: InMemoryExactIndex 精确余弦
|       |   +-- vectorstore_faiss.py extras: FAISS (CPU)
|       |   +-- vectorstore_qdrant.py extras: Qdrant
|       |   +-- lexical_bm25.py      默认: Robertson-IDF BM25 倒排
|       |   +-- lexical_elastic.py   extras: Elasticsearch
|       |   +-- reranker_lexical.py  默认: IDF 加权词法重排
|       |   +-- reranker_crossencoder.py extras: bge-reranker / Cohere
|       |   +-- llm_mock.py          默认: ExtractiveMockLLM 抽取式真实现
|       |   +-- llm_openai.py        extras: OpenAI Chat Completions
|       |   +-- llm_ollama.py        extras: Ollama
|       |   +-- llm_llamacpp.py      extras: llama-cpp-python
|       |   +-- docstore_memory.py   默认: 进程内 dict 存储
|       |   +-- docstore_sqlite.py   stdlib sqlite3 持久化 (非默认)
|       |   +-- docstore_postgres.py extras: Postgres
|       |   +-- observer_memory.py   默认: 环形缓冲 Observer
|       |   +-- observer_otlp.py     extras: OpenTelemetry / Prometheus
|       |   +-- chunker_recursive.py 默认: 递归字符分块
|       |   +-- chunker_semantic.py  extras: 语义分块
|       |   +-- tool_mcp.py          extras: MCP 工具源
|       +-- retrieval/
|       |   +-- __init__.py
|       |   +-- chunking.py          Chunker 门面 + 摄入时的分块编排
|       |   +-- sparse.py            稀疏召回编排 + 双语改写接入点
|       |   +-- dense.py             稠密召回编排
|       |   +-- fusion.py            RRF / 加权融合, 输出 source=fused
|       |   +-- hybrid.py            HybridRetriever: 双路 -> 融合 -> 重排
|       |   +-- evidence.py          证据取回与**单行化**渲染 (FM-03)
|       +-- agents/
|       |   +-- __init__.py
|       |   +-- react.py             ReAct 循环 + 最终答案提取兜底 (FM-07)
|       |   +-- planner_rule.py      默认: RuleBasedPlanner
|       |   +-- planner_llm.py       LLMPlanner, 缺 LLM 时降级
|       |   +-- dag.py               步骤 DAG: 拓扑排序 + 环检测 + 并行占位
|       |   +-- tools.py             工具注册表与路由 (归一化后匹配, FM-05)
|       |   +-- builtin_tools.py     calculator / retriever / dictionary / echo
|       |   +-- verify.py            接地与忠实度核验 (剥离引用标记, FM-10)
|       +-- serving/
|       |   +-- __init__.py
|       |   +-- app.py               stdlib ThreadingHTTPServer 启动与路由分发
|       |   +-- schemas.py           请求/响应 dataclass 与校验
|       |   +-- errors.py            HTTP 错误信封与状态码映射
|       |   +-- routes_health.py     GET /health
|       |   +-- routes_ingest.py     POST /v1/ingest, DELETE /v1/documents/{id}
|       |   +-- routes_query.py      POST /v1/query
|       |   +-- routes_chat.py       POST /v1/chat/completions (含 SSE 流)
|       |   +-- routes_eval.py       POST /v1/evaluate
|       |   +-- routes_metrics.py    GET /v1/metrics, GET /v1/traces/{id}
|       |   +-- routes_admin.py      POST /v1/admin/reset
|       |   +-- openai_compat.py     OpenAI 兼容请求/响应转换
|       +-- eval/
|       |   +-- __init__.py
|       |   +-- metrics.py           doc_hit_rate/recall/mrr/ndcg/grounding/faithfulness
|       |   +-- scorecard.py         聚合 + 门限判定 + EvalGateError
|       |   +-- runner.py            套件执行 (每 case 全新管道, FM-18)
|       |   +-- baseline.py          基线读写与 threshold 计算
|       |   +-- baseline.json        实测基线 (由 record_baseline.py 生成)
|       |   +-- datasets/
|       |   |   +-- __init__.py
|       |   |   +-- golden_retrieval.jsonl  检索黄金集
|       |   |   +-- golden_rag.jsonl        RAG 问答黄金集
|       |   |   +-- golden_agent.jsonl      工具路由黄金集
|       |   +-- suites/
|       |       +-- __init__.py
|       |       +-- retrieval_suite.py  检索指标套件
|       |       +-- rag_suite.py         接地/忠实度套件
|       |       +-- agent_suite.py       工具路由与执行套件
|       +-- observability/
|           +-- __init__.py
|           +-- trace.py             span 上下文管理器, 墙钟只在此处产生
|           +-- metrics.py           counter/gauge/histogram 聚合与分位
|           +-- exporters.py         JSON / Prometheus 文本导出
+-- tests/
|   +-- __init__.py
|   +-- conftest.py                  公共 fixture: 固定种子、临时 env
|   +-- fakes.py                     FakeLLM/FakeEmbedder/RecordingObserver/FakeTool 等
|   +-- test_core_mathx.py
|   +-- test_core_bm25.py            FM-01
|   +-- test_core_rank_corr.py       FM-09
|   +-- test_core_safe_eval.py       FM-04
|   +-- test_core_tokenizer.py       FM-05
|   +-- test_core_text.py            FM-03 / FM-10
|   +-- test_core_seeding.py         deterministic_view / assert_deterministic
|   +-- test_core_accel_parity.py    FM-17 (无 numpy 时 skip)
|   +-- test_ports_contracts.py      10 个 Protocol 的 runtime_checkable 校验
|   +-- test_adapters_default_impl.py 默认实现非 stub 断言 (真检索能召回)
|   +-- test_adapters_fallback.py    FM-16
|   +-- test_retrieval_ingest.py     FM-02 / FM-13 / FM-15
|   +-- test_retrieval_hybrid.py     FM-06 / FM-14
|   +-- test_agents_react.py         FM-07
|   +-- test_agents_planner.py
|   +-- test_agents_verify.py        FM-10
|   +-- test_serving_api.py          全端点契约与错误码
|   +-- test_serving_openai_compat.py
|   +-- test_eval_runner.py          FM-08 / FM-18
|   +-- test_eval_gates.py           门限计算与基线比对
|   +-- test_pipeline_assembly.py    build_pipeline 各开关组合
|   +-- test_layering.py             分层依赖静态检查 (禁止反向 import)
|   +-- test_repo_layout.py          FM-12
+-- docs/
    +-- ARCHITECTURE.md              本文档
    +-- API.md                       HTTP 接口参考 (由 schemas.py 生成)
    +-- ADAPTERS.md                  如何新增一个后端实现
    +-- EVAL.md                      评估与基线操作手册
    +-- DETERMINISM.md               确定性约定细则
```

---

## 13. 版本与依赖锁定

| 项 | 值 |
| --- | --- |
| Python | `>=3.11`, 开发实测 `3.13.14` |
| 运行时依赖 | 无 (`dependencies = []`) |
| 可选 extras | `[prod]`: faiss-cpu, qdrant-client, fastembed, openai, ollama, llama-cpp-python, httpx, elasticsearch; `[dev]`: pytest 9.1.1, ruff, mypy |
| numpy | 可选加速, 默认路径不 import; 已装 2.5.3 |
| pytest | 9.1.1 (仅测试) |
| 锁定策略 | `pyproject.toml` 只写下限 `>=`, 精确版本写入 `requirements-dev.txt`; CI 用 `pip install -r requirements-dev.txt` |
| 指纹锁定 | `VectorStore.fingerprint` 与 `Embedder.model_id` 参与索引校验, 防止静默混存 |

---

## 14. 成本与延迟

| 维度 | 默认栈 (零依赖, CPU) | 生产栈 (extras) |
| --- | --- | --- |
| 货币成本 | 0 元 (无 API 调用, 纯本地 CPU) | 按调用量计, 见下 |
| 嵌入成本 | 0 | `cost = tokens_in * price_in + 0` (`price_*` 从 `WF_PRICE_TABLE` 读取, 不在代码内硬编码; 单价以厂商官方定价页当日信息为准) |
| 生成成本 | 0 | `cost = prompt_tokens * price_in + completion_tokens * price_out` |
| 缓存策略 | 证据渲染与提示词按 `(query_norm, chunk_ids)` 哈希缓存, 命中跳过重排 | 同左 + 远端响应缓存 (`WF_CACHE=on`) |
| 批处理 | 摄入走批量 `embed(texts)` 而非逐条 | 同左 |
| 降级 | `WF_STRICT=off` 时任一端不可用即回退默认栈 | 预算上限 `WF_COST_CAP` 触发后强制回退 `mock` LLM |

延迟预算 (CPU 单进程, 目标值待实跑校准, 由 `scripts/bench.py` 填实际值):

| 阶段 | 预算 | 说明 |
| --- | --- | --- |
| 嵌入 1 条查询 | <= 5 ms | hashing 实现, O(tokens) |
| BM25 稀疏召回 (10k 块) | <= 20 ms | 倒排 + 堆排序 |
| 稠密精确召回 (10k 块) | <= 60 ms | 纯 Python; 开 numpy 后 <= 10 ms |
| RRF 融合 | <= 2 ms | |
| 词法重排 | <= 5 ms | |
| LLM 生成 (mock) | <= 15 ms | 抽取式, 无网络 |
| 端到端 `/v1/query` p95 | <= 150 ms | 默认栈 |
| 端到端 `/v1/chat/completions` p95 | <= 300 ms | 默认栈, 含核验 |

---

## 15. 实施顺序建议 (工期受限时的最小可交付集)

| 优先级 | 模块 | 理由 |
| --- | --- | --- |
| P0 | `types.py` + `errors.py` + `core/` (mathx, hash_embed, bm25, tokenizer, safe_eval, text, seeding) | 一切的地基, 且是失败模式最集中的地方 |
| P0 | `ports/` 全部 10 个 Protocol | 契约先行, 后续所有实现可并行 |
| P0 | 默认 adapters (hashing / memory / bm25 / lexical / mock / recursive) + `registry.py` | 决定"零依赖可跑"这条铁律能否成立 |
| P0 | `pipeline.py` + `scripts/verify.py` | 交付门禁, 没有它无法证明可跑 |
| P1 | `retrieval/` (chunking, sparse, dense, fusion, hybrid, evidence) | 主场景 RAG |
| P1 | `serving/` (`/health`, `/v1/ingest`, `/v1/query`) | 可演示 |
| P2 | `agents/` (planner_rule, react, tools, builtin_tools, verify) | Agent 编排 |
| P2 | `serving/` (`/v1/chat/completions`, OpenAI 兼容) | 生态兼容 |
| P3 | `eval/` + `baseline.json` + `record_baseline.py` | 回归守护 |
| P3 | `observability/` + `/v1/metrics` | 可观测 |
| P4 | 生产 adapters 全家桶 + `Dockerfile` + CI | 仅为交付物, 不阻塞主线 |
