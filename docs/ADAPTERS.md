# 如何新增一个后端实现 (Adapter 开发指南)

- 对应架构: `docs/ARCHITECTURE.md` 第 4 节 (Ports 契约)、第 5 节 (可插拔矩阵)、第 6 节 (装配)
- 相关决策: `docs/decisions/ADR-0001.md` (零依赖核心与注入, D3 真实现、D4 静默回退)
- 适用范围: `src/worldforge/adapters/`

---

## 0. 结论先行

| 规则 | 内容 |
| --- | --- |
| 命名 | 文件 `adapters/<capability>_<backend>.py`; 类 `<Backend><Capability>`; 工厂 `_<capability>_<backend>(cfg)` |
| 继承 | 不需要继承任何基类, `typing.Protocol` 是结构化子类型, 方法签名对齐即算实现 |
| 默认实现 | 必须是真实现, 禁止 stub (`ADR-0001` D3) |
| 生产实现 | 依赖放 extras; 惰性 import; 失败抛 `BackendUnavailable`, **绝不让 `ImportError` 冒泡到调用方** |
| 降级 | `pick()` 捕获 -> `warn_once` 一次 -> 回退默认实现 -> `degraded` 记录 tag |
| 严格模式 | `WORLDFORGE_STRICT=on` 时不回退, 直接向外抛 `BackendUnavailable` |
| 注册 | 必须同时改 `registry.py` 与 `config.py`, 缺一处则环境变量不生效 |

---

## 1. 十步流程

| # | 步骤 | 产物 | 常见漏项 |
| --- | --- | --- | --- |
| 1 | 选定要实现的 Port | 从下表选一个 | 选错层: 把能力写进 `retrieval/` 而不是 `adapters/` |
| 2 | 建文件 `adapters/<capability>_<backend>.py` | 新文件 | 文件名不遵约定, 后面 registry 找不到 |
| 3 | 实现 Protocol 的**全部**方法 | 类 | 漏实现某个方法, 运行时才报 AttributeError |
| 4 | 惰性 import 第三方库, 失败抛 `BackendUnavailable` | 构造函数 | 顶层 `import torch`, 模块一导入就崩 |
| 5 | 写工厂函数 `_<capability>_<backend>(cfg)` | 函数 | 工厂里直接 import 且不做异常转换 |
| 6 | 在 `adapters/registry.py` 注册表加一行 | 字典条目 | 只加了类没加工厂 |
| 7 | 在 `config.py` 加环境变量取值分支 | `CHOICES` 元组 | 忘了加 choices, 配置解析报 `ConfigError` |
| 8 | 补正常路径单测 | `tests/test_adapter_<backend>.py` | 只测了能构造, 没测行为 |
| 9 | 补降级路径单测 (依赖缺失时必须回退) | 同文件 | 最容易漏, 也是线上最难发现的一条 |
| 10 | 更新 `docs/ARCHITECTURE.md` 第 5 节矩阵与 `README` | 文档 | 不更新, 使用者以为不支持 |

### 1.1 可选 Port 一览

| capability | Port | 默认实现 | 文件前缀 |
| --- | --- | --- | --- |
| `embedder` | `Embedder` | `HashingEmbedder` | `embedder_*` |
| `vectorstore` | `VectorStore` | `InMemoryExactIndex` | `vectorstore_*` |
| `lexical` | `LexicalIndex` | `BM25Index` | `lexical_*` |
| `reranker` | `Reranker` | `LexicalReranker` | `reranker_*` |
| `llm` | `LLM` | `ExtractiveMockLLM` | `llm_*` |
| `docstore` | `DocStore` | `MemoryDocStore` | `docstore_*` |
| `observer` | `Observer` | `MemoryObserver` | `observer_*` |
| `chunker` | `Chunker` | `RecursiveCharChunker` | `chunker_*` |
| `tool` | `Tool` | builtin 工具集 | `tool_*` |
| `planner` | `Planner` | `RuleBasedPlanner` | `planner_*` (位于 `agents/`) |

新增一个**新 Port** 与新增一个**已有 Port 的后端**不是同一件事: 前者要改 `ports/` 与 `ARCHITECTURE.md` 第 4 节, 需要架构评审; 后者照本文档走即可。

---

## 2. 完整示例 (一): 零依赖后端 `CharGramEmbedder`

一个可以直接照抄、零第三方依赖即可跑通的完整 adapter: 字符 n-gram 有符号散列嵌入。

### 2.1 先看 Port 契约

`Embedder` 要求的方法 (见 `ARCHITECTURE.md` 第 4.3 节):

| 成员 | 签名 | 必须满足 |
| --- | --- | --- |
| `dim` | `@property -> int` | 与 `embed` 输出长度一致 |
| `model_id` | `@property -> str` | 写入索引指纹, 不同实现必须不同 |
| `embed` | `(texts: Sequence[str]) -> list[list[float]]` | 空输入返回 `[]`; 输出已 L2 归一化; 同输入逐位一致 |
| `embed_one` | `(text: str) -> list[float]` | 等价 `embed([text])[0]` |

### 2.2 完整代码

```python
# src/worldforge/adapters/embedder_chargram.py
"""字符 n-gram 有符号散列嵌入 (零第三方依赖)。

设计要点:
1. 中文按字切、英文按词切, 兼容中英混排 (复用 core/tokenizer)
2. 有符号散列: 用 BLAKE2b 摘要的末字节决定符号, 抵消单向累加偏差
3. 输出 L2 归一化, 使余弦相似度等价于点积
4. 无任何 I/O、无全局状态, 天然确定性
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from worldforge.core import mathx, tokenizer
from worldforge.errors import WorldforgeError
from worldforge.ports.embedder import Embedder  # noqa: F401  仅用于静态自检

_MAX_TEXT = 100_000  # 单文本字符上限, 超过抛 ValueError


class CharGramEmbedder:
    """Embedder 的 chargram 后端实现 (不继承 Embedder, 结构化子类型自动成立)。"""

    name = "chargram"

    def __init__(self, dim: int = 256, ngram: int = 3, salt: str = "wf-chargram-v1"):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if ngram <= 0:
            raise ValueError(f"ngram must be positive, got {ngram}")
        self._dim = dim
        self._ngram = ngram
        self._salt = salt.encode("utf-8")

    # ---- Port 契约: 属性 ----

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        # 参与索引指纹, 必须区分 dim/ngram/salt 的差异
        return f"chargram:d{self._dim}:n{self._ngram}:{self._salt.decode('utf-8')}"

    # ---- Port 契约: 方法 ----

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for text in texts:
            if text is None or not isinstance(text, str):
                raise ValueError(f"text must be str, got {type(text).__name__}")
            if len(text) > _MAX_TEXT:
                raise ValueError(f"text too long: {len(text)} > {_MAX_TEXT}")
            out.append(self._embed_one(text))
        return out

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one(text)

    # ---- 内部实现 ----

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        grams = tokenizer.char_ngrams(text, n=self._ngram)
        if not grams:
            # 空文本返回零向量, 由调用方决定是否入索引
            return vec
        for gram in grams:
            idx, sign = self._bucket(gram)
            vec[idx] += sign * 1.0
        return mathx.l2_normalize(vec)

    def _bucket(self, gram: str) -> tuple[int, int]:
        digest = hashlib.blake2b(
            gram.encode("utf-8"), digest_size=8, key=self._salt[:16]
        ).digest()
        value = int.from_bytes(digest[:6], "big")
        idx = value % self._dim
        sign = 1 if digest[7] & 0x01 else -1
        return idx, sign
```

### 2.3 注册到 `registry.py`

```python
# src/worldforge/adapters/registry.py (追加)

def _embedder_hashing(cfg):
    from worldforge.adapters.embedder_hashing import HashingEmbedder
    return HashingEmbedder(dim=cfg.embed_dim)


def _embedder_chargram(cfg):
    # 工厂内部 import: 无论零依赖还是生产实现, 一律保持这个形状
    from worldforge.adapters.embedder_chargram import CharGramEmbedder
    return CharGramEmbedder(dim=cfg.embed_dim, ngram=cfg.embed_ngram)


def _embedder_fastembed(cfg):
    from worldforge.adapters.embedder_fastembed import FastEmbedEmbedder
    return FastEmbedEmbedder(model_id=cfg.embed_model)


EMBEDDERS = {
    "hashing":   _embedder_hashing,     # 默认, 必须排在第一位语义上作为兜底
    "chargram":  _embedder_chargram,
    "fastembed": _embedder_fastembed,
}
```

### 2.4 加环境变量分支 `config.py`

```python
# src/worldforge/config.py (追加)

EMBEDDER_CHOICES = ("hashing", "chargram", "fastembed")

@dataclass(frozen=True)
class Config:
    embedder: str = "hashing"
    embed_dim: int = 256
    embed_ngram: int = 3
    # ...

    @classmethod
    def from_env(cls, env):
        return cls(
            embedder=_enum(env, "WORLDFORGE_EMBEDDER", "hashing", EMBEDDER_CHOICES),
            embed_dim=_int(env, "WORLDFORGE_EMBED_DIM", 256),
            embed_ngram=_int(env, "WORLDFORGE_EMBED_NGRAM", 3),
            # ...
        )
```

`_enum` 遇到未登记的取值必须抛 `ConfigError`, 不允许静默忽略 -- 否则拼错环境变量时会静默用默认后端。

### 2.5 单测

```python
# tests/test_adapter_chargram.py
import math
import pytest

from worldforge.adapters.embedder_chargram import CharGramEmbedder
from worldforge.core import mathx
from worldforge.core.seeding import assert_deterministic


def test_chargram_is_real_not_stub():
    """ADR-0001 D3: 默认/新增实现必须是真实现。"""
    e = CharGramEmbedder(dim=64)
    v1 = e.embed_one("检索层怎么分层")
    v2 = e.embed_one("向量库怎么选型")
    assert len(v1) == 64
    assert v1 != v2                                  # 不是常量向量
    assert sum(1 for x in v1 if x != 0.0) > 1        # 不是 one-hot 假实现


def test_chargram_normalized():
    e = CharGramEmbedder(dim=64)
    for text in ["hello world", "中文测试", "a"]:
        v = e.embed_one(text)
        assert abs(mathx.l2_norm(v) - 1.0) < 1e-12


def test_chargram_deterministic_across_instances():
    a = CharGramEmbedder(dim=64)
    b = CharGramEmbedder(dim=64)
    assert a.embed(["x", "y", "z"]) == b.embed(["x", "y", "z"])


def test_chargram_deterministic_two_runs():
    e = CharGramEmbedder(dim=64)
    assert_deterministic(lambda: e.embed(["x", "y", "z"]), runs=2)


def test_chargram_empty_input_returns_empty():
    assert CharGramEmbedder(dim=64).embed([]) == []


def test_chargram_model_id_distinguishes_config():
    a = CharGramEmbedder(dim=64)
    b = CharGramEmbedder(dim=128)
    assert a.model_id != b.model_id                  # 指纹必须能挡住混存
```

---

## 3. 完整示例 (二): 生产后端与降级契约 `CrossEncoderReranker`

生产后端的关键不是算法, 是**导入失败时的行为**。

### 3.1 实现

```python
# src/worldforge/adapters/reranker_crossencoder.py
from __future__ import annotations

from typing import Sequence

from worldforge.errors import BackendUnavailable
from worldforge.ports.reranker import Reranker  # noqa: F401
from worldforge.core.sortutil import stable_sort
from worldforge.types import Hit


class CrossEncoderReranker:
    """Reranker 的 cross-encoder 后端 (extras: sentence-transformers)。"""

    name = "crossencoder"

    def __init__(self, model_id: str = "BAAI/bge-reranker-base", batch_size: int = 16):
        try:
            from sentence_transformers import CrossEncoder   # 惰性 import, 绝不在模块顶层
        except ImportError as exc:
            # 关键: 转成受控异常, 不让 ImportError 冒泡
            raise BackendUnavailable(
                f"reranker=crossencoder requires sentence-transformers: {exc}"
            ) from exc
        self._model = CrossEncoder(model_id)
        self.model_id = model_id
        self._batch_size = batch_size
        self._text_cache: dict[str, str] = {}

    def rerank(self, query: str, candidates: Sequence[Hit], top_k: int) -> list[Hit]:
        if not candidates:
            return []
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        pairs = [(query, self._text_of(h)) for h in candidates]
        scores = self._model.predict(pairs, batch_size=self._batch_size)

        reranked = [
            Hit(chunk_id=h.chunk_id, score=float(s), source="rerank", meta=dict(h.meta))
            for h, s in zip(candidates, scores)
        ]
        return stable_sort(reranked, key=lambda h: h.score, tie_key=lambda h: h.chunk_id)[:top_k]

    def _text_of(self, hit: Hit) -> str:
        # 生产实现自行解决文本取回; 不要依赖调用方塞 text 字段
        return hit.meta.get("text", "")
```

### 3.2 降级契约 (强制)

| 规则 | 要求 | 违反后果 |
| --- | --- | --- |
| 惰性 import | 第三方库只在构造函数/工厂内部 import, 模块顶层零依赖 | 模块一被 import 就崩, 连 `/health` 都起不来 |
| 异常转换 | `except ImportError: raise BackendUnavailable(...)` | `ImportError` 冒泡到 `pick()` 之外, 服务 500 |
| 连接失败同理 | 远端不可达也抛 `BackendUnavailable`, 不是裸 `ConnectionError` | 装配期无法区分「没装」与「装了但连不上」 |
| 工厂形状一致 | `_<capability>_<backend>(cfg)` 内部 import | registry 无法统一处理 |
| 默认实现无降级 | `hashing` / `memory` / `bm25` 等不得有 try/except 回退 | 兜底层自己再兜底, 出现无限回退 |
| warn 一次 | `warn_once(tag)` 用模块级 set 去重, 进程内每个 tag 只 warn 一次 | 日志被刷屏 |
| degraded 可见 | 回退时 `degraded.append(tag)`, 暴露在 `/health` | 用户以为在用 FAISS, 实际在跑暴力检索 |
| 严格模式 | `WORLDFORGE_STRICT=on` 时 `pick()` 直接抛出, 不回退 | 生产环境误配无法提前发现 |

`pick()` 的参考实现 (与 `ARCHITECTURE.md` 第 6.2 节一致):

```python
def pick(kind, table, default_key, tag, cfg, degraded):
    factory = table.get(kind)
    if factory is None:
        raise config.ConfigError(f"unknown backend '{kind}' for {tag}")
    if kind == default_key:
        return factory(cfg)
    try:
        return factory(cfg)
    except (ImportError, BackendUnavailable) as exc:   # ImportError 兜底防漏网
        if cfg.strict:
            raise
        warn_once(f"[{tag}] '{kind}' unavailable ({exc}); fallback -> '{default_key}'")
        degraded.append(tag)
        return table[default_key](cfg)
```

### 3.3 降级单测 (必写)

```python
# tests/test_adapters_fallback.py
import builtins
import pytest

from worldforge.errors import BackendUnavailable
from worldforge.pipeline import build_pipeline


def _block(monkeypatch, *prefixes):
    """让指定前缀的 import 失败, 模拟 extras 未安装。"""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if any(name.startswith(p) for p in prefixes):
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_backend_silent_fallback(monkeypatch):
    _block(monkeypatch, "sentence_transformers")
    pipe = build_pipeline({"WORLDFORGE_RERANKER": "crossencoder"})
    health = pipe.health()
    assert health["backends"]["reranker"] == "lexical"   # 回退到默认
    assert "reranker" in health["degraded"]              # 降级可见
    assert pipe.query("任意查询") is not None             # 服务仍可用


def test_strict_mode_raises_instead_of_fallback(monkeypatch):
    _block(monkeypatch, "sentence_transformers")
    with pytest.raises(BackendUnavailable):
        build_pipeline({"WORLDFORGE_RERANKER": "crossencoder", "WORLDFORGE_STRICT": "on"})


def test_default_backend_never_degrades():
    pipe = build_pipeline({"WORLDFORGE_RERANKER": "lexical"})
    assert pipe.health()["degraded"] == []


def test_unknown_backend_is_config_error():
    from worldforge.errors import ConfigError
    with pytest.raises(ConfigError):
        build_pipeline({"WORLDFORGE_RERANKER": "no-such-backend"})
```

---

## 4. 契约测试: 所有后端跑同一组断言

新增后端后, 把它加进契约测试的 parametrize 列表, 与默认实现跑同一组断言。这样两套实现的口径不会悄悄漂移。

```python
# tests/test_adapters_contract.py
import pytest
from worldforge.adapters import registry

BACKENDS = [
    ("hashing",  registry.EMBEDDERS),
    ("chargram", registry.EMBEDDERS),
]

@pytest.mark.parametrize("kind,table", BACKENDS)
def test_embedder_contract(kind, table):
    from worldforge.config import Config
    cfg = Config()
    emb = table[kind](cfg)

    assert emb.dim > 0
    assert isinstance(emb.model_id, str) and emb.model_id
    assert emb.embed([]) == []

    vs = emb.embed(["alpha", "beta", "中文文本"])
    assert len(vs) == 3
    assert all(len(v) == emb.dim for v in vs)
    assert vs[0] != vs[1]                       # 不是常量 stub
    assert emb.embed(["alpha"])[0] == vs[0]     # 确定性
```

契约测试跳过规则: 生产后端未安装时用 `pytest.importorskip` 或 `pytest.mark.skipif`, **不得** 直接 fail。

---

## 5. 检查清单 (提交前逐项打勾)

| # | 检查项 | 不满足的后果 |
| --- | --- | --- |
| 1 | 文件命名 `<capability>_<backend>.py` | registry 与文档对不上 |
| 2 | Protocol 全部方法已实现 (逐个比对第 1.1 节对应 Port) | 运行时 AttributeError |
| 3 | 第三方 import 在构造函数/工厂内部 | 模块导入即崩 |
| 4 | `ImportError` 已转为 `BackendUnavailable` | 异常逃逸, 服务 500 |
| 5 | 已加 `_<capability>_<backend>(cfg)` 工厂 | registry 无法调用 |
| 6 | 已加进 `registry.<CAPABILITY>` 字典 | 环境变量不生效 |
| 7 | 已加进 `config.py` 的 `*_CHOICES` | 配了却被判为非法值 |
| 8 | `model_id` / 指纹能区分配置差异 | 旧索引与新向量混存, 检索全乱 |
| 9 | 排序走 `stable_sort((-score, id))` | 同分顺序漂移 |
| 10 | 有正常路径单测 + 降级路径单测 | 线上静默降级无人知 |
| 11 | 已加进 `tests/test_adapters_contract.py` 参数列表 | 两套实现漂移 |
| 12 | 已更新 `ARCHITECTURE.md` 第 5 节矩阵 | 使用者不知道有这个后端 |
| 13 | 许可合规 (见 `ADR-0005` 白名单) | 引入 AGPL / 专有许可依赖 |
| 14 | README 标注能力上限 | 用户把演示级实现当生产用 |

---

## 6. 常见错误

| 症状 | 根因 | 修法 |
| --- | --- | --- |
| 配了 `WORLDFORGE_VECTORSTORE=faiss` 却仍在跑暴力检索 | 只加了类没加工厂, 或没进 registry 字典 | 检查第 5 节清单第 5、6 项 |
| 启动就崩, 报 `ModuleNotFoundError: faiss` | 模块顶层 `import faiss` | 移进构造函数, 并转 `BackendUnavailable` |
| `/health` 显示 `degraded` 但没日志 | `warn_once` 去重后只输出一次, 被日志级别过滤 | 以 `/health.degraded` 为准, 日志只作辅助 |
| 换后端后检索结果全乱 | `model_id` 未区分配置, 新旧向量混存 | 让 `model_id` 带上后端名与维度; 索引 `fingerprint` 校验 |
| 评估分数在生产后端下反而下降 | 两套实现口径不一致 (归一化/分词/排序) | 跑契约测试; 两环境两列数字对照 (见 `DETERMINISM.md`) |
| 单测里 import 了真实后端导致 CI 挂 | 契约测试未做 skipif | 用 `pytest.importorskip` |
| 新增后端后 `ConfigError: unknown backend` | 忘了加 `*_CHOICES` | 补 `config.py` |
| 重复摄入后旧分块残留 | 新后端只实现了 `upsert` 没实现 `drop_document` | `VectorStore` / `LexicalIndex` 必须实现 `drop_document`, ingest 入口先 drop (`ARCHITECTURE.md` FM-02) |
