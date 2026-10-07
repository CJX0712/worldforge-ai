"""后端工厂注册表、一次性告警与降级标记。

作者：晨星
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

_WARNED: set[str] = set()
_DEGRADED: list[str] = []
DEGRADED: tuple[str, ...] = ()


def warn_once(message: str, *, tag: str | None = None) -> None:
    """按标签去重告警，避免装配期刷屏。"""

    key = tag or message
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def reset_warnings() -> None:
    """清空一次性告警记录，仅供测试使用。"""

    _WARNED.clear()


def mark_degraded(tag: str) -> tuple[str, ...]:
    """记录一个发生回退的能力标签。"""

    if tag not in _DEGRADED:
        _DEGRADED.append(tag)
    return degraded()


def clear_degraded() -> None:
    """清空降级标记，仅供测试与新建管道使用。"""

    _DEGRADED.clear()


def degraded() -> tuple[str, ...]:
    """返回当前进程累计的降级标签。"""

    return tuple(_DEGRADED)


def pick(
    kind: str,
    table: Mapping[str, Callable[[Any], Any]],
    default_key: str,
    tag: str,
    cfg: Any,
) -> Any:
    """选择后端：生产实现不可用时回退默认实现并记录降级。"""

    factory = table.get(kind)
    if factory is None:
        from worldforge.errors import ConfigError

        raise ConfigError(f"unknown backend '{kind}' for {tag}")
    if kind == default_key:
        return factory(cfg)
    try:
        return factory(cfg)
    except (ImportError, _backend_unavailable()) as exc:
        if getattr(cfg, "strict", False):
            raise
        warn_once(f"[{tag}] '{kind}' unavailable ({exc}); fallback -> '{default_key}'", tag=tag)
        mark_degraded(tag)
        return table[default_key](cfg)


def _backend_unavailable() -> type[BaseException]:
    from worldforge.errors import BackendUnavailable

    return BackendUnavailable


def _embedder_hashing(cfg: Any) -> Any:
    from worldforge.adapters.embedder_hashing import HashingEmbedder

    return HashingEmbedder(dim=cfg.embed_dim)


def _embedder_fastembed(cfg: Any) -> Any:
    from worldforge.adapters.embedder_fastembed import FastEmbedEmbedder

    return FastEmbedEmbedder(model_id=cfg.embed_model)


def _embedder_openai(cfg: Any) -> Any:
    from worldforge.adapters.embedder_openai import OpenAIEmbedder

    return OpenAIEmbedder(model_id=cfg.embed_model, api_base=cfg.api_base, api_key_env=cfg.api_key_env)


def _vectorstore_memory(cfg: Any) -> Any:
    from worldforge.adapters.vectorstore_memory import InMemoryExactIndex

    return InMemoryExactIndex(dim=cfg.embed_dim)


def _vectorstore_faiss(cfg: Any) -> Any:
    from worldforge.adapters.vectorstore_faiss import FaissStore

    return FaissStore(dim=cfg.embed_dim)


def _vectorstore_qdrant(cfg: Any) -> Any:
    from worldforge.adapters.vectorstore_qdrant import QdrantStore

    return QdrantStore(dim=cfg.embed_dim, url=cfg.vector_url)


def _lexical_bm25(cfg: Any) -> Any:
    from worldforge.adapters.lexical_bm25 import BM25Index

    return BM25Index()


def _lexical_elastic(cfg: Any) -> Any:
    from worldforge.adapters.lexical_elastic import ElasticLexical

    return ElasticLexical(url=cfg.vector_url)


def _reranker_lexical(cfg: Any) -> Any:
    from worldforge.adapters.reranker_lexical import LexicalReranker

    return LexicalReranker()


def _reranker_crossencoder(cfg: Any) -> Any:
    from worldforge.adapters.reranker_crossencoder import CrossEncoderReranker

    return CrossEncoderReranker()


def _llm_mock(cfg: Any) -> Any:
    from worldforge.adapters.llm_mock import ExtractiveMockLLM

    return ExtractiveMockLLM(model_id=cfg.llm_model, max_tokens=512)


def _llm_openai(cfg: Any) -> Any:
    from worldforge.adapters.llm_openai import OpenAILLM

    return OpenAILLM(model_id=cfg.llm_model, api_base=cfg.api_base, api_key_env=cfg.api_key_env)


def _llm_ollama(cfg: Any) -> Any:
    from worldforge.adapters.llm_ollama import OllamaLLM

    return OllamaLLM(model_id=cfg.llm_model, api_base=cfg.api_base)


def _llm_llamacpp(cfg: Any) -> Any:
    from worldforge.adapters.llm_llamacpp import LlamaCppLLM

    return LlamaCppLLM(model_id=cfg.llm_model)


def _docstore_memory(cfg: Any) -> Any:
    from worldforge.adapters.docstore_memory import MemoryDocStore

    return MemoryDocStore()


def _docstore_sqlite(cfg: Any) -> Any:
    from worldforge.adapters.docstore_sqlite import SqliteDocStore

    return SqliteDocStore(path=cfg.sqlite_path)


def _docstore_postgres(cfg: Any) -> Any:
    from worldforge.adapters.docstore_postgres import PostgresDocStore

    return PostgresDocStore(dsn=cfg.vector_url)


def _observer_memory(cfg: Any) -> Any:
    from worldforge.adapters.observer_memory import MemoryObserver

    return MemoryObserver(buffer_size=cfg.trace_buffer)


def _observer_otlp(cfg: Any) -> Any:
    from worldforge.adapters.observer_otlp import OtlpObserver

    return OtlpObserver(buffer_size=cfg.trace_buffer)


def _chunker_recursive(cfg: Any) -> Any:
    from worldforge.adapters.chunker_recursive import RecursiveCharChunker

    return RecursiveCharChunker(chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)


def _chunker_semantic(cfg: Any) -> Any:
    from worldforge.adapters.chunker_semantic import SemanticChunker

    return SemanticChunker(chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)


def _tool_mcp(cfg: Any) -> Any:
    from worldforge.adapters.tool_mcp import MCPToolSource

    return MCPToolSource(endpoint=cfg.api_base)


EMBEDDERS: dict[str, Callable[[Any], Any]] = {
    "hashing": _embedder_hashing,
    "fastembed": _embedder_fastembed,
    "openai": _embedder_openai,
}
VECTORSTORES: dict[str, Callable[[Any], Any]] = {
    "memory": _vectorstore_memory,
    "faiss": _vectorstore_faiss,
    "qdrant": _vectorstore_qdrant,
}
LEXICALS: dict[str, Callable[[Any], Any]] = {
    "bm25": _lexical_bm25,
    "elastic": _lexical_elastic,
}
RERANKERS: dict[str, Callable[[Any], Any]] = {
    "lexical": _reranker_lexical,
    "crossencoder": _reranker_crossencoder,
}
LLMS: dict[str, Callable[[Any], Any]] = {
    "mock": _llm_mock,
    "openai": _llm_openai,
    "ollama": _llm_ollama,
    "llamacpp": _llm_llamacpp,
}
DOCSTORES: dict[str, Callable[[Any], Any]] = {
    "memory": _docstore_memory,
    "sqlite": _docstore_sqlite,
    "postgres": _docstore_postgres,
}
OBSERVERS: dict[str, Callable[[Any], Any]] = {
    "memory": _observer_memory,
    "otlp": _observer_otlp,
}
CHUNKERS: dict[str, Callable[[Any], Any]] = {
    "recursive": _chunker_recursive,
    "semantic": _chunker_semantic,
}
TOOLS: dict[str, Callable[[Any], Any]] = {
    "mcp": _tool_mcp,
}

MOCK_LLM_TYPE: tuple[type, ...] = ()
DEFAULTS = {
    "embedder": "hashing",
    "vectorstore": "memory",
    "lexical": "bm25",
    "reranker": "lexical",
    "llm": "mock",
    "docstore": "memory",
    "observer": "memory",
    "chunker": "recursive",
}


def set_mock_llm_type(*types: type) -> None:
    """登记抽取式 LLM 类型，供 planner 判定是否需要降级。"""

    global MOCK_LLM_TYPE
    MOCK_LLM_TYPE = tuple(types)


__all__ = [
    "warn_once",
    "reset_warnings",
    "mark_degraded",
    "clear_degraded",
    "degraded",
    "pick",
    "set_mock_llm_type",
    "EMBEDDERS",
    "VECTORSTORES",
    "LEXICALS",
    "RERANKERS",
    "LLMS",
    "DOCSTORES",
    "OBSERVERS",
    "CHUNKERS",
    "TOOLS",
    "MOCK_LLM_TYPE",
    "DEFAULTS",
]
__author__ = "晨星"
