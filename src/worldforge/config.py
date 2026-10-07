"""环境变量配置解析与边界校验。

作者：晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Mapping

from worldforge.errors import ConfigError


@dataclass(frozen=True, slots=True)
class Config:
    """WorldForge 的不可变运行配置。"""

    seed: int = 1337
    embed_dim: int = 256
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 20
    rerank_top_k: int = 5
    fusion: str = "rrf"
    rrf_k: int = 60
    hybrid: bool = True
    verify: bool = True
    bilingual: bool = True
    log_level: str = "INFO"
    trace_buffer: int = 1000
    host: str = "127.0.0.1"
    port: int = 8000
    strict: bool = False
    max_chunks: int = 5000
    embedder: str = "hashing"
    vectorstore: str = "memory"
    lexical: str = "bm25"
    reranker: str = "lexical"
    llm: str = "mock"
    docstore: str = "memory"
    observer: str = "memory"
    tools: str = "builtin"
    planner: str = "rule"
    chunker: str = "recursive"
    server: str = "stdlib"
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "worldforge-mock-1"
    vector_url: str = ""
    api_base: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    sqlite_path: str = ":memory:"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """从环境映射解析配置，拒绝未知键、别名冲突及非法值。"""

        source = os.environ if env is None else env
        known_suffixes = {field.name.upper() for field in fields(cls)}
        backend_fields = {
            "EMBEDDER",
            "VECTORSTORE",
            "LEXICAL",
            "RERANKER",
            "LLM",
            "DOCSTORE",
            "OBSERVER",
            "TOOLS",
            "PLANNER",
            "CHUNKER",
            "SERVER",
        }
        prefixes = ("WF_", "WORLDFORGE_")
        unknown = []
        for key in source:
            if key.startswith("WORLDFORGE_BACKEND_"):
                suffix = key.removeprefix("WORLDFORGE_BACKEND_")
                if suffix not in backend_fields:
                    unknown.append(key)
            elif key.startswith(prefixes):
                prefix = "WF_" if key.startswith("WF_") else "WORLDFORGE_"
                if key.removeprefix(prefix) not in known_suffixes:
                    unknown.append(key)
        if unknown:
            raise ConfigError(f"未知配置变量: {', '.join(sorted(unknown))}")

        raw: dict[str, str] = {}
        for suffix in sorted(known_suffixes):
            aliases = [f"WORLDFORGE_{suffix}", f"WF_{suffix}"]
            if suffix in backend_fields:
                aliases.insert(1, f"WORLDFORGE_BACKEND_{suffix}")
            values: list[tuple[str, str]] = []
            for name in aliases:
                if name not in source:
                    continue
                value = source[name]
                if not isinstance(value, str):
                    raise ConfigError(f"{name} 必须是字符串")
                values.append((name, value.strip()))
            distinct = {value.casefold() if suffix != "HOST" else value for _, value in values}
            if len(distinct) > 1:
                names = ", ".join(name for name, _ in values)
                raise ConfigError(f"配置别名冲突: {names}")
            if values:
                raw[suffix.lower()] = values[0][1]

        integer_fields = {
            "seed",
            "embed_dim",
            "chunk_size",
            "chunk_overlap",
            "top_k",
            "rerank_top_k",
            "rrf_k",
            "trace_buffer",
            "port",
            "max_chunks",
        }
        boolean_fields = {"hybrid", "verify", "bilingual", "strict"}
        lowercase_fields = {
            "fusion",
            "embedder",
            "vectorstore",
            "lexical",
            "reranker",
            "llm",
            "docstore",
            "observer",
            "tools",
            "planner",
            "chunker",
            "server",
        }
        parsed: dict[str, object] = {}
        for name, value in raw.items():
            # 空串视为未设置：CI 与容器里 `FOO=` 这种写法太常见，静默当默认值
            # 比抛错更符合直觉，也让别名冲突检测不会被空值污染
            if value is None or not value.strip():
                continue
            if name in integer_fields:
                try:
                    parsed[name] = int(value)
                except ValueError as exc:
                    raise ConfigError(f"{name} 必须是整数") from exc
            elif name in boolean_fields:
                parsed[name] = _parse_switch(name, value)
            elif name == "log_level":
                parsed[name] = value.upper()
            elif name in lowercase_fields:
                parsed[name] = value.lower()
            else:
                parsed[name] = value

        config = cls(**parsed)
        config._validate()
        return config

    def _validate(self) -> None:
        """验证数值关系和枚举范围。"""

        positive = {
            "embed_dim": self.embed_dim,
            "chunk_size": self.chunk_size,
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k,
            "rrf_k": self.rrf_k,
            "trace_buffer": self.trace_buffer,
            "max_chunks": self.max_chunks,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ConfigError(f"{name} 必须大于 0")
        if self.seed < 0:
            raise ConfigError("seed 不能为负数")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ConfigError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
        if self.rerank_top_k > self.top_k:
            raise ConfigError("rerank_top_k 不能大于 top_k")
        if not 1 <= self.port <= 65535:
            raise ConfigError("port 必须位于 1 到 65535")
        if not self.host:
            raise ConfigError("host 不能为空")

        choices = {
            "fusion": (self.fusion, {"rrf", "weighted"}),
            "embedder": (self.embedder, {"hashing", "fastembed", "openai"}),
            "vectorstore": (self.vectorstore, {"memory", "faiss", "qdrant"}),
            "lexical": (self.lexical, {"bm25", "elastic"}),
            "reranker": (self.reranker, {"lexical", "crossencoder", "cohere"}),
            "llm": (self.llm, {"mock", "openai", "ollama", "llamacpp"}),
            "docstore": (self.docstore, {"memory", "sqlite", "postgres"}),
            "observer": (self.observer, {"memory", "otlp"}),
            "tools": (self.tools, {"builtin", "mcp"}),
            "planner": (self.planner, {"rule", "react", "llm"}),
            "chunker": (self.chunker, {"recursive", "semantic"}),
            "server": (self.server, {"stdlib", "uvicorn"}),
            "log_level": (self.log_level, {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                options = ", ".join(sorted(allowed))
                raise ConfigError(f"{name} 必须是以下值之一: {options}")


def _parse_switch(name: str, value: str) -> bool:
    normalized = value.lower()
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    raise ConfigError(f"{name} 必须是 on 或 off")


__all__ = ["Config"]
__author__ = "晨星"
