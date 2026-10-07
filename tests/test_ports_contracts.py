"""十个运行时可检查 Port 的结构契约测试。

作者：晨星
"""

from __future__ import annotations

import pytest

from worldforge.config import Config
from worldforge.errors import ConfigError
from worldforge.ports import Chunker, DocStore, Embedder, LexicalIndex, LLM, Observer, Planner, Reranker, Tool, VectorStore
from tests.fakes import (
    FakeChunker,
    FakeDocStore,
    FakeEmbedder,
    FakeLexicalIndex,
    FakeLLM,
    FakePlanner,
    FakeReranker,
    FakeTool,
    FakeVectorStore,
    RecordingObserver,
)


def test_all_protocols_are_runtime_checkable() -> None:
    pairs = [
        (FakeLLM(), LLM),
        (FakeEmbedder(), Embedder),
        (FakeVectorStore(), VectorStore),
        (FakeLexicalIndex(), LexicalIndex),
        (FakeReranker(), Reranker),
        (FakeDocStore(), DocStore),
        (RecordingObserver(), Observer),
        (FakeTool(), Tool),
        (FakePlanner(), Planner),
        (FakeChunker(), Chunker),
    ]
    assert all(isinstance(instance, protocol) for instance, protocol in pairs)


def test_missing_method_breaks_structural_contract() -> None:
    class Incomplete:
        marker = "missing-port-methods"

    assert not isinstance(Incomplete(), LLM)
    assert not isinstance(Incomplete(), VectorStore)


def test_config_parses_and_validates_environment() -> None:
    config = Config.from_env({"WF_TOP_K": "8", "WF_RERANK_TOP_K": "4", "WF_HYBRID": "off"})
    assert (config.top_k, config.rerank_top_k, config.hybrid) == (8, 4, False)
    with pytest.raises(ConfigError):
        Config.from_env({"WF_UNKNOWN": "value"})
    with pytest.raises(ConfigError):
        Config.from_env({"WF_CHUNK_SIZE": "32", "WF_CHUNK_OVERLAP": "32"})


__author__ = "晨星"
