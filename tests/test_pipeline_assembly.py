"""管道装配、环境变量别名与配置冲突。"""

from __future__ import annotations

import pytest

from worldforge.config import Config
from worldforge.errors import ConfigError
from worldforge.pipeline import PLANNERS, build_pipeline


def test_canonical_env_prefix():
    cfg = Config.from_env({"WORLDFORGE_TOP_K": "7"})
    assert cfg.top_k == 7


def test_short_alias_is_equivalent():
    assert Config.from_env({"WF_TOP_K": "9"}).top_k == 9


def test_backend_alias_is_equivalent():
    assert Config.from_env({"WORLDFORGE_BACKEND_VECTORSTORE": "memory"}).vectorstore == "memory"


def test_conflicting_alias_raises():
    with pytest.raises(ConfigError):
        Config.from_env({"WORLDFORGE_TOP_K": "5", "WF_TOP_K": "6"})


def test_matching_alias_does_not_raise():
    assert Config.from_env({"WORLDFORGE_TOP_K": "5", "WF_TOP_K": "5"}).top_k == 5


def test_empty_value_falls_back_to_default():
    assert Config.from_env({"WORLDFORGE_TOP_K": ""}).top_k == Config().top_k


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError):
        Config.from_env({"WORLDFORGE_NOT_A_FIELD": "1"})


def test_invalid_value_is_rejected():
    with pytest.raises(ConfigError):
        Config.from_env({"WORLDFORGE_EMBED_DIM": "0"})


def test_planner_factory_returns_instance():
    planner = PLANNERS["rule"]()
    assert hasattr(planner, "plan") and hasattr(planner, "replan")


def test_pipeline_parts_are_all_ports():
    from worldforge.ports.chunker import Chunker
    from worldforge.ports.docstore import DocStore
    from worldforge.ports.embedder import Embedder
    from worldforge.ports.lexical import LexicalIndex
    from worldforge.ports.llm import LLM
    from worldforge.ports.observer import Observer
    from worldforge.ports.reranker import Reranker
    from worldforge.ports.vectorstore import VectorStore

    parts = build_pipeline(Config()).parts
    assert isinstance(parts["embedder"], Embedder)
    assert isinstance(parts["vectorstore"], VectorStore)
    assert isinstance(parts["lexical"], LexicalIndex)
    assert isinstance(parts["reranker"], Reranker)
    assert isinstance(parts["llm"], LLM)
    assert isinstance(parts["docstore"], DocStore)
    assert isinstance(parts["observer"], Observer)
    assert isinstance(parts["chunker"], Chunker)


def test_sqlite_docstore_opt_in():
    pipeline = build_pipeline(Config(docstore="sqlite"))
    assert pipeline.config.docstore == "sqlite"
    pipeline.ingest("d", "内容")
    assert pipeline.parts["docstore"].get_document("d") is not None


def test_reset_clears_all_backends():
    pipeline = build_pipeline(Config())
    pipeline.ingest("d1", "内容")
    pipeline.reset()
    assert pipeline.parts["docstore"].count_chunks() == 0
    assert pipeline.parts["vectorstore"].count() == 0
