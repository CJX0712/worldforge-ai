"""生产后端缺失依赖时必须降级而不是抛给调用方。"""

from __future__ import annotations

import warnings

import pytest

from worldforge.adapters.registry import (
    clear_degraded,
    degraded,
    pick,
    reset_warnings,
    warn_once,
)
from worldforge.config import Config
from worldforge.errors import BackendUnavailable, ConfigError


@pytest.fixture(autouse=True)
def _reset():
    clear_degraded()
    reset_warnings()
    yield
    clear_degraded()
    reset_warnings()


def test_pick_falls_back_when_backend_missing():
    def broken(_cfg):
        raise ImportError("no such package")

    table = {"broken": broken, "default": lambda cfg: "default-impl"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chosen = pick("broken", table, "default", "demo", Config())
    assert chosen == "default-impl"
    assert degraded() == ("demo",)
    assert any("fallback" in str(w.message) for w in caught)


def test_pick_warns_only_once_per_tag():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_once("same message", tag="t")
        warn_once("same message", tag="t")
    assert len(caught) == 1


def test_strict_mode_reraises_instead_of_degrading():
    def broken(_cfg):
        raise ImportError("nope")

    table = {"broken": broken, "default": lambda cfg: "d"}
    with pytest.raises(ImportError):
        pick("broken", table, "default", "demo", Config(strict=True))
    assert degraded() == ()


def test_unknown_backend_raises_config_error():
    with pytest.raises(ConfigError):
        pick("nope", {"default": lambda c: 1}, "default", "demo", Config())


def test_backend_unavailable_is_treated_as_degradable():
    def unavailable(_cfg):
        raise BackendUnavailable("service down")

    table = {"x": unavailable, "default": lambda cfg: "d"}
    assert pick("x", table, "default", "svc", Config()) == "d"
    assert degraded() == ("svc",)


def test_pipeline_builds_with_all_default_backends():
    from worldforge.pipeline import build_pipeline

    pipeline = build_pipeline(Config())
    assert pipeline.degraded == ()
    assert set(pipeline.health()["backends"]) >= {"embedder", "vectorstore", "llm", "chunker"}
