"""测试路径、固定种子及隔离环境夹具。

作者：晨星
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from worldforge.core.seeding import seed_everything


@pytest.fixture(autouse=True)
def fixed_seed_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个测试使用相同随机种子且不继承 WorldForge 配置。"""

    for key in tuple(os.environ):
        if key.startswith(("WF_", "WORLDFORGE_")):
            monkeypatch.delenv(key, raising=False)
    seed_everything(1337)


@pytest.fixture
def temp_env() -> dict[str, str]:
    """返回可独立修改的空环境映射。"""

    return {}


__author__ = "晨星"
