"""分层依赖静态检查：禁止反向 import。"""

from __future__ import annotations

import ast
from pathlib import Path
from sys import stdlib_module_names

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "worldforge"

# layer -> 层级序号。数字越小越底层。
LAYERS: dict[str, int] = {
    "core": 1,
    "ports": 1,
    "adapters": 2,
    "retrieval": 3,
    "agents": 3,
    "observability": 3,
    "eval": 3,
    "serving": 4,
}

# 根级模块不是分层目录，单独校验它们不被上层反向依赖
ROOT_MODULES: dict[str, int] = {"types": 0, "errors": 0, "config": 0, "pipeline": 5}

FORBIDDEN: dict[str, set[str]] = {
    "core": {"adapters", "retrieval", "agents", "serving", "eval", "pipeline"},
    "ports": {"adapters", "retrieval", "agents", "serving", "eval", "pipeline"},
    "types": {"adapters", "retrieval", "agents", "serving", "eval", "pipeline"},
    "errors": {"adapters", "retrieval", "agents", "serving", "eval", "pipeline"},
    "config": {"adapters", "retrieval", "agents", "serving", "eval", "pipeline"},
    "adapters": {"retrieval", "agents", "serving", "eval", "pipeline"},
    "retrieval": {"serving", "pipeline"},
    "agents": {"serving", "pipeline"},
    "eval": {"serving", "pipeline"},
    "serving": {"pipeline"},
}


def _layer_of(path: Path) -> str | None:
    try:
        return path.relative_to(SRC).parts[0]
    except ValueError:
        return None


def _imported_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    layers: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "worldforge" and parts[1] in LAYERS:
                layers.add(parts[1])
    return layers


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_import_upstream(layer):
    violations: list[str] = []
    for path in (SRC / layer).rglob("*.py"):
        if "kernels" in path.parts:
            continue
        illegal = _imported_layers(path) & FORBIDDEN[layer]
        if illegal:
            violations.append(f"{path.name} -> {sorted(illegal)}")
    assert not violations, f"{layer} 出现反向依赖: {violations}"


def test_every_layer_directory_exists():
    for layer in LAYERS:
        assert (SRC / layer).is_dir(), f"缺少分层目录 {layer}"
    for module in ROOT_MODULES:
        assert (SRC / f"{module}.py").is_file(), f"缺少根级模块 {module}.py"


def test_core_kernels_import_nothing_from_upper_layers():
    kernels = SRC / "core" / "kernels"
    for path in kernels.rglob("*.py"):
        illegal = _imported_layers(path) & {"adapters", "retrieval", "agents", "serving", "eval", "pipeline"}
        assert not illegal, f"{path.name} 依赖了上层 {illegal}"


def test_no_third_party_import_in_core_and_ports():
    """核心层只能用标准库。

    用 sys.stdlib_module_names 做白名单而不是手写清单：手写清单会漏掉新版本
    CPython 新增的模块，也会把标准库误判成第三方。
    """

    stdlib = set(stdlib_module_names) | {"__future__", "worldforge"}
    # core/accel.py 是唯一允许的可选加速点，单独由 test_accel_numpy_import_is_optional 守护
    allowed_optional = {"accel.py"}
    offenders: list[str] = []
    for layer in ("core", "ports"):
        for path in (SRC / layer).rglob("*.py"):
            if path.name in allowed_optional:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name not in stdlib:
                        offenders.append(f"{path.relative_to(SRC)} -> {name}")
    assert not offenders, f"核心层引入第三方依赖: {offenders}"


def test_accel_numpy_import_is_optional():
    """core/accel.py 允许 numpy，但必须包在 try/except ImportError 里。"""

    source = (SRC / "core" / "accel.py").read_text(encoding="utf-8")
    assert "import numpy" in source
    assert "except ImportError" in source
