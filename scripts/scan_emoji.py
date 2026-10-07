"""P0 字符门禁：全仓扫描禁用字符。

emoji、Unicode 框线字符与非 ASCII 箭头在 Windows 控制台、GBK 代码页与
Markdown 渲染里都会变成乱码或错位。源码里一律不得出现，检测逻辑本身只允许
用 ord() 码点范围表达规则。

作者：晨星
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 用码点区间而非符号字面量，否则本脚本自己就会触发门禁
_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F300, 0x1FAFF),
    (0x1F000, 0x1F2FF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
    (0xFE00, 0xFE0F),
    (0x1F900, 0x1F9FF),
    (0x2705, 0x2705),
    (0x274C, 0x274C),
)
_BOX_RANGES: tuple[tuple[int, int], ...] = ((0x2500, 0x257F),)
_ARROW_CHARS: frozenset[int] = frozenset({0x2192, 0x2190, 0x2191, 0x2193, 0x21D2, 0x21D4})

TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".example", ""}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "_deps", ".ruff_cache"}


def scan_text(name: str, text: str) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for char in line:
            code = ord(char)
            if any(low <= code <= high for low, high in _EMOJI_RANGES):
                findings.append((name, line_no, f"emoji/symbol U+{code:04X}"))
            elif any(low <= code <= high for low, high in _BOX_RANGES):
                findings.append((name, line_no, f"box drawing U+{code:04X}"))
            elif code in _ARROW_CHARS:
                findings.append((name, line_no, f"unicode arrow U+{code:04X}"))
    return findings


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "Makefile", ".env.example"}:
            yield path


def main() -> int:
    findings: list[tuple[str, int, str]] = []
    scanned = 0
    for path in iter_files(ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if text.startswith("\ufeff"):
            findings.append((str(path.relative_to(ROOT)), 1, "UTF-8 BOM"))
        scanned += 1
        findings.extend(scan_text(str(path.relative_to(ROOT)), text))
    if findings:
        print(f"P0 gate FAILED: {len(findings)} finding(s) in {scanned} file(s)")
        for name, line_no, reason in findings[:50]:
            print(f"  {name}:{line_no} {reason}")
        return 1
    print(f"P0 gate OK: {scanned} file(s) scanned, 0 findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
__author__ = "晨星"
