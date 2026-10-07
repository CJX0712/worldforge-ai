"""证据单行化、渲染与引用标记剥离。

作者：晨星
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from worldforge.types import Evidence

_WHITESPACE_RE = re.compile(r"\s+")
_BRACKET_CITATION_RE = re.compile(r"\[[^\]\n]*#\d+\]|\[\d+\]")
_WIDE_BRACKET_CITATION_RE = re.compile(re.escape(chr(0x3010)) + r"\d+" + re.escape(chr(0x3011)))
_PAREN_OPEN = rf"(?:\(|{re.escape(chr(0xFF08))})"
_PAREN_CLOSE = rf"(?:\)|{re.escape(chr(0xFF09))})"
_COLON = rf"[:{re.escape(chr(0xFF1A))}]"
_PAREN_CITATION_RE = re.compile(
    _PAREN_OPEN + rf"(?:依据|来源|source)\s*{_COLON}\s*[^)" + re.escape(chr(0xFF09)) + r"]*" + _PAREN_CLOSE,
    re.IGNORECASE,
)


def single_line(text: str) -> str:
    """把任意空白折叠为单个空格。"""

    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    return _WHITESPACE_RE.sub(" ", text).strip()


def render_evidence(evidences: Sequence[Evidence]) -> str:
    """把每个证据块渲染成恰好一行。"""

    return "\n".join(f"[{evidence.chunk_id}] {single_line(evidence.text)}" for evidence in evidences)


def parse_evidence_lines(rendered: str) -> list[str]:
    """按行解析证据渲染结果并忽略空行。"""

    if not isinstance(rendered, str):
        raise TypeError("rendered 必须是字符串")
    return [line for line in rendered.splitlines() if line.strip()]


def strip_citations(text: str) -> str:
    """剥离内部引用标记并规范剩余空白。"""

    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    stripped = _BRACKET_CITATION_RE.sub(" ", text)
    stripped = _WIDE_BRACKET_CITATION_RE.sub(" ", stripped)
    stripped = _PAREN_CITATION_RE.sub(" ", stripped)
    return single_line(stripped)


__all__ = ["single_line", "render_evidence", "parse_evidence_lines", "strip_citations"]
__author__ = "晨星"
