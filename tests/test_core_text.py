"""证据渲染与引用剥离测试。

作者：晨星
"""

from __future__ import annotations

from worldforge.core.text import parse_evidence_lines, render_evidence, single_line, strip_citations
from worldforge.types import Evidence


def test_evidence_render_parse_round_trip_keeps_block_count() -> None:
    evidence = [
        Evidence(chunk_id="doc-a#0000", doc_id="doc-a", text="第一行\n第二行", score=1.0),
        Evidence(chunk_id="doc-b#0001", doc_id="doc-b", text="第三行\r\n\r\n第四行", score=0.5),
    ]
    rendered = render_evidence(evidence)
    parsed = parse_evidence_lines(rendered)
    assert len(parsed) == len(evidence)
    assert all("\n" not in block and "\r" not in block for block in parsed)
    assert "第一行 第二行" in parsed[0]


def test_single_line_folds_all_whitespace() -> None:
    assert single_line(" a\n\tb  c ") == "a b c"


def test_strip_citations_covers_supported_forms() -> None:
    wide = chr(0x3010) + "3" + chr(0x3011)
    text = f"答案 [doc#2] [7] {wide} (依据: 文档) (来源: 检索) (source: result)"
    assert strip_citations(text) == "答案"


__author__ = "晨星"
