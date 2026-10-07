"""混合分词与全角标点归一测试。

作者：晨星
"""

from __future__ import annotations

from worldforge.core.bilingual import expand_query
from worldforge.core.tokenizer import normalize_text, tokenize


def test_mixed_chinese_english_tokenization() -> None:
    assert tokenize("Hello 人工智能 WORLD") == ["hello", "人工", "工智", "智能", "world"]


def test_fullwidth_punctuation_is_removed() -> None:
    punctuation = "".join(chr(codepoint) for codepoint in (0xFF1F, 0xFF01, 0x3002, 0xFF0C, 0xFF1A, 0xFF1B))
    normalized = normalize_text(f"query{punctuation}next")
    assert normalized == "query next"


def test_calculator_operators_are_preserved() -> None:
    expression = "1+2-3*4/5"
    assert normalize_text(expression) == expression
    assert all(operator in normalize_text(expression) for operator in "+-*/")


def test_bilingual_expansion_is_deterministic() -> None:
    expanded = expand_query("人工智能")
    assert expanded == expand_query("人工智能")
    assert "artificial" in expanded.split()
    assert "intelligence" in expanded.split()


__author__ = "晨星"
