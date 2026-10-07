"""中英文混合文本的确定性分词与标点归一。

作者：晨星
"""

from __future__ import annotations

import re
import unicodedata

_FULLWIDTH_PUNCTUATION = frozenset({0xFF1F, 0xFF01, 0x3002, 0xFF0C, 0xFF1A, 0xFF1B})
_WORD_RE = re.compile(r"[a-z0-9_]+")


def normalize_text(text: str) -> str:
    """归一化宽字符、大小写与句读，同时保留算术运算符。"""

    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    without_punctuation = "".join(" " if ord(char) in _FULLWIDTH_PUNCTUATION else char for char in text)
    normalized = unicodedata.normalize("NFKC", without_punctuation).casefold()
    return " ".join(normalized.split())


def tokenize(text: str) -> list[str]:
    """切分英文单词和数字，并将连续中文字符切成字符二元组。"""

    normalized = normalize_text(text)
    tokens: list[str] = []
    chinese_run: list[str] = []
    latin_run: list[str] = []

    def flush_chinese() -> None:
        if not chinese_run:
            return
        if len(chinese_run) == 1:
            tokens.append(chinese_run[0])
        else:
            tokens.extend("".join(chinese_run[index : index + 2]) for index in range(len(chinese_run) - 1))
        chinese_run.clear()

    def flush_latin() -> None:
        if not latin_run:
            return
        tokens.extend(_WORD_RE.findall("".join(latin_run)))
        latin_run.clear()

    for char in normalized:
        codepoint = ord(char)
        if _is_cjk(codepoint):
            flush_latin()
            chinese_run.append(char)
        else:
            flush_chinese()
            latin_run.append(char)
    flush_chinese()
    flush_latin()
    return tokens


def _is_cjk(codepoint: int) -> bool:
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


__all__ = ["normalize_text", "tokenize"]
__author__ = "晨星"
