"""面向稀疏检索分支的双语词典查询扩展。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from worldforge.core.tokenizer import normalize_text, tokenize

_LEXICON: dict[str, tuple[str, ...]] = {
    "人工智能": ("artificial", "intelligence", "ai"),
    "机器学习": ("machine", "learning"),
    "深度学习": ("deep", "learning"),
    "向量": ("vector",),
    "检索": ("retrieval", "search"),
    "模型": ("model",),
    "代理": ("agent",),
    "智能体": ("agent",),
    "artificial intelligence": ("人工智能",),
    "machine learning": ("机器学习",),
    "deep learning": ("深度学习",),
    "vector": ("向量",),
    "retrieval": ("检索",),
    "search": ("检索",),
    "model": ("模型",),
    "agent": ("智能体", "代理"),
}


def expand_tokens(tokens: Sequence[str]) -> list[str]:
    """按词典扩展词元并稳定去重。"""

    expanded: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for candidate in (token, *_LEXICON.get(normalize_text(token), ())):
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def expand_query(query: str) -> str:
    """扩展查询文本；调用方只能在稀疏检索路径使用结果。"""

    normalized = normalize_text(query)
    candidates = tokenize(query)
    for phrase, translations in _LEXICON.items():
        if phrase in normalized:
            candidates.extend(translations)
    return " ".join(expand_tokens(candidates))


__all__ = ["expand_tokens", "expand_query"]
__author__ = "晨星"
