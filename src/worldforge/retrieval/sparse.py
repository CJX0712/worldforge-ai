"""稀疏召回编排。

作者：晨星
"""

from __future__ import annotations

from worldforge.core.bilingual import expand_query
from worldforge.types import Hit


class SparseRetriever:
    """BM25 倒排召回，可选接入双语查询扩展。

    不变量
        I1  双语扩展**只作用于本分支**。同时喂给稠密分支会毁掉两路互补性：
            稠密向量本来靠语义相近度取胜，人为塞入跨语言词会把它拉向词面匹配
    """

    def __init__(self, lexical: object, bilingual: bool = False) -> None:
        self._lexical = lexical
        self._bilingual = bilingual
        self.last_expanded = ""

    def search(self, query: str, top_k: int) -> list[Hit]:
        prepared = expand_query(query) if self._bilingual else query
        self.last_expanded = prepared
        return list(self._lexical.search(prepared, top_k))  # type: ignore[attr-defined]


__all__ = ["SparseRetriever"]
__author__ = "晨星"
