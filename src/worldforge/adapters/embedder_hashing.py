"""BLAKE2b 有符号散列嵌入（默认后端）。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from worldforge.core.hash_embed import hash_embed
from worldforge.core.mathx import l2_norm

_MAX_TEXT = 100_000


class HashingEmbedder:
    """零依赖的确定性文本嵌入后端。"""

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须大于 0")
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return f"hashing:d{self._dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("待嵌入文本必须是字符串")
            if len(text) > _MAX_TEXT:
                raise ValueError("待嵌入文本超过长度上限")
            vectors.append(hash_embed(text, self._dim))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def assert_real_implementation(embedder: HashingEmbedder) -> None:
    """自检：不同文本必须产生不同向量且范数为 1。"""

    first = embedder.embed_one("检索层怎么分层")
    second = embedder.embed_one("向量库怎么选型")
    if first == second:
        raise ValueError("嵌入实现退化为常量向量")
    if abs(l2_norm(first) - 1.0) > 1e-12:
        raise ValueError("嵌入向量未归一化")


__all__ = ["HashingEmbedder", "assert_real_implementation"]
__author__ = "晨星"
