"""基于 BLAKE2b 的确定性有符号散列嵌入。

作者：晨星
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from worldforge.core.mathx import l2_normalize
from worldforge.core.tokenizer import tokenize

_PERSON = b"worldforge-v1"


def signed_hash(token: str, dim: int) -> tuple[int, float]:
    """把一个词元映射到稳定维度与正负符号。"""

    if dim <= 0:
        raise ValueError("dim 必须大于 0")
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16, person=_PERSON).digest()
    index = int.from_bytes(digest[:8], "little") % dim
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def hash_embed(text: str, dim: int = 256) -> list[float]:
    """对文本执行有符号特征散列并返回 L2 归一化向量。"""

    if dim <= 0:
        raise ValueError("dim 必须大于 0")
    vector = [0.0] * dim
    for token in tokenize(text):
        index, sign = signed_hash(token, dim)
        vector[index] += sign
    return l2_normalize(vector)


def hash_embed_many(texts: Sequence[str], dim: int = 256) -> list[list[float]]:
    """批量生成散列嵌入，保持输入顺序。"""

    return [hash_embed(text, dim) for text in texts]


__all__ = ["signed_hash", "hash_embed", "hash_embed_many"]
__author__ = "晨星"
