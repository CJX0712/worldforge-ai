"""文本嵌入 Port 契约。

作者：晨星
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """批量及单条文本嵌入接口。"""

    @property
    def dim(self) -> int:
        """返回稳定向量维度。"""
        ...

    @property
    def model_id(self) -> str:
        """返回索引指纹使用的稳定模型标识。"""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """返回已经 L2 归一化的文本向量。"""
        ...

    def embed_one(self, text: str) -> list[float]:
        """嵌入单条文本。"""
        ...


__all__ = ["Embedder"]
__author__ = "晨星"
