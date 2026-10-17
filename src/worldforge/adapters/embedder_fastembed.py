"""fastembed 嵌入后端（可选 extras: vector-embed-fastembed）。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from worldforge.core.mathx import l2_normalize


class FastEmbedEmbedder:
    """基于 ONNX Runtime 的 CPU 嵌入实现。

    依赖必须在工厂函数内惰性导入；缺失时抛 ImportError，由 registry 回退到
    默认的 HashingEmbedder，而不是把异常抛给调用方。
    """

    def __init__(self, model_id: str = "BAAI/bge-small-zh-v1.5", dim: int = 512) -> None:
        from fastembed import TextEmbedding  # noqa: PLC0415 - 惰性导入是本文件的核心约束

        self._model = TextEmbedding(model_name=model_id)
        self._model_id = model_id
        self._dim = dim
        probe = next(iter(self._model.embed(["dimension probe"])))
        self._dim = len(probe)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [l2_normalize(vec) for vec in self._model.embed(list(texts))]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


__all__ = ["FastEmbedEmbedder"]
__author__ = "晨星"
