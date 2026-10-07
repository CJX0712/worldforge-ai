"""交叉编码器重排后端（可选 extras: rerank-crossencoder）。

作者：晨星
"""

from __future__ import annotations

from typing import Sequence

from worldforge.adapters.reranker_lexical import LexicalReranker
from worldforge.errors import BackendUnavailable
from worldforge.types import Hit

# 用官方 Xenova 镜像：ModelScope 的 bge-* 仓库只传 safetensors，没有 onnx 目录
_DEFAULT_MODEL = "Xenova/bge-reranker-base"


class CrossEncoderReranker:
    """bge-reranker 类交叉编码器。

    模型在第一次 rerank 时才下载，异常会发生在检索热路径上，所以这里在构造
    阶段就 warmup 一次；失败则直接抛 BackendUnavailable，让 registry 回退。
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL, docstore: object | None = None) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: PLC0415 - 惰性导入
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise BackendUnavailable(f"fastembed is not installed: {exc}") from exc
        self._model = TextCrossEncoder(model_name=model_id)
        self._fallback = LexicalReranker()
        self._docstore = docstore
        self._ready = False
        try:
            self._warmup()
        except Exception as exc:  # noqa: BLE001 - 任何后端异常都必须归一为可降级错误
            raise BackendUnavailable(f"cross encoder warmup failed: {exc}") from exc

    def _warmup(self) -> None:
        list(self._model.rerank("warmup", [("warmup", "warmup")]))
        self._ready = True

    def rerank(self, query: str, candidates: Sequence[Hit], top_k: int) -> list[Hit]:
        if not candidates:
            return []
        if not self._ready:
            return self._fallback.rerank(query, candidates, top_k)
        passages = [self._text_of(candidate) for candidate in candidates]
        ranked = list(self._model.rerank(query, passages))
        order = [int(item["index"]) for item in ranked]
        out: list[Hit] = []
        for rank, idx in enumerate(order):
            if rank >= top_k:
                break
            hit = candidates[idx]
            out.append(Hit(chunk_id=hit.chunk_id, score=float(ranked[rank]["score"]), source="cross-encoder", meta=hit.meta))
        return out

    def _text_of(self, hit: Hit) -> str:
        if self._docstore is not None:
            chunks = self._docstore.get_chunks([hit.chunk_id])  # type: ignore[attr-defined]
            if chunks:
                return chunks[0].text
        return str(hit.meta.get("text", ""))


__all__ = ["CrossEncoderReranker"]
__author__ = "晨星"
