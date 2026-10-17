"""语义分块后端（可选 extras: chunk-semantic）。

作者：晨星
"""

from __future__ import annotations

from worldforge.adapters.chunker_recursive import RecursiveCharChunker
from worldforge.errors import BackendUnavailable
from worldforge.types import Chunk, Document


class SemanticChunker:
    """用相邻句嵌入余弦的拐点决定切分位置。

    嵌入后端惰性导入；缺失时抛 BackendUnavailable 让 registry 回退到递归分块，
    而不是静默降级成"看起来能用但边界很差"的实现。
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64, embedder: object | None = None) -> None:
        self._size = chunk_size
        self._overlap = overlap
        self._fallback = RecursiveCharChunker(chunk_size=chunk_size, overlap=overlap)
        self._embedder = embedder or self._default_embedder()

    @staticmethod
    def _default_embedder() -> object:
        try:
            from fastembed import TextEmbedding  # noqa: PLC0415 - 惰性导入是本文件的核心约束
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise BackendUnavailable(f"fastembed is not installed: {exc}") from exc
        return TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = [s for s in doc.text.replace("\n", " ").split("。") if s.strip()]
        if len(sentences) < 2:
            return self._fallback.chunk(doc)
        vectors = list(self._embedder.embed([s + "。" for s in sentences]))  # type: ignore[attr-defined]
        cuts = [0]
        for i in range(1, len(sentences)):
            # 语义距离出现局部极大即视为边界，跳过边界句避免切得过碎
            distance = 1.0 - sum(a * b for a, b in zip(vectors[i - 1], vectors[i], strict=False))
            previous = 1.0 - sum(a * b for a, b in zip(vectors[i - 2], vectors[i - 1], strict=False)) if i >= 2 else -1.0
            if distance > previous and distance > 0.15:
                cuts.append(i)
        cuts.append(len(sentences))

        out: list[Chunk] = []
        ordinal = 0
        for start, end in zip(cuts, cuts[1:], strict=False):
            body = "。".join(s.strip() for s in sentences[start:end]) + "。"
            if not body.strip():
                continue
            for piece in self._hard_split(body):
                out.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{ordinal}",
                        doc_id=doc.doc_id,
                        ordinal=ordinal,
                        text=piece,
                        meta={"source": "semantic", "sentence_span": [start, end]},
                    )
                )
                ordinal += 1
        return out

    def _hard_split(self, text: str) -> list[str]:
        """语义边界产出的片段仍可能超长，交给递归分块兜底并重新编号。"""

        if len(text) <= self._size:
            return [text]
        scratch = Document(doc_id="semantic-scratch", text=text)
        return [c.text for c in self._fallback.chunk(scratch)]


__all__ = ["SemanticChunker"]
__author__ = "晨星"
