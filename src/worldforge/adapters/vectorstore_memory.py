"""精确余弦内存向量索引（默认后端）。

作者：晨星
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

from worldforge.core import accel
from worldforge.core.mathx import cosine_similarity, dot, l2_norm
from worldforge.core.sortutil import stable_sort
from worldforge.errors import DimensionMismatch, VectorStoreError
from worldforge.types import Hit

_DOC_PREFIX_SEPARATOR = "#"


class InMemoryExactIndex:
    """暴力精确余弦检索，结果等于余弦定义本身。"""

    name = "memory"

    def __init__(self, dim: int, model_id: str = "hashing") -> None:
        if dim <= 0:
            raise ValueError("dim 必须大于 0")
        self._dim = int(dim)
        self._model_id = model_id
        self._vectors: dict[str, list[float]] = {}
        self._metas: dict[str, dict[str, object]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"{self._model_id}:{self._dim}"

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metas: Sequence[Mapping[str, Any]],
    ) -> int:
        if not (len(ids) == len(vectors) == len(metas)):
            raise ValueError("ids、vectors、metas 长度必须相等")
        for chunk_id, vector in zip(ids, vectors):
            self._validate_vector(vector)
            self._vectors[chunk_id] = [float(value) for value in vector]
        for chunk_id, meta in zip(ids, metas):
            self._metas[chunk_id] = dict(meta)
        return len(ids)

    def delete(self, ids: Sequence[str]) -> int:
        removed = 0
        for chunk_id in ids:
            if chunk_id in self._vectors:
                del self._vectors[chunk_id]
                self._metas.pop(chunk_id, None)
                removed += 1
        return removed

    def drop_document(self, doc_id: str) -> int:
        prefix = f"{doc_id}{_DOC_PREFIX_SEPARATOR}"
        targets = [chunk_id for chunk_id in self._vectors if chunk_id.startswith(prefix)]
        return self.delete(targets)

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[Hit]:
        if top_k < 0:
            raise ValueError("top_k 不能为负数")
        self._validate_vector(vector)
        if not self._vectors:
            return []
        ordered_ids = sorted(self._vectors)
        matrix = [self._vectors[chunk_id] for chunk_id in ordered_ids]
        if accel.numpy_available():
            scores = accel.cosine_scores(vector, matrix)
        else:
            scores = [cosine_similarity(vector, row) for row in matrix]
        hits: list[Hit] = []
        for chunk_id, score in zip(ordered_ids, scores):
            meta = self._metas.get(chunk_id, {})
            if where and not _matches(meta, where):
                continue
            hits.append(Hit(chunk_id=chunk_id, score=float(score), source="dense", meta=dict(meta)))
        ordered = stable_sort(hits)
        return ordered[:top_k] if top_k else []

    def count(self) -> int:
        return len(self._vectors)

    def clear(self) -> None:
        self._vectors.clear()
        self._metas.clear()

    def save(self, path: str) -> None:
        payload = {
            "fingerprint": self.fingerprint,
            "vectors": {key: value for key, value in sorted(self._vectors.items())},
            "metas": {key: value for key, value in sorted(self._metas.items())},
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)

    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise VectorStoreError(f"索引文件无法读取: {exc}") from exc
        if payload.get("fingerprint") != self.fingerprint:
            raise DimensionMismatch(
                f"索引指纹 {payload.get('fingerprint')} 与当前 {self.fingerprint} 不符，需要重建索引"
            )
        self._vectors = {key: [float(value) for value in vector] for key, vector in payload["vectors"].items()}
        self._metas = {key: dict(value) for key, value in payload["metas"].items()}

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dim:
            raise DimensionMismatch(f"向量维度 {len(vector)} 与索引维度 {self._dim} 不符")
        if any(not math.isfinite(float(value)) for value in vector):
            raise VectorStoreError("向量包含非有限数值")


def _matches(meta: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    return all(meta.get(key) == value for key, value in where.items())


def exact_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """供测试对照的余弦定义实现。"""

    denominator = l2_norm(left) * l2_norm(right)
    if denominator == 0.0:
        return 0.0
    return dot(left, right) / denominator


__all__ = ["InMemoryExactIndex", "exact_cosine"]
__author__ = "晨星"
