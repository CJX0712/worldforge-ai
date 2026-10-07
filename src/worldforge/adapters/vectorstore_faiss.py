"""FAISS 向量库后端（可选 extras: vector-faiss）。

作者：晨星
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldforge.adapters.vectorstore_memory import InMemoryExactIndex
from worldforge.errors import BackendUnavailable, DimensionMismatch


class FaissStore(InMemoryExactIndex):
    """FAISS IndexFlatIP 精确内积索引（向量已 L2 归一化，等价于余弦）。

    元数据与 id 映射单独用 JSON 落盘：FAISS 只存向量，把业务数据塞进索引会让
    指纹校验与重排都变脆。
    """

    def __init__(self, dim: int, model_id: str = "faiss") -> None:
        super().__init__(dim=dim, model_id=model_id)
        try:
            import faiss  # noqa: PLC0415 - 惰性导入是本文件的核心约束
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise BackendUnavailable(f"faiss is not installed: {exc}") from exc
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metas: Sequence[Mapping[str, Any]],
    ) -> int:
        for vector in vectors:
            self._validate_vector(vector)
        payload = [list(map(float, v)) for v in vectors]
        self._index.add(payload)
        self._ids.extend(ids)
        self._vectors.update({cid: list(map(float, v)) for cid, v in zip(ids, vectors)})
        self._metas.update({cid: dict(m) for cid, m in zip(ids, metas)})
        return len(ids)

    def count(self) -> int:
        return int(self._index.ntotal)

    def clear(self) -> None:
        self._index.reset()
        self._ids.clear()
        super().clear()

    def save(self, path: str) -> None:
        self._faiss.write_index(self._index, str(Path(path) / "vectors.faiss"))
        sidecar = Path(path) / "vectors_meta.json"
        sidecar.write_text(
            json.dumps(
                {"ids": self._ids, "metas": self._metas, "dim": self.dim, "fingerprint": self.fingerprint},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self, path: str) -> None:
        meta_path = Path(path) / "vectors_meta.json"
        if not meta_path.exists():
            raise BackendUnavailable(f"index sidecar missing: {meta_path}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload["fingerprint"] != self.fingerprint:
            raise DimensionMismatch("index fingerprint mismatch; run scripts/reindex.py")
        self._index = self._faiss.read_index(str(Path(path) / "vectors.faiss"))
        self._ids = list(payload["ids"])
        self._metas = dict(payload["metas"])
        self._vectors = {cid: [] for cid in self._ids}


__all__ = ["FaissStore"]
__author__ = "晨星"
