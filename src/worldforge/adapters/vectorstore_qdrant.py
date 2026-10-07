"""Qdrant 向量库后端（可选 extras: vector-qdrant）。

作者：晨星
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence
from uuid import uuid5, NAMESPACE_URL

from worldforge.adapters.vectorstore_memory import InMemoryExactIndex
from worldforge.errors import BackendUnavailable, DimensionMismatch

_CHUNK_NS = NAMESPACE_URL


def _point_id(chunk_id: str) -> str:
    return str(uuid5(_CHUNK_NS, chunk_id))


class QdrantStore(InMemoryExactIndex):
    """Qdrant 远程集合封装，HTTP 走标准库 urllib。"""

    def __init__(self, dim: int, url: str = "http://127.0.0.1:6333", model_id: str = "qdrant") -> None:
        super().__init__(dim=dim, model_id=model_id)
        if not url:
            raise BackendUnavailable("qdrant url is empty")
        self._url = url.rstrip("/")
        self._collection = "worldforge"
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        body = json.dumps({"vectors": {"size": self.dim, "distance": "Cosine"}}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._url}/collections/{self._collection}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        try:
            urllib.request.urlopen(request, timeout=10).close()
        except urllib.error.HTTPError as exc:
            # 409 = 集合已存在，属于幂等成功而非失败
            if exc.code != 409:
                raise BackendUnavailable(f"qdrant collection init failed: {exc}") from exc
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"qdrant unreachable: {exc}") from exc

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metas: Sequence[Mapping[str, Any]],
    ) -> int:
        for vector in vectors:
            self._validate_vector(vector)
        points = [
            {"id": _point_id(cid), "vector": list(map(float, v)), "payload": {**dict(m), "chunk_id": cid}}
            for cid, v, m in zip(ids, vectors, metas)
        ]
        body = json.dumps({"points": points}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._url}/collections/{self._collection}/points?wait=true",
            data=body,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        try:
            urllib.request.urlopen(request, timeout=20).close()
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"qdrant upsert failed: {exc}") from exc
        return len(ids)

    def count(self) -> int:
        request = urllib.request.Request(f"{self._url}/collections/{self._collection}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return int(json.loads(response.read().decode("utf-8"))["result"]["points_count"])
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"qdrant count failed: {exc}") from exc

    def clear(self) -> None:
        super().clear()

    def save(self, path: str) -> None:
        # 远程集合由服务端持久化，此处只记录指纹用于跨环境一致性检查
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"fingerprint": self.fingerprint, "collection": self._collection}, handle)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("fingerprint") != self.fingerprint:
            raise DimensionMismatch("index fingerprint mismatch; run scripts/reindex.py")


__all__ = ["QdrantStore"]
__author__ = "晨星"
