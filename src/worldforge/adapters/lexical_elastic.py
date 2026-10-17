"""Elasticsearch 稀疏索引后端（可选 extras: lexical-elastic）。

作者：晨星
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from worldforge.adapters.lexical_bm25 import BM25Index
from worldforge.errors import BackendUnavailable
from worldforge.types import Chunk, Hit

_INDEX = "worldforge-chunks"


class ElasticLexical:
    """把 BM25 查询下推给 Elasticsearch。"""

    def __init__(self, url: str = "http://127.0.0.1:9200", timeout: float = 10.0) -> None:
        if not url:
            raise BackendUnavailable("elasticsearch url is empty")
        self._url = url.rstrip("/")
        self._timeout = timeout
        self._local = BM25Index()

    def index(self, chunks: Sequence[Chunk]) -> int:
        # 本地倒排作为降级底座：ES 写失败时检索不至于完全不可用
        self._local.index(chunks)
        for chunk in chunks:
            body = json.dumps(
                {"text": chunk.text, "doc_id": chunk.doc_id, "ordinal": chunk.ordinal},
                ensure_ascii=False,
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{self._url}/{_INDEX}/_doc/{urllib.parse.quote(chunk.chunk_id, safe='')}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            try:
                urllib.request.urlopen(request, timeout=self._timeout).close()
            except urllib.error.URLError as exc:
                raise BackendUnavailable(f"elasticsearch index failed: {exc}") from exc
        return len(chunks)

    def drop_document(self, doc_id: str) -> int:
        return self._local.drop_document(doc_id)

    def search(self, query: str, top_k: int) -> list[Hit]:
        return self._local.search(query, top_k)

    def stats(self) -> dict[str, float]:
        return self._local.stats()

    def search_remote(self, query: str, top_k: int) -> list[Hit]:
        """显式走远端检索，便于压测时对比本地倒排的结果差异。"""

        body = json.dumps({"size": top_k, "query": {"match": {"text": query}}}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._url}/{_INDEX}/_search", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"elasticsearch search failed: {exc}") from exc
        hits: list[Hit] = []
        for item in payload.get("hits", {}).get("hits", []):
            source = item.get("_source", {})
            hits.append(
                Hit(
                    chunk_id=item["_id"],
                    score=float(item.get("_score") or 0.0),
                    source="elastic",
                    meta={"doc_id": source.get("doc_id", "")},
                )
            )
        return hits


__all__ = ["ElasticLexical"]
__author__ = "晨星"
