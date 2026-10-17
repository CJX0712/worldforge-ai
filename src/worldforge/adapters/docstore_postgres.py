"""Postgres 文档存储（可选 extras: docstore-postgres）。

作者：晨星
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from worldforge.errors import BackendUnavailable
from worldforge.types import Chunk, Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wf_documents (
    doc_id TEXT PRIMARY KEY, text TEXT NOT NULL, meta JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS wf_chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, ordinal INT NOT NULL,
    text TEXT NOT NULL, meta JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS wf_chunks_doc ON wf_chunks(doc_id, ordinal);
"""


class PostgresDocStore:
    """psycopg 驱动的 DocStore，接口与默认内存实现完全一致。"""

    def __init__(self, dsn: str = "", min_size: int = 1, max_size: int = 4) -> None:
        if not dsn:
            raise BackendUnavailable("postgres dsn is empty")
        try:
            from psycopg_pool import ConnectionPool  # noqa: PLC0415 - 惰性导入
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise BackendUnavailable(f"psycopg is not installed: {exc}") from exc
        self._pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True)
        with self._pool.connection() as conn:
            conn.execute(_SCHEMA)

    def put_document(self, doc: Document) -> None:
        self._run(
            "INSERT OR REPLACE INTO wf_documents(doc_id, text, meta) VALUES (%s, %s, %s)",
            (doc.doc_id, doc.text, json.dumps(dict(doc.meta), ensure_ascii=False)),
        )

    def get_document(self, doc_id: str) -> Document | None:
        rows = self._run("SELECT * FROM wf_documents WHERE doc_id = %s", (doc_id,))
        if not rows:
            return None
        row = rows[0]
        return Document(doc_id=row[0], text=row[1], meta=json.loads(row[2]))

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[str]:
        rows = self._run("SELECT doc_id FROM wf_documents ORDER BY doc_id LIMIT %s OFFSET %s", (limit, offset))
        return [r[0] for r in rows]

    def put_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        args: list[Any] = []
        for c in chunks:
            args.extend([c.chunk_id, c.doc_id, c.ordinal, c.text, json.dumps(dict(c.meta), ensure_ascii=False)])
        placeholders = ",".join(["(%s, %s, %s, %s, %s)"] * len(chunks))
        self._run(
            f"INSERT OR REPLACE INTO wf_chunks(chunk_id, doc_id, ordinal, text, meta) VALUES {placeholders}",
            tuple(args),
        )
        return len(chunks)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        rows = self._run(
            "SELECT * FROM wf_chunks WHERE chunk_id = ANY(%s) ORDER BY chunk_id", (list(chunk_ids),)
        )
        return [Chunk(chunk_id=r[0], doc_id=r[1], ordinal=int(r[2]), text=r[3], meta=json.loads(r[4])) for r in rows]

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        rows = self._run("SELECT * FROM wf_chunks WHERE doc_id = %s ORDER BY ordinal", (doc_id,))
        return [Chunk(chunk_id=r[0], doc_id=r[1], ordinal=int(r[2]), text=r[3], meta=json.loads(r[4])) for r in rows]

    def drop_document(self, doc_id: str) -> int:
        rows = self._run("DELETE FROM wf_chunks WHERE doc_id = %s RETURNING 1", (doc_id,))
        self._run("DELETE FROM wf_documents WHERE doc_id = %s", (doc_id,))
        return len(rows)

    def count_chunks(self) -> int:
        rows = self._run("SELECT COUNT(*) FROM wf_chunks", ())
        return int(rows[0][0])

    def _run(self, sql: str, args: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        with self._pool.connection() as conn:
            cursor = conn.execute(sql, args)
            return list(cursor.fetchall()) if cursor.description else []


__all__ = ["PostgresDocStore"]
__author__ = "晨星"
