"""sqlite3 文档存储（stdlib，非默认后端）。

作者：晨星
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Sequence

from worldforge.types import Chunk, Document


class SqliteDocStore:
    """单文件持久化 DocStore。

    非默认：sqlite 的行返回顺序依赖插入历史，`chunks_of` 必须显式 ORDER BY
    ordinal 才能跨运行复现，否则评估指标会抖。
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY, text TEXT NOT NULL, meta TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                text TEXT NOT NULL, meta TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);
            """
        )
        self._conn.commit()

    def put_document(self, doc: Document) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?)",
            (doc.doc_id, doc.text, json.dumps(dict(doc.meta), ensure_ascii=False)),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        return Document(doc_id=row["doc_id"], text=row["text"], meta=json.loads(row["meta"]))

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[str]:
        rows = self._conn.execute(
            "SELECT doc_id FROM documents ORDER BY doc_id LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [r["doc_id"] for r in rows]

    def put_chunks(self, chunks: Sequence[Chunk]) -> int:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
            [(c.chunk_id, c.doc_id, c.ordinal, c.text, json.dumps(dict(c.meta), ensure_ascii=False)) for c in chunks],
        )
        self._conn.commit()
        return len(chunks)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders}) ORDER BY chunk_id", tuple(chunk_ids)
        ).fetchall()
        return [self._to_chunk(r) for r in rows]

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
        ).fetchall()
        return [self._to_chunk(r) for r in rows]

    def drop_document(self, doc_id: str) -> int:
        cursor = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return cursor.rowcount

    def count_chunks(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    @staticmethod
    def _to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            ordinal=int(row["ordinal"]),
            text=row["text"],
            meta=json.loads(row["meta"]),
        )


__all__ = ["SqliteDocStore"]
__author__ = "晨星"
