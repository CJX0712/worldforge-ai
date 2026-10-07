"""HTTP 请求与响应体的解析校验。

作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ValidationError(ValueError):
    """请求体字段缺失或类型不符。"""


def require(payload: Mapping[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in payload:
        raise ValidationError(f"missing required field '{key}'")
    value = payload[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        expected = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise ValidationError(f"field '{key}' must be {expected}")
    return value


def optional_int(payload: Mapping[str, Any], key: str, default: int, *, minimum: int = 1) -> int:
    raw = payload.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValidationError(f"field '{key}' must be int")
    if raw < minimum:
        raise ValidationError(f"field '{key}' must be >= {minimum}")
    return raw


@dataclass(frozen=True)
class IngestRequest:
    doc_id: str
    text: str
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class QueryRequest:
    query: str
    top_k: int
    explain: bool


def parse_ingest(payload: Mapping[str, Any]) -> IngestRequest:
    meta = payload.get("meta", {})
    if not isinstance(meta, Mapping):
        raise ValidationError("field 'meta' must be an object")
    return IngestRequest(
        doc_id=str(require(payload, "doc_id", str)),
        text=str(require(payload, "text", str)),
        meta=meta,
    )


def parse_query(payload: Mapping[str, Any]) -> QueryRequest:
    return QueryRequest(
        query=str(require(payload, "query", str)),
        top_k=optional_int(payload, "top_k", 5),
        explain=bool(payload.get("explain", False)),
    )


def parse_chat(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise ValidationError("field 'messages' must be a non-empty array")
    out: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise ValidationError("each message must be an object")
        role = item.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValidationError(f"unsupported message role: {role}")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValidationError("message content must be a non-empty string")
        out.append({"role": str(role), "content": content})
    return out


__all__ = [
    "ValidationError",
    "IngestRequest",
    "QueryRequest",
    "require",
    "optional_int",
    "parse_ingest",
    "parse_query",
    "parse_chat",
]
__author__ = "晨星"
