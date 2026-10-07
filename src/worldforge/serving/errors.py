"""HTTP 错误信封与状态码映射。

作者：晨星
"""

from __future__ import annotations

from typing import Any

from worldforge.errors import (
    BackendUnavailable,
    ConfigError,
    DimensionMismatch,
    DocNotFound,
    EvalGateError,
    IngestError,
    LLMError,
    PlanError,
    SafeEvalError,
    ToolError,
    WorldforgeError,
)

_STATUS: tuple[tuple[type[WorldforgeError], int, str], ...] = (
    (DocNotFound, 404, "not_found"),
    (EvalGateError, 422, "eval_gate_failed"),
    (IngestError, 400, "ingest_failed"),
    (ConfigError, 400, "bad_config"),
    (DimensionMismatch, 409, "dimension_mismatch"),
    (BackendUnavailable, 503, "backend_unavailable"),
    (LLMError, 502, "llm_failed"),
    (SafeEvalError, 400, "safe_eval_failed"),
    (ToolError, 400, "tool_failed"),
    (PlanError, 422, "plan_failed"),
)


def status_for(exc: BaseException) -> tuple[int, str]:
    """把领域异常映射为 HTTP 状态码与机器可读错误码。"""

    for error_type, status, code in _STATUS:
        if isinstance(exc, error_type):
            return status, code
    if isinstance(exc, WorldforgeError):
        return 400, "worldforge_error"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return 400, "bad_request"
    return 500, "internal_error"


def error_body(exc: BaseException, trace_id: str) -> dict[str, Any]:
    """构造统一错误信封。"""

    status, code = status_for(exc)
    return {
        "error": {
            "code": code,
            "message": str(exc),
            "type": type(exc).__name__,
            "trace_id": trace_id,
        }
    }


__all__ = ["status_for", "error_body"]
__author__ = "晨星"
