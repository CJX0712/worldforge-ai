"""全部 HTTP 路由：健康、摄入、检索、对话、评估、指标、运维。

作者：晨星
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Mapping

from worldforge.serving.errors import error_body
from worldforge.serving.schemas import (
    ValidationError,
    parse_chat,
    parse_ingest,
    parse_query,
)
from worldforge.types import Message

Handler = Callable[["Request"], "Response"]


class Request:
    """已解析的请求对象。"""

    def __init__(self, method: str, path: str, params: Mapping[str, str], payload: Mapping[str, Any] | None) -> None:
        self.method = method
        self.path = path
        self.params = dict(params)
        self.payload = payload or {}
        self.trace_id = uuid.uuid4().hex[:16]


class Response:
    """带显式状态码的响应，body 交给上层统一 JSON 编码。"""

    def __init__(self, body: Any, status: int = 200, headers: Mapping[str, str] | None = None) -> None:
        self.body = body
        self.status = status
        self.headers = dict(headers or {})


class Router:
    """极简路径路由：字面量段 + 单层 `{name}` 占位符。"""

    def __init__(self) -> None:
        self._routes: list[tuple[str, str, Handler]] = []

    def add(self, method: str, pattern: str, handler: Handler) -> None:
        self._routes.append((method, pattern, handler))

    def resolve(self, method: str, path: str) -> tuple[Handler, dict[str, str]] | None:
        parts = [p for p in path.split("/") if p]
        allowed: set[str] = set()
        for route_method, pattern, handler in self._routes:
            pattern_parts = [p for p in pattern.split("/") if p]
            if len(pattern_parts) != len(parts):
                continue
            params: dict[str, str] = {}
            for expected, actual in zip(pattern_parts, parts):
                if expected.startswith("{") and expected.endswith("}"):
                    params[expected[1:-1]] = actual
                elif expected != actual:
                    break
            else:
                if route_method != method:
                    allowed.add(route_method)
                    continue
                return handler, params
        if allowed:
            raise MethodNotAllowed(sorted(allowed))
        return None


class MethodNotAllowed(Exception):
    def __init__(self, allowed: list[str]) -> None:
        super().__init__(f"allowed methods: {', '.join(allowed)}")
        self.allowed = allowed


def build_router(pipeline: Any, evaluate_fn: Callable[[Any], Mapping[str, Any]] | None = None) -> Router:
    """把 Pipeline 暴露成 10 个端点。"""

    router = Router()

    def health(_: Request) -> Response:
        return Response(pipeline.health())

    def ingest(req: Request) -> Response:
        parsed = parse_ingest(req.payload)
        count = pipeline.ingest(parsed.doc_id, parsed.text, dict(parsed.meta))
        return Response({"doc_id": parsed.doc_id, "chunks": count}, 201)

    def drop_document(req: Request) -> Response:
        removed = pipeline.drop(req.params["doc_id"])
        return Response({"doc_id": req.params["doc_id"], "removed_chunks": removed})

    def list_documents(req: Request) -> Response:
        return Response({"documents": pipeline.documents()})

    def query(req: Request) -> Response:
        parsed = parse_query(req.payload)
        started = time.perf_counter()
        hits = pipeline.retrieve(parsed.query, parsed.top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        body: dict[str, Any] = {
            "query": parsed.query,
            "hits": [
                {"chunk_id": h.chunk_id, "score": round(h.score, 6), "source": h.source, "doc_id": h.meta.get("doc_id", "")}
                for h in hits
            ],
            "latency_ms": round(latency_ms, 3),
        }
        if parsed.explain:
            retriever = pipeline.parts["retriever"]
            body["explain"] = {
                "sparse": [(h.chunk_id, round(h.score, 6)) for h in retriever.last_sparse],
                "dense": [(h.chunk_id, round(h.score, 6)) for h in retriever.last_dense],
            }
        return Response(body)

    def ask(req: Request) -> Response:
        parsed = parse_query(req.payload)
        return Response(pipeline.answer(parsed.query))

    def chat(req: Request) -> Response:
        messages = parse_chat(req.payload)
        if bool(req.payload.get("tool_call")):
            return Response(pipeline.act(messages[-1]["content"]))
        result = pipeline.chat([Message(role=m["role"], content=m["content"]) for m in messages])
        return Response(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": result["model_id"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": result["finish_reason"],
                    }
                ],
            }
        )

    def stream_chat(req: Request) -> Response:
        messages = [Message(role=m["role"], content=m["content"]) for m in parse_chat(req.payload)]
        completion = pipeline.parts["llm"].chat(messages, seed=pipeline.config.seed)
        lines = [
            f"data: {chunk}\n\n" for chunk in _chunks(completion.text, 24)
        ]
        lines.append("data: [DONE]\n\n")
        return Response("".join(lines), headers={"Content-Type": "text/event-stream"})

    def run_evaluation(req: Request) -> Response:
        # 参数名必须与本函数不同名：局部 def 会覆盖同名闭包变量，导致无限递归
        if evaluate_fn is None:
            return Response({"error": {"code": "eval_disabled", "message": "评估套件未装配"}}, 501)
        return Response(dict(evaluate_fn(pipeline)))

    def metrics(_: Request) -> Response:
        return Response(pipeline.metrics_snapshot())

    def traces(_: Request) -> Response:
        return Response(pipeline.observability_snapshot())

    def reset(_: Request) -> Response:
        pipeline.reset()
        return Response({"status": "reset"})

    router.add("GET", "/health", health)
    router.add("POST", "/v1/ingest", ingest)
    router.add("GET", "/v1/documents", list_documents)
    router.add("DELETE", "/v1/documents/{doc_id}", drop_document)
    router.add("POST", "/v1/query", query)
    router.add("POST", "/v1/ask", ask)
    router.add("POST", "/v1/chat/completions", chat)
    router.add("POST", "/v1/chat/completions/stream", stream_chat)
    router.add("POST", "/v1/evaluate", run_evaluation)
    router.add("GET", "/v1/metrics", metrics)
    router.add("GET", "/v1/traces", traces)
    router.add("POST", "/v1/admin/reset", reset)
    return router


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


__all__ = ["Router", "Request", "Response", "MethodNotAllowed", "build_router", "error_body", "ValidationError"]
__author__ = "晨星"
