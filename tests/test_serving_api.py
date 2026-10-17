"""HTTP 网关契约：路由、错误码与端点行为。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

import pytest

from worldforge.config import Config
from worldforge.serving.app import build_server
from worldforge.serving.errors import status_for
from worldforge.serving.routes import MethodNotAllowed, Router


@pytest.fixture(scope="module")
def server():
    from worldforge.pipeline import build_pipeline

    config = Config(port=8974, host="127.0.0.1")
    httpd = build_server(config, pipeline=build_pipeline(config))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:8974"
    for _ in range(80):
        try:
            urllib.request.urlopen(base + "/health", timeout=2).close()
            break
        except (urllib.error.URLError, TimeoutError):
            threading.Event().wait(0.25)
    yield base
    httpd.shutdown()
    httpd.server_close()


def call(base: str, method: str, path: str, payload: Any = None) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, json.loads(raw) if raw else None


def test_health(server):
    status, body = call(server, "GET", "/health")
    assert status == 200 and body["status"] == "ok"


def test_full_rag_flow(server):
    status, _ = call(server, "POST", "/v1/ingest", {"doc_id": "k1", "text": "BM25 由 Robertson 提出，用逆文档频率衡量区分度。"})
    assert status == 201
    call(server, "POST", "/v1/ingest", {"doc_id": "k2", "text": "Transformer 用多头注意力建模任意位置的依赖关系。"})
    call(server, "POST", "/v1/ingest", {"doc_id": "k3", "text": "检索增强生成先召回文档再生成答案，可缓解幻觉。"})
    status, body = call(server, "POST", "/v1/ask", {"query": "BM25 是谁提出的"})
    assert status == 200 and "Robertson" in body["answer"]
    assert body["evidence"] and body["grounding"] > 0


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"text": "x"}, "bad_request"),
        ({"doc_id": "x"}, "bad_request"),
        ({"doc_id": "x", "text": 5}, "bad_request"),
    ],
)
def test_ingest_validation(server, payload, code):
    status, body = call(server, "POST", "/v1/ingest", payload)
    assert status == 400 and body["error"]["code"] == code


def test_query_rejects_non_positive_top_k(server):
    status, body = call(server, "POST", "/v1/query", {"query": "x", "top_k": 0})
    assert status == 400


def test_chat_is_openai_compatible(server):
    status, body = call(server, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "你好"}]})
    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_chat_rejects_bad_role(server):
    status, body = call(server, "POST", "/v1/chat/completions", {"messages": [{"role": "root", "content": "x"}]})
    assert status == 400


def test_unknown_route_is_404(server):
    assert call(server, "GET", "/v1/nope")[0] == 404


def test_method_not_allowed(server):
    status, body = call(server, "DELETE", "/health")
    assert status == 405 and body["error"]["code"] == "method_not_allowed"


def test_document_lifecycle(server):
    call(server, "POST", "/v1/ingest", {"doc_id": "temp-doc", "text": "临时文档内容。"})
    status, body = call(server, "GET", "/v1/documents")
    assert status == 200 and "temp-doc" in body["documents"]
    assert call(server, "DELETE", "/v1/documents/temp-doc")[0] == 200
    status, body = call(server, "GET", "/v1/documents")
    assert "temp-doc" not in body["documents"]


def test_traces_exclude_latency(server):
    call(server, "POST", "/v1/query", {"query": "BM25"})
    status, body = call(server, "GET", "/v1/traces")
    assert status == 200
    assert all("latency_ms" not in span for span in body["spans"])


def test_router_reports_allowed_methods():
    router = Router()
    router.add("GET", "/a", lambda r: None)
    with pytest.raises(MethodNotAllowed):
        router.resolve("POST", "/a")
    assert router.resolve("GET", "/a") is not None


def test_error_status_mapping():
    from worldforge.errors import DocNotFound, LLMError, SafeEvalError

    assert status_for(DocNotFound("x"))[0] == 404
    assert status_for(SafeEvalError("x"))[0] == 400
    assert status_for(LLMError("x"))[0] == 502
    assert status_for(ValueError("x"))[0] == 400
