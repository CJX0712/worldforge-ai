"""自包含 HTTP 端到端冒烟：起服务 -> 打全流程 -> 断言 -> 回收。

不用 curl / docker compose：沙箱里它们不可靠。进程由本脚本自己拉起并在 finally
里回收（Windows 需 taskkill /T 才能真正结束子进程树，否则端口泄漏会让下次失败）。
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from worldforge.config import Config  # noqa: E402
from worldforge.eval.runner import evaluate as run_evaluation  # noqa: E402
from worldforge.pipeline import build_pipeline  # noqa: E402
from worldforge.serving.app import build_server  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def call(base: str, method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            return response.status, (json.loads(body) if body.startswith(("{", "[")) else body)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(label)
    else:
        FAILED.append(f"{label} :: {detail}")


def main() -> int:
    port = 8973
    config = Config(port=port, host="127.0.0.1")
    pipeline = build_pipeline(config)
    # 每个评估 case 必须在全新管道上跑，复用运行期单例管道会污染指标
    server = build_server(
        config,
        pipeline=pipeline,
        evaluate=lambda _active: run_evaluation(lambda: build_pipeline(config), enforce=False),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/health", timeout=2).close()
                break
            except (urllib.error.URLError, TimeoutError):
                threading.Event().wait(0.25)

        status, body = call(base, "GET", "/health")
        check("health 200", status == 200, str(status))
        check("health 报出后端", isinstance(body, dict) and body.get("backends", {}).get("llm") == "mock", str(body))

        status, body = call(base, "POST", "/v1/ingest", {"doc_id": "d1", "text": "BM25 由 Robertson 提出，用逆文档频率衡量区分度。"})
        check("ingest 201", status == 201, str(status))
        status, body = call(base, "POST", "/v1/ingest", {"text": "缺少 doc_id"})
        check("ingest 缺字段 400", status == 400 and body["error"]["code"] == "bad_request", str(body))

        for doc_id, text in (
            ("d2", "Transformer 用多头注意力建模任意位置之间的依赖关系，长程依赖可在单层内建模。"),
            ("d3", "检索增强生成先召回相关文档再让模型生成答案，可缓解幻觉与知识过期问题。"),
        ):
            call(base, "POST", "/v1/ingest", {"doc_id": doc_id, "text": text})

        status, body = call(base, "POST", "/v1/query", {"query": "BM25 是谁提出的", "explain": True})
        check("query 200", status == 200, str(status))
        check("query 命中目标文档", body["hits"][0]["doc_id"] == "d1", str(body["hits"][:1]))
        check("query 返回 explain 双路", set(body.get("explain", {})) == {"sparse", "dense"}, str(body.get("explain")))

        status, body = call(base, "POST", "/v1/ask", {"query": "BM25 是谁提出的"})
        check("ask 200", status == 200, str(status))
        check("ask 答案含 Robertson", "Robertson" in body["answer"], body["answer"][:80])
        check("ask 返回证据", len(body["evidence"]) > 0, str(body.get("evidence")))

        status, body = call(
            base,
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "你好"}]},
        )
        check("chat 200", status == 200, str(status))
        check("chat 兼容字段", body.get("object") == "chat.completion" and body["choices"][0]["message"]["role"] == "assistant", str(body)[:120])

        status, body = call(base, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "算 6*7"}], "tool_call": True})
        check("chat tool_call 200", status == 200 and body["answer"] == "42", str(body)[:120])

        status, body = call(base, "GET", "/v1/metrics")
        check("metrics 200", status == 200 and "counters" in body, str(body)[:80])
        status, body = call(base, "GET", "/v1/traces")
        check("traces 200 且剔除延迟", status == 200 and all("latency_ms" not in s for s in body["spans"]), "latency 未剔除")

        status, body = call(base, "POST", "/v1/evaluate", {})
        check("evaluate 200", status == 200 and "retrieval_metrics" in body, str(body)[:80])

        status, body = call(base, "DELETE", "/v1/documents/d1")
        check("delete 200", status == 200, str(status))
        status, body = call(base, "GET", "/v1/documents")
        check("文档已删除", "d1" not in body["documents"], str(body))

        status, body = call(base, "GET", "/v1/nope")
        check("未知路径 404", status == 404, str(status))
        status, body = call(base, "DELETE", "/health")
        check("方法不允许 405", status == 405, str(status))
    finally:
        server.shutdown()
        server.server_close()

    print(f"通过：{len(PASSED)} / 失败：{len(FAILED)}")
    for item in FAILED:
        print(f"  FAIL {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
__author__ = "晨星"
