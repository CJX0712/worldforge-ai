"""标准库 HTTP 网关：ThreadingHTTPServer + 路由分发。

选 stdlib 而非 FastAPI 是为了让裸 CPython 直接跑通服务，第三方框架只作为可选
依赖出现在文档里，而不是启动前置条件。
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from worldforge.config import Config
from worldforge.serving.errors import error_body
from worldforge.serving.routes import MethodNotAllowed, Request, Response, Router, build_router

_MAX_BODY_BYTES = 8 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    """把 BaseHTTPRequestHandler 适配到 Router。"""

    router: Router
    protocol_version = "HTTP/1.1"
    server_version = "WorldForge/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - 覆盖父类以静音 stderr
        return

    def _dispatch(self, method: str) -> None:
        try:
            payload = self._read_json() if method in {"POST", "PUT", "PATCH"} else None
            resolved = self.router.resolve(method, self.path.split("?")[0])
            if resolved is None:
                self._send(Response({"error": {"code": "not_found", "message": self.path}}, 404))
                return
            handler, params = resolved
            request = Request(method, self.path, params, payload)
            self._send(handler(request))
        except MethodNotAllowed as exc:
            self._send(
                Response({"error": {"code": "method_not_allowed", "message": str(exc)}}, 405, {"Allow": ", ".join(exc.allowed)})
            )
        except Exception as exc:  # noqa: BLE001 - 网关边界必须把异常转成响应
            self._send(Response(error_body(exc, ""), _status_of(exc)))
        finally:
            self._pipeline_flush()

    def _pipeline_flush(self) -> None:
        pipeline = getattr(self.server, "pipeline", None)
        if pipeline is not None:
            pipeline.metrics.incr("http_requests")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > _MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("request body must be a JSON object")
        return decoded

    def _send(self, response: Response) -> None:
        is_text = response.headers.get("Content-Type", "").startswith("text/")
        raw = response.body.encode("utf-8") if is_text else json.dumps(response.body, ensure_ascii=False).encode("utf-8")
        self.send_response(response.status)
        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json; charset=utf-8"))
        self.send_header("Content-Length", str(len(raw)))
        for key, value in response.headers.items():
            if key != "Content-Type":
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")


def _status_of(exc: BaseException) -> int:
    from worldforge.serving.errors import status_for

    return status_for(exc)[0]


def create_app(pipeline: Any, evaluate: Callable[[Any], Any] | None = None) -> Router:
    """构造可直接挂载的路由表。"""

    return build_router(pipeline, evaluate)


def build_server(
    config: Config | None = None,
    pipeline: Any = None,
    evaluate: Callable[[Any], Any] | None = None,
) -> ThreadingHTTPServer:
    """构造绑定好 pipeline 的 HTTP 服务。"""

    from worldforge.pipeline import build_pipeline

    cfg = config or Config.from_env()
    active = pipeline or build_pipeline(cfg)
    handler = type("BoundHandler", (_Handler,), {"router": create_app(active, evaluate)})
    server = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    server.pipeline = active  # type: ignore[attr-defined]
    return server


def serve(config: Config | None = None) -> None:  # pragma: no cover - 阻塞入口
    cfg = config or Config.from_env()
    server = build_server(cfg)
    host, port = server.server_address[:2]
    print(f"WorldForge listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(serve() and 0)


__all__ = ["create_app", "build_server", "serve"]
__author__ = "晨星"
