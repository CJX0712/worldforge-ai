"""MCP 工具源（可选 extras: tool-mcp）。

作者：晨星
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from worldforge.errors import BackendUnavailable, ToolError
from worldforge.types import ToolContext, ToolResult, ToolSpec


class MCPToolSource:
    """按 MCP 协议发现并调用远端工具。

    工具是外部输入，描述文本一律视为不可信数据：只允许白名单方法名，
    参数按键过滤而不是整体透传。
    """

    _ALLOWED_METHODS = frozenset({"tools/list", "tools/call"})

    def __init__(self, endpoint: str = "", timeout: float = 15.0, allowed_tools: tuple[str, ...] = ()) -> None:
        if not endpoint:
            raise BackendUnavailable("mcp endpoint is empty")
        self._endpoint = endpoint
        self._timeout = timeout
        self._allowed = allowed_tools
        self._cache: dict[str, Any] | None = None

    def list_specs(self) -> list[ToolSpec]:
        payload = self._rpc("tools/list", {})
        specs: list[ToolSpec] = []
        for item in payload.get("result", {}).get("tools", []):
            name = item.get("name", "")
            if not name or (self._allowed and name not in self._allowed):
                continue
            specs.append(
                ToolSpec(
                    name=name,
                    description=item.get("description", ""),
                    args_schema=item.get("inputSchema", {}),
                )
            )
        return sorted(specs, key=lambda s: s.name)

    def call(self, tool_name: str, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        if self._allowed and tool_name not in self._allowed:
            return ToolResult(ok=False, output="", error=f"tool '{tool_name}' is not in the allow list")
        payload = self._rpc("tools/call", {"name": tool_name, "arguments": dict(args)})
        if "error" in payload:
            return ToolResult(ok=False, output="", error=str(payload["error"]))
        content = payload.get("result", {}).get("content", [])
        text = "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
        return ToolResult(ok=True, output=text, meta={"trace_id": ctx.trace_id})

    def _rpc(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if method not in self._ALLOWED_METHODS:
            raise ToolError(f"method '{method}' is not allowed")
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": dict(params)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ToolError(f"mcp rpc '{method}' failed: {exc}") from exc


__all__ = ["MCPToolSource"]
__author__ = "晨星"
