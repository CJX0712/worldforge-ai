"""内置工具：算术、检索、回声、词典。

作者：晨星
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

from worldforge.core.safe_eval import safe_eval
from worldforge.core.tokenizer import normalize_text
from worldforge.errors import ToolError
from worldforge.types import ToolContext, ToolResult, ToolSpec

# 中文问句里的全角运算符与数字必须先归一，否则 AST 直接语法错误
_FULLWIDTH = {0xFF10 + offset: 0x30 + offset for offset in range(10)}
_FULLWIDTH.update({0xFF0B: 0x2B, 0xFF0D: 0x2D, 0xFF0A: 0x2A, 0xFF0F: 0x2F, 0xFF08: 0x28, 0xFF09: 0x29, 0xFF1A: 0x3A})
# 排版符号：乘号 U+00D7、减号 U+2212、除号 U+00F7 在中文排版里极常见
_FULLWIDTH.update({0x00D7: 0x2A, 0x2212: 0x2D, 0x00F7: 0x2F})
_EXPRESSION_CHARS = set("0123456789+-*/(). ")

_ARITHMETIC_RE = re.compile(r"[0-9+\-*/().\s]{3,}")


def map_fullwidth(text: str) -> str:
    """把全角数字与运算符映射为半角，供正则与 AST 共用。"""

    return "".join(chr(_FULLWIDTH.get(ord(ch), ord(ch))) for ch in text)


def normalize_expression(text: str) -> str:
    """归一化全角数字与运算符；裁剪集绝不能含 `+ - * /`，否则会切坏合法算式。"""

    trimmed = re.sub(r"^[\s?!,;:?=]+|[\s?!,;:?=]+$", "", map_fullwidth(text))
    return trimmed if all(ch in _EXPRESSION_CHARS for ch in trimmed) else ""


def detect_arithmetic(text: str) -> str | None:
    """从自然语言里抽出算式片段。

    不能对整句做 ast.parse：中文问句必然语法错误，必须先用正则截出连续片段
    再逐个校验。全角映射必须在这一步之前完成，否则全角算式抽不出来。
    """

    best = ""
    for candidate in _ARITHMETIC_RE.findall(map_fullwidth(text)):
        candidate = candidate.strip()
        if not any(op in candidate for op in "+-*/"):
            continue
        try:
            ast.parse(candidate, mode="eval")
        except SyntaxError:
            continue
        if len(candidate) > len(best):
            best = candidate
    return best or None


class CalculatorTool:
    """安全算术求值。"""

    def __init__(self) -> None:
        self._spec = ToolSpec(
            name="calculator",
            description="计算仅含加减乘除与括号的算式",
            args_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def invoke(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        raw = str(args.get("expression", ""))
        expression = normalize_expression(raw) or raw.strip()
        if not expression:
            return ToolResult(ok=False, output="", error="expression is empty")
        try:
            value = safe_eval(expression)
        except ToolError as exc:
            return ToolResult(ok=False, output="", error=str(exc))
        return ToolResult(ok=True, output=repr(value), meta={"expression": expression})


class EchoTool:
    """原样回声，用于链路连通性自检。"""

    def __init__(self) -> None:
        self._spec = ToolSpec(
            name="echo", description="回显输入文本", args_schema={"type": "object", "properties": {"text": {"type": "string"}}}
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def invoke(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, output=str(args.get("text", "")), meta={"trace_id": ctx.trace_id})


class RetrieverTool:
    """把混合检索包装成工具。"""

    def __init__(self, retriever: Any, top_k: int = 5) -> None:
        self._retriever = retriever
        self._top_k = top_k
        self._spec = ToolSpec(
            name="search",
            description="在知识库中检索与问题相关的段落",
            args_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def invoke(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, output="", error="query is empty")
        hits = self._retriever.search(query)[: self._top_k]
        lines = [f"[{h.chunk_id}] {h.meta.get('text', '')}" for h in hits]
        return ToolResult(ok=True, output="\n".join(lines), meta={"hit_count": len(hits)})


class DictionaryTool:
    """术语释义，缺词时明确返回未命中而不是编造。"""

    def __init__(self, glossary: Mapping[str, str] | None = None) -> None:
        self._glossary = dict(glossary or {})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="dictionary",
            description="查询术语释义",
            args_schema={"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]},
        )

    def invoke(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        term = normalize_text(str(args.get("term", "")))
        if term in self._glossary:
            return ToolResult(ok=True, output=self._glossary[term])
        return ToolResult(ok=False, output="", error=f"term '{term}' not found")


def builtin_tools(retriever: Any = None, top_k: int = 5) -> list[Any]:
    """构造默认工具集；retriever 为空时不提供检索工具。"""

    tools: list[Any] = [CalculatorTool(), EchoTool(), DictionaryTool()]
    if retriever is not None:
        tools.append(RetrieverTool(retriever, top_k=top_k))
    return tools


__all__ = [
    "CalculatorTool",
    "EchoTool",
    "RetrieverTool",
    "DictionaryTool",
    "builtin_tools",
    "map_fullwidth",
    "normalize_expression",
    "detect_arithmetic",
]
__author__ = "晨星"
