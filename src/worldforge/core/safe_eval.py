"""基于 AST 白名单的安全算术求值器。

作者：晨星
"""

from __future__ import annotations

import ast
import math
import operator
import unicodedata
from collections.abc import Callable
from typing import TypeAlias

from worldforge.errors import SafeEvalError

Number: TypeAlias = int | float
_MAX_EXPRESSION_LENGTH = 512
_MAX_ABSOLUTE_VALUE = 1e308
_MAX_EXPONENT = 1000
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_EDGE_PUNCTUATION = " \t\r\n?!,;:" + "".join(chr(codepoint) for codepoint in (0xFF1F, 0xFF01, 0x3002, 0xFF0C, 0xFF1A, 0xFF1B))


def safe_eval(expression: str) -> Number:
    """只计算有限数值算术，所有失败统一抛出 SafeEvalError。"""

    try:
        if not isinstance(expression, str):
            raise TypeError("表达式必须是字符串")
        normalized = unicodedata.normalize("NFKC", expression).strip(_EDGE_PUNCTUATION)
        if not normalized:
            raise ValueError("表达式不能为空")
        if len(normalized) > _MAX_EXPRESSION_LENGTH:
            raise ValueError("表达式过长")
        tree = ast.parse(normalized, mode="eval")
        return _evaluate(tree.body)
    except SafeEvalError:
        raise
    except Exception as exc:
        raise SafeEvalError(f"算术表达式无效: {exc}") from exc


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise TypeError("只允许数值常量")
        return _bounded(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _bounded(_UNARY_OPERATORS[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise OverflowError("指数超过安全上限")
        return _bounded(_BINARY_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.Name):
        raise NameError(f"未定义名字: {node.id}")
    raise ValueError(f"不允许的语法: {type(node).__name__}")


def _bounded(value: Number) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("结果必须是实数")
    if isinstance(value, float) and not math.isfinite(value):
        raise OverflowError("结果不是有限数")
    if abs(value) > _MAX_ABSOLUTE_VALUE:
        raise OverflowError("结果超过安全上限")
    return value


__all__ = ["safe_eval"]
__author__ = "晨星"
