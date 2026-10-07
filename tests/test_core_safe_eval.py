"""安全算术求值测试。

作者：晨星
"""

from __future__ import annotations

import pytest

from worldforge.core.safe_eval import safe_eval
from worldforge.errors import SafeEvalError, ToolError


def test_safe_eval_accepts_only_arithmetic() -> None:
    assert safe_eval("1 + 2 * 3") == 7
    assert safe_eval("-(8 // 3) + 5") == 3
    expression = chr(0xFF11) + chr(0xFF0B) + chr(0xFF11) + chr(0xFF1F)
    assert safe_eval(expression) == 2


@pytest.mark.parametrize("expression", ["1/0", "10**100000", "1 +", "undefined_name", 'int("x")'])
def test_all_failures_are_normalized(expression: str) -> None:
    with pytest.raises(SafeEvalError) as captured:
        safe_eval(expression)
    assert isinstance(captured.value, ToolError)


def test_edge_trimming_preserves_arithmetic_operators() -> None:
    assert safe_eval("+2") == 2
    assert safe_eval("-2") == -2
    assert safe_eval("8/4") == 2
    assert safe_eval("3*4") == 12


__author__ = "晨星"
