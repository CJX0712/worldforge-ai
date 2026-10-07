"""工具路由、ReAct 收敛与接地核验。"""

from __future__ import annotations

import pytest

from worldforge.agents.builtin_tools import CalculatorTool, DictionaryTool, EchoTool, detect_arithmetic, normalize_expression
from worldforge.agents.planner_rule import RuleBasedPlanner, topological_sort
from worldforge.agents.react import parse_final
from worldforge.agents.tools import ToolRegistry
from worldforge.agents.verify import faithfulness, grounding_score
from worldforge.config import Config
from worldforge.errors import PlanError
from worldforge.pipeline import build_pipeline
from worldforge.types import Step, ToolContext

CTX = ToolContext(trace_id="t", seed=1337, observer=None)


@pytest.fixture()
def registry():
    return ToolRegistry([CalculatorTool(), EchoTool(), DictionaryTool({"bm25": "相关性排序函数"})])


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("帮我算 12*(3+4) 等于多少", "12*(3+4)"),
        ("12 * (3 + 4) = ?", "12 * (3 + 4) "),
        ("计算 7/2 的值", "7/2 "),
    ],
)
def test_detect_arithmetic_on_chinese_questions(utterance, expected):
    assert detect_arithmetic(utterance) is not None
    assert expected.strip() in (detect_arithmetic(utterance) or "")


def test_detect_arithmetic_handles_fullwidth():
    assert normalize_expression("１２×（３＋４）") == "12*(3+4)"
    assert detect_arithmetic("１２×（３＋４）是多少") == "12*(3+4)"


def test_calculator_rejects_zero_division():
    result = CalculatorTool().invoke({"expression": "1/0"}, CTX)
    assert not result.ok and result.error


def test_registry_routes_arithmetic_deterministically(registry):
    name, args = registry.route("帮我算 12*(3+4)")
    assert name == "calculator"
    assert registry.invoke(name, args, CTX).output == "84"


def test_registry_invokes_by_explicit_name(registry):
    assert registry.invoke("echo", {"text": "hi"}, CTX).output == "hi"
    assert not registry.invoke("nope", {}, CTX).ok


def test_dictionary_reports_miss_instead_of_inventing(registry):
    result = registry.invoke("dictionary", {"term": "不存在的词"}, CTX)
    assert not result.ok and "not found" in (result.error or "")


def test_parse_final_skips_protocol_lines():
    text = "Thought: 我该算数\nAction: calculator\nObservation: 84\nFinal Answer: 84"
    assert parse_final(text) == "84"
    # 协议行必须被排除：否则最后一条 Action 会被当成答案返回
    assert parse_final("Thought: t\nAction: search") == ""
    assert parse_final("Observation: 只是观测") == ""
    assert parse_final("Thought: t\n真正的结论在这里") == "真正的结论在这里"


def test_react_converges_on_arithmetic():
    pipeline = build_pipeline(Config())
    run = pipeline.act("帮我算 12*(3+4) 等于多少")
    assert run["answer"] == "84"
    assert run["converged"] is True
    assert run["tool_calls"] == 1


def test_react_never_exceeds_step_budget():
    pipeline = build_pipeline(Config())
    run = pipeline.act("今天天气怎么样")
    assert run["tool_calls"] <= 4


def test_rule_planner_routes_arithmetic_to_single_step():
    specs = [CalculatorTool().spec, EchoTool().spec]
    plan = RuleBasedPlanner().plan("12*3+4 等于多少", specs, {})
    assert len(plan.steps) == 1 and plan.steps[0].payload["tool"] == "calculator"


def test_rule_planner_adds_generate_step_for_questions():
    specs = [CalculatorTool().spec]
    plan = RuleBasedPlanner().plan("BM25 是什么", specs, {})
    assert [s.kind for s in plan.steps] == ["generate"]


def test_rule_planner_without_calculator_falls_back_to_generate():
    plan = RuleBasedPlanner().plan("12*3+4 等于多少", [], {})
    assert [s.kind for s in plan.steps] == ["generate"], "工具不可用时不得硬造 calculator 步骤"


def test_topological_sort_detects_cycle():
    steps = (Step("a", "tool", {}, ("b",)), Step("b", "tool", {}, ("a",)))
    with pytest.raises(PlanError):
        topological_sort(steps)


def test_topological_sort_detects_unknown_dependency():
    with pytest.raises(PlanError):
        topological_sort((Step("a", "tool", {}, ("ghost",)),))


def test_grounding_ignores_citation_markers():
    answer = "BM25 由 Robertson 提出。 [kb-bm25#0]"
    evidence = ["BM25 是一种相关性排序函数，由 Robertson 提出。"]
    assert grounding_score(answer, evidence) == grounding_score("BM25 由 Robertson 提出。", evidence)
    assert grounding_score(answer, evidence) > 0.5


def test_faithfulness_drops_when_answer_is_invented():
    evidence = ["BM25 由 Robertson 提出。"]
    assert faithfulness("BM25 由 Robertson 提出。", evidence) > 0.5
    assert faithfulness("量子纠缠在区块链上的应用", evidence) < 0.3
