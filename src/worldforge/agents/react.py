"""ReAct 循环与最终答案提取。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from worldforge.agents.tools import ToolRegistry
from worldforge.types import ToolContext

_PROTOCOL_PREFIXES = ("action:", "thought:", "observation:")


@dataclass
class ReactRun:
    """一次 ReAct 执行的可观测结果。"""

    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    converged: bool = False
    tool_calls: int = 0


def parse_final(text: str) -> str:
    """从模型输出里取最终答案。

    单行兜底必须排除 `action:` / `thought:` / `observation:` 开头的行，否则最后
    一条 Action 协议行会被当成答案返回。
    """

    marker = "Final Answer:"
    index = text.rfind(marker)
    if index != -1:
        return text[index + len(marker) :].strip()
    for line in reversed(text.strip().splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.lower().startswith(_PROTOCOL_PREFIXES):
            continue
        return candidate
    return ""


class ReActAgent:
    """Thought / Action / Observation 循环。

    不变量
        I1  有观测即必须收敛：步数上限之外还有"答案非空"硬上界，否则模型会
            一路跑到步数上限
        I2  打分用词集合余弦而不是"查询覆盖率 |Q∩S|/|Q|"：后者对长问句极敏感，
            所有短答案句得分会一起崩塌，模型永不收敛
    """

    def __init__(self, registry: ToolRegistry, llm: Any, max_steps: int = 4) -> None:
        self._registry = registry
        self._llm = llm
        self._max_steps = max_steps

    def run(self, question: str, ctx: ToolContext, prompt_builder: Any = None) -> ReactRun:
        run = ReactRun(answer="")
        transcript: list[str] = []

        # 确定性前置路由：小模型的路由与算术都不可靠，检出任一算式就先算掉，
        # 结果作为 Observation 注入，禁止模型重算。
        tool_name, args = self._registry.route(question)
        if tool_name is not None:
            result = self._registry.invoke(tool_name, args, ctx)
            run.tool_calls += 1
            run.steps.append({"kind": "action", "tool": tool_name, "args": args, "ok": result.ok, "stage": "pre-route"})
            observation = result.output if result.ok else f"error: {result.error}"
            run.observations.append(observation)
            transcript.append(f"Observation: {observation}")
            if result.ok and tool_name == "calculator":
                # 算式已有确定答案，不必再问模型
                run.answer = observation
                run.converged = True
                return run

        for _ in range(self._max_steps):
            if prompt_builder is not None:
                prompt = prompt_builder(question, transcript)
            else:
                prompt = question + ("\n" + "\n".join(transcript) if transcript else "")
            completion = self._llm.complete(prompt, seed=ctx.seed)
            text = completion.text
            final = parse_final(text)
            if final:
                run.answer = final
                run.converged = True
                run.steps.append({"kind": "final", "text": text})
                return run

            routed, routed_args = self._registry.route(text)
            if routed is None:
                run.steps.append({"kind": "noop", "text": text})
                continue
            result = self._registry.invoke(routed, routed_args, ctx)
            run.tool_calls += 1
            run.steps.append({"kind": "action", "tool": routed, "args": routed_args, "ok": result.ok})
            observation = result.output if result.ok else f"error: {result.error}"
            run.observations.append(observation)
            transcript.append(f"Action: {routed}\nObservation: {observation}")
        if not run.answer and run.observations:
            run.answer = run.observations[-1]
        return run

    def specs(self) -> Sequence[Any]:
        return self._registry.specs()


__all__ = ["ReActAgent", "ReactRun", "parse_final"]
__author__ = "晨星"
