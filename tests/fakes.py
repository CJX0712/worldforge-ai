"""供端口契约和后续集成测试复用的确定性替身。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from worldforge.core.hash_embed import hash_embed
from worldforge.types import Chunk, Completion, Document, Hit, Message, Plan, Step, ToolContext, ToolResult, ToolSpec


@dataclass
class FakeLLM:
    """按提示映射返回文本的确定性语言模型。"""

    responses: Mapping[str, str] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return "fake-llm-v1"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        text = self.responses.get(prompt, prompt)[:max_tokens]
        return Completion(text=text, model_id=self.model_id, finish_reason="stop")

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        if not messages:
            raise ValueError("messages 不能为空")
        return self.complete(messages[-1].content, temperature=temperature, max_tokens=max_tokens, stop=stop, seed=seed)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Iterator[str]:
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens, seed=seed).text


@dataclass
class FakeEmbedder:
    """使用小维度散列生成稳定测试向量。"""

    _dim: int = 8

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return f"fake-hash-{self.dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [hash_embed(text, self.dim) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FakeVectorStore:
    """只为结构契约提供完整方法表。"""

    dim = 8
    fingerprint = "fake-hash-8:8"

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metas: Sequence[Mapping[str, Any]]) -> int:
        return len(ids)

    def delete(self, ids: Sequence[str]) -> int:
        return len(ids)

    def drop_document(self, doc_id: str) -> int:
        return 0

    def search(self, vector: Sequence[float], top_k: int, where: Mapping[str, Any] | None = None) -> list[Hit]:
        return []

    def count(self) -> int:
        return 0

    def clear(self) -> None:
        return None

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None


class FakeLexicalIndex:
    """稀疏索引结构替身。"""

    def index(self, chunks: Sequence[Chunk]) -> int:
        return len(chunks)

    def drop_document(self, doc_id: str) -> int:
        return 0

    def search(self, query: str, top_k: int) -> list[Hit]:
        return []

    def stats(self) -> dict[str, float]:
        return {"docs": 0.0, "avgdl": 0.0, "vocab": 0.0}


class FakeReranker:
    """按既有顺序截断的重排结构替身。"""

    def rerank(self, query: str, candidates: Sequence[Hit], top_k: int) -> list[Hit]:
        return list(candidates[:top_k])


class FakeDocStore:
    """文档存储结构替身。"""

    def put_document(self, doc: Document) -> None:
        return None

    def get_document(self, doc_id: str) -> Document | None:
        return None

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[str]:
        return []

    def put_chunks(self, chunks: Sequence[Chunk]) -> int:
        return len(chunks)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return []

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        return []

    def drop_document(self, doc_id: str) -> int:
        return 0

    def count_chunks(self) -> int:
        return 0


class RecordingObserver:
    """在内存中记录事件和指标的观察器。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.metrics: list[tuple[str, float, Mapping[str, str] | None]] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        self.events.append((name, dict(attrs)))
        yield

    def emit(self, event: str, **fields: Any) -> None:
        self.events.append((event, dict(fields)))

    def metric(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        self.metrics.append((name, value, tags))

    def trace_id(self) -> str:
        return "trace-test"

    def flush(self) -> dict[str, Any]:
        result = {"spans": list(self.events), "metrics": list(self.metrics)}
        self.events.clear()
        self.metrics.clear()
        return result


class FakeTool:
    """返回固定成功结果的工具替身。"""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="fake", description="测试工具", args_schema={"type": "object"})

    def invoke(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, output=str(dict(args)))


class FakePlanner:
    """返回单步骤计划的计划器替身。"""

    def plan(self, goal: str, tools: Sequence[ToolSpec], ctx: Mapping[str, Any]) -> Plan:
        return Plan(goal=goal, steps=(Step(step_id="answer", kind="answer", payload={}),))

    def replan(self, plan: Plan, done: Sequence[Step], observation: str, ctx: Mapping[str, Any]) -> Plan:
        return Plan(goal=plan.goal, steps=plan.steps, meta={"observation": observation})


class FakeChunker:
    """将非空文档整体作为一个分块的结构替身。"""

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []
        return [Chunk(chunk_id=f"{doc.doc_id}#0000", doc_id=doc.doc_id, ordinal=0, text=doc.text)]


__all__ = [
    "FakeLLM",
    "FakeEmbedder",
    "FakeVectorStore",
    "FakeLexicalIndex",
    "FakeReranker",
    "FakeDocStore",
    "RecordingObserver",
    "FakeTool",
    "FakePlanner",
    "FakeChunker",
]
__author__ = "晨星"
