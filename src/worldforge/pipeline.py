"""装配根：按配置把 10 个 Port 注入成可运行的 Pipeline。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Sequence

from worldforge.adapters.registry import (
    CHUNKERS,
    DEFAULTS,
    DOCSTORES,
    EMBEDDERS,
    LEXICALS,
    LLMS,
    OBSERVERS,
    RERANKERS,
    VECTORSTORES,
    degraded,
    pick,
)
from worldforge.agents.builtin_tools import builtin_tools
from worldforge.agents.planner_rule import RuleBasedPlanner
from worldforge.agents.react import ReactRun
from worldforge.agents.tools import ToolRegistry
from worldforge.agents.verify import faithfulness, grounding_score
from worldforge.config import Config
from worldforge.core.seeding import seed_everything
from worldforge.errors import DocNotFound
from worldforge.observability.trace import Metrics, Tracer
from worldforge.retrieval.chunking import Ingestor
from worldforge.retrieval.dense import DenseRetriever
from worldforge.retrieval.evidence import dedupe_evidence, render_for_prompt, to_evidence
from worldforge.retrieval.hybrid import HybridRetriever
from worldforge.retrieval.sparse import SparseRetriever
from worldforge.types import Document, ToolContext

PLANNERS = {"rule": RuleBasedPlanner}

_SYSTEM_PROMPT = (
    "你是检索增强助手。只依据给定材料回答；材料不足时明确说不知道。\n"
    "材料区块：\n"
)


class Pipeline:
    """门面：所有跨层操作都从这里进入，入口本身不含业务逻辑。"""

    def __init__(self, config: Config, parts: dict[str, Any]) -> None:
        self.config = config
        self.parts = parts
        self.metrics = Metrics()

    @property
    def tracer(self) -> Tracer:
        return self.parts["tracer"]

    @property
    def degraded(self) -> tuple[str, ...]:
        return degraded()

    # ---------- 文档 ----------

    def ingest(self, doc_id: str, text: str, meta: dict[str, Any] | None = None) -> int:
        document = Document(doc_id=doc_id, text=text, meta=meta or {})
        chunks = self.parts["chunker"].chunk(document)
        with self.tracer.span("ingest", doc_id=doc_id, chunks=len(chunks)):
            return self.parts["ingestor"].ingest(document, chunks)

    def drop(self, doc_id: str) -> int:
        return self.parts["ingestor"].drop(doc_id)

    def documents(self, limit: int = 100, offset: int = 0) -> list[str]:
        return self.parts["docstore"].list_documents(limit=limit, offset=offset)

    def document(self, doc_id: str) -> Document:
        found = self.parts["docstore"].get_document(doc_id)
        if found is None:
            raise DocNotFound(f"document '{doc_id}' not found")
        return found

    # ---------- 检索与问答 ----------

    def retrieve(self, query: str, top_k: int | None = None) -> list[Any]:
        with self.tracer.span("retrieve", query_len=len(query)):
            hits = self.parts["retriever"].search(query)
        return hits[: (top_k or self.config.rerank_top_k)]

    def answer(self, question: str) -> dict[str, Any]:
        """检索 -> 渲染证据 -> 生成 -> 接地核验。"""

        with self.tracer.span("answer", question_len=len(question)) as span:
            hits = self.parts["retriever"].search(question)
            evidences = dedupe_evidence(to_evidence(hits), limit=self.config.rerank_top_k)
            prompt = _SYSTEM_PROMPT + render_for_prompt(evidences) + "\n\n问题：" + question
            completion = self.parts["llm"].complete(prompt, seed=self.config.seed)
            texts = [e.text for e in evidences]
            span["hit_count"] = len(hits)
            self.metrics.incr("query_total")
        return {
            "answer": completion.text,
            "model_id": completion.model_id,
            "evidence": [
                {"chunk_id": e.chunk_id, "doc_id": e.doc_id, "score": round(e.score, 6)} for e in evidences
            ],
            "grounding": grounding_score(completion.text, texts),
            "faithfulness": faithfulness(completion.text, texts),
        }

    def chat(self, messages: Sequence[Any]) -> dict[str, Any]:
        completion = self.parts["llm"].chat(list(messages), seed=self.config.seed)
        return {"text": completion.text, "model_id": completion.model_id, "finish_reason": completion.finish_reason}

    def act(self, utterance: str) -> dict[str, Any]:
        """工具路由与执行，供 /v1/chat/completions 的 function-calling 分支使用。"""

        ctx = ToolContext(trace_id=self.tracer._observer.trace_id(), seed=self.config.seed, observer=self.parts["observer"])  # noqa: SLF001
        run: ReactRun = self.parts["agent"].run(utterance, ctx)
        self.metrics.incr("agent_runs")
        return {
            "answer": run.answer,
            "converged": run.converged,
            "tool_calls": run.tool_calls,
            "steps": run.steps,
        }

    def plan(self, goal: str) -> Any:
        return self.parts["planner"].plan(goal, self.parts["tools"].specs(), {})

    # ---------- 运维 ----------

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "seed": self.config.seed,
            "backends": {
                "embedder": self.config.embedder,
                "vectorstore": self.config.vectorstore,
                "lexical": self.config.lexical,
                "reranker": self.config.reranker,
                "llm": self.config.llm,
                "docstore": self.config.docstore,
                "observer": self.config.observer,
                "chunker": self.config.chunker,
            },
            "degraded": list(self.degraded),
            "chunks": self.parts["docstore"].count_chunks(),
        }

    def observability_snapshot(self) -> dict[str, Any]:
        return self.tracer.flush()

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot()

    def reset(self) -> None:
        self.parts["vectorstore"].clear()
        for doc_id in self.parts["docstore"].list_documents(limit=10**6):
            self.parts["docstore"].drop_document(doc_id)


def build_pipeline(config: Config | None = None) -> Pipeline:
    """按配置装配全部 Port；缺依赖时静默回退默认实现并记录降级标签。"""

    cfg = config or Config.from_env()
    seed_everything(cfg.seed)

    embedder = pick(cfg.embedder, EMBEDDERS, DEFAULTS["embedder"], "embedder", cfg)
    vectorstore = pick(cfg.vectorstore, VECTORSTORES, DEFAULTS["vectorstore"], "vectorstore", cfg)
    lexical = pick(cfg.lexical, LEXICALS, DEFAULTS["lexical"], "lexical", cfg)
    reranker = pick(cfg.reranker, RERANKERS, DEFAULTS["reranker"], "reranker", cfg)
    llm = pick(cfg.llm, LLMS, DEFAULTS["llm"], "llm", cfg)
    docstore = pick(cfg.docstore, DOCSTORES, DEFAULTS["docstore"], "docstore", cfg)
    observer = pick(cfg.observer, OBSERVERS, DEFAULTS["observer"], "observer", cfg)
    chunker = pick(cfg.chunker, CHUNKERS, DEFAULTS["chunker"], "chunker", cfg)
    planner_factory = PLANNERS.get(cfg.planner, RuleBasedPlanner)
    planner = planner_factory()

    retriever = HybridRetriever(
        sparse=SparseRetriever(lexical, bilingual=cfg.bilingual),
        dense=DenseRetriever(embedder, vectorstore),
        reranker=reranker,
        hybrid=cfg.hybrid,
        fusion=cfg.fusion,
        rrf_k=cfg.rrf_k,
        top_k=cfg.top_k,
        rerank_top_k=cfg.rerank_top_k,
    )
    tools = ToolRegistry(builtin_tools(retriever, top_k=cfg.rerank_top_k))
    from worldforge.agents.react import ReActAgent  # 局部导入避免 agents <-> pipeline 循环

    return Pipeline(
        cfg,
        {
            "embedder": embedder,
            "vectorstore": vectorstore,
            "lexical": lexical,
            "reranker": reranker,
            "llm": llm,
            "docstore": docstore,
            "observer": observer,
            "chunker": chunker,
            "planner": planner,
            "retriever": retriever,
            "tools": tools,
            "ingestor": Ingestor(docstore, embedder, vectorstore, lexical, max_chunks=cfg.max_chunks),
            "tracer": Tracer(observer),
            "agent": ReActAgent(tools, llm, max_steps=4),
        },
    )


__all__ = ["Pipeline", "build_pipeline"]
__author__ = "晨星"
