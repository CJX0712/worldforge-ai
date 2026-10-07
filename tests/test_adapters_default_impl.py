"""默认实现必须是真实现，不能是 stub。"""

from __future__ import annotations

import pytest

from worldforge.adapters.chunker_recursive import RecursiveCharChunker
from worldforge.adapters.docstore_memory import MemoryDocStore
from worldforge.adapters.embedder_hashing import HashingEmbedder
from worldforge.adapters.lexical_bm25 import BM25Index
from worldforge.adapters.llm_mock import ExtractiveMockLLM
from worldforge.adapters.observer_memory import MemoryObserver
from worldforge.adapters.reranker_lexical import LexicalReranker
from worldforge.adapters.vectorstore_memory import InMemoryExactIndex
from worldforge.types import Chunk, Document, ToolContext

# 每篇都必须显式超过分块阈值，否则多分块路径根本没被覆盖
DOCS = (
    ("kb-a", "BM25 由 Robertson 提出。它用逆文档频率衡量词项区分度，并用文档长度归一化抑制长文档偏置，因此在小语料上依然稳健。"),
    ("kb-b", "Transformer 完全基于自注意力。它用多头注意力同时建模任意两个位置的依赖关系，不再依赖循环结构，长程依赖可在单层内建模。"),
    ("kb-c", "检索增强生成先召回相关文档再让模型生成答案。证据被压成单行注入提示，避免消费方按行解析时丢掉绝大部分上下文。"),
)


def _pipeline_docs():
    chunker = RecursiveCharChunker(chunk_size=120, overlap=20)
    embedder = HashingEmbedder(dim=128)
    store = InMemoryExactIndex(dim=embedder.dim, model_id=embedder.model_id)
    lexical = BM25Index()
    docstore = MemoryDocStore()
    for doc_id, text in DOCS:
        document = Document(doc_id=doc_id, text=text)
        chunks = chunker.chunk(document)
        docstore.put_document(document)
        docstore.put_chunks(chunks)
        store.upsert(
            [c.chunk_id for c in chunks],
            embedder.embed([c.text for c in chunks]),
            [{"doc_id": c.doc_id, "ordinal": c.ordinal, "text": c.text} for c in chunks],
        )
        lexical.index(chunks)
    return chunker, embedder, store, lexical, docstore


def test_default_chunker_produces_multiple_chunks():
    chunker, *_ = _pipeline_docs()
    document = Document(doc_id="long", text="\n".join(f"第{i}段。" + "填充内容。" * 60 for i in range(6)))
    chunks = chunker.chunk(document)
    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_default_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=128)
    a = embedder.embed_one("BM25 相关性排序")
    b = embedder.embed_one("BM25 相关性排序")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert len(a) == 128


def test_default_vectorstore_recalls_correct_document():
    *_, store, _, _ = _pipeline_docs()
    embedder = HashingEmbedder(dim=128)
    hits = store.search(embedder.embed_one("Robertson 提出的相关性排序函数"), top_k=3)
    assert hits, "默认向量库检索结果为空"
    assert hits[0].meta["doc_id"] == "kb-a"


def test_default_lexical_recalls_correct_document():
    *_, lexical, _ = _pipeline_docs()
    hits = lexical.search("Robertson 提出的相关性排序函数", top_k=3)
    assert hits, "默认稀疏索引检索结果为空"
    assert hits[0].meta["doc_id"] == "kb-a"


def test_default_reranker_orders_by_coverage():
    _, _, store, _, docstore = _pipeline_docs()
    candidates = [
        chunk
        for chunk in (docstore.chunks_of("kb-b") + docstore.chunks_of("kb-a"))
    ]
    from worldforge.types import Hit

    hits = [Hit(chunk_id=c.chunk_id, score=0.5, source="dense", meta={"text": c.text, "doc_id": c.doc_id}) for c in candidates]
    ranked = LexicalReranker().rerank("Robertson 相关性排序", hits, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].score >= ranked[1].score
    assert ranked[0].source == "rerank"


def test_default_mock_llm_extracts_from_evidence():
    llm = ExtractiveMockLLM()
    prompt = "指令文本\n<kb-context>\n[doc#0] BM25 由 Robertson 提出。\n</kb-context>\n\n问题：谁提出 BM25"
    out = llm.complete(prompt)
    assert "Robertson" in out.text
    assert "指令文本" not in out.text


def test_default_observer_records_latency_and_flushes():
    observer = MemoryObserver(buffer_size=4)
    with observer.span("demo", step=1):
        pass
    observer.metric("hits", 3.0)
    payload = observer.flush()
    assert payload["spans"][0]["name"] == "demo"
    assert "latency_ms" in payload["spans"][0]
    assert payload["metrics"][0]["value"] == 3.0
    assert observer.flush() == {"spans": [], "events": [], "metrics": []}


def test_default_docstore_is_replace_not_upsert():
    store = MemoryDocStore()
    store.put_chunks([Chunk("x#0", "x", 0, "旧"), Chunk("x#1", "x", 1, "旧尾巴")])
    assert store.count_chunks() == 2
    store.drop_document("x")
    store.put_chunks([Chunk("x#0", "x", 0, "新")])
    assert store.count_chunks() == 1


def test_tool_context_carries_seed_and_trace():
    ctx = ToolContext(trace_id="t1", seed=1337, observer=None)
    assert (ctx.trace_id, ctx.seed) == ("t1", 1337)


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_document_produces_no_chunks(text):
    assert RecursiveCharChunker().chunk(Document(doc_id="e", text=text)) == []
