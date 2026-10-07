"""摄入编排与分块继承。"""

from __future__ import annotations

import pytest

from worldforge.config import Config
from worldforge.errors import IngestError
from worldforge.pipeline import build_pipeline
from worldforge.types import Document

# 刻意非均匀：等长语料会让分块边界总落在同一位置，标题继承路径覆盖不到
DOC_TEXT = """# 第一章 总述

Transformer 是一种完全基于自注意力的神经网络架构，它在 2017 年由 Vaswani 等人提出。
它用多头注意力同时建模序列中任意两个位置之间的依赖关系，因此不再依赖循环结构。

## 1.1 位置编码

自注意力本身不感知顺序，位置编码补足了这个缺陷。正弦位置编码是确定性方案，
可学习位置编码则把位置当作参数与embedding 一起训练。两种方案在原始论文里效果接近。

# 第二章 BM25

BM25 是一种经典的相关性排序函数，由 Robertson 提出并命名，常被称为 Okapi BM25。
它用逆文档频率衡量词项区分度，并用文档长度归一化抑制长文档天然占便宜的偏置。
"""


@pytest.fixture()
def pipeline():
    return build_pipeline(Config(chunk_size=160, chunk_overlap=20))


def test_ingest_replaces_instead_of_upsert(pipeline):
    pipeline.ingest("d1", DOC_TEXT)
    first = pipeline.parts["docstore"].count_chunks()
    assert first > 1
    pipeline.ingest("d1", "只有一句话的短文档。")
    assert pipeline.parts["docstore"].chunks_of("d1").__len__() == 1
    # 旧修订的孤儿分块不应再能被检索到
    hits = pipeline.retrieve("位置编码 自学习 可学习位置编码", top_k=5)
    assert all(h.meta.get("doc_id") != "d1" or h.chunk_id == "d1#0" for h in hits)


def test_heading_line_becomes_its_own_chunk(pipeline):
    chunks = pipeline.parts["chunker"].chunk(Document(doc_id="x", text=DOC_TEXT))
    texts = [c.text for c in chunks]
    assert any(t.strip().startswith("# 第一章") for t in texts)
    assert any("Vaswani" in t for t in texts), "标题段落把正文吞掉了"


def test_body_chunk_inherits_heading_at_start(pipeline):
    chunks = pipeline.parts["chunker"].chunk(Document(doc_id="x", text=DOC_TEXT))
    bodies = [c for c in chunks if "Vaswani" in c.text]
    assert bodies, "未找到正文分块"
    assert bodies[0].meta["heading_path"].startswith("第一章")


def test_chunk_ordinals_are_contiguous(pipeline):
    chunks = pipeline.parts["chunker"].chunk(Document(doc_id="x", text=DOC_TEXT))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_drop_removes_from_all_three_backends(pipeline):
    pipeline.ingest("d1", DOC_TEXT)
    assert pipeline.drop("d1") > 0
    assert pipeline.parts["docstore"].get_document("d1") is None
    assert pipeline.parts["docstore"].chunks_of("d1") == []
    assert pipeline.parts["vectorstore"].count() == 0


def test_ingest_rejects_empty_document(pipeline):
    with pytest.raises(IngestError):
        pipeline.ingest("empty", "   \n  ")


def test_evidence_render_is_single_line_per_chunk(pipeline):
    from worldforge.core.text import parse_evidence_lines, render_evidence
    from worldforge.retrieval.evidence import to_evidence

    pipeline.ingest("d1", DOC_TEXT)
    hits = pipeline.retrieve("BM25 逆文档频率", top_k=3)
    rendered = render_evidence(to_evidence(hits))
    assert len(parse_evidence_lines(rendered)) == len(hits)
    for line in parse_evidence_lines(rendered):
        assert "\n" not in line
