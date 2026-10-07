"""抽取式默认语言模型实现。

作者：晨星
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence

from worldforge.core.seeding import seed_everything
from worldforge.core.sortutil import stable_sort
from worldforge.core.tokenizer import tokenize
from worldforge.types import Completion, Message

_CONTEXT_OPEN = "<kb-context>"
_CONTEXT_CLOSE = "</kb-context>"

# 英文句点只在后接空白或位于结尾时才切分，避免把版本号 Python 3.10 截断
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])|(?<=[.!?])(?=\s|$)")
_CITATION_RE = re.compile(r"\[[^\]\n]*#\d+\]")


class ExtractiveMockLLM:
    """从证据中按 IDF 权重抽取原句，输出必带引用标记。"""

    name = "mock"

    def __init__(self, model_id: str = "worldforge-mock-1", max_tokens: int = 512, seed: int = 1337) -> None:
        self._model_id = model_id
        self._max_tokens = int(max_tokens)
        seed_everything(seed)

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        context = extract_context(prompt)
        query = extract_query(prompt)
        text = self._compose(context, query, min(max_tokens, self._max_tokens))
        if stop:
            for marker in stop:
                if marker in text:
                    text = text.split(marker)[0]
                    break
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(text)}
        return Completion(text=text, model_id=self.model_id, finish_reason="stop", usage=usage)

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
        allowed = {"system", "user", "assistant", "tool"}
        for message in messages:
            if message.role not in allowed:
                raise ValueError(f"非法 role: {message.role}")
        prompt = "\n".join(message.content for message in messages)
        return self.complete(prompt, temperature=temperature, max_tokens=max_tokens, stop=stop, seed=seed)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Iterator[str]:
        completion = self.chat(messages, temperature=temperature, max_tokens=max_tokens, seed=seed)
        for index in range(0, len(completion.text), 32):
            yield completion.text[index : index + 32]

    def _compose(self, context: str, query: str, max_tokens: int) -> str:
        units = split_context_units(context)
        if not units:
            return "未检索到可用证据。"
        query_tokens = set(tokenize(query))
        document_frequency = {
            token: sum(1 for item in units if token in set(tokenize(item["sentence"]))) for token in query_tokens
        }
        total = len(units)
        idf = {token: math.log1p((total - count + 0.5) / (count + 0.5)) for token, count in document_frequency.items()}
        scored = []
        for order, item in enumerate(units):
            tokens = set(tokenize(item["sentence"]))
            score = math.fsum(idf.get(token, 0.0) for token in tokens if token in query_tokens)
            scored.append({"order": order, "sentence": item["sentence"], "chunk_id": item["chunk_id"], "score": score})
        ordered = stable_sort(scored, key=lambda item: item["score"], tie_key=lambda item: str(item["order"]))
        positive = [item for item in ordered if item["score"] > 0.0]
        selected = sorted((positive or ordered)[:3], key=lambda item: item["order"])
        parts = [f"{item['sentence']} [{item['chunk_id']}#{item['order']}]" for item in selected]
        text = " ".join(parts)
        return text[:max_tokens]


def split_sentences(text: str) -> list[str]:
    """按中文句末与英文句末切分，英文句点需后接空白或结尾。"""

    if not text:
        return []
    stripped = _CITATION_RE.sub("", text)
    pieces = [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(stripped)]
    return [piece for piece in pieces if piece]


def extract_context(prompt: str) -> str:
    """抽取全局唯一限定名包裹的上下文块，指令文本不会被误取。"""

    start = prompt.find(_CONTEXT_OPEN)
    end = prompt.find(_CONTEXT_CLOSE)
    if start == -1 or end == -1 or end < start:
        return ""
    return prompt[start + len(_CONTEXT_OPEN) : end]


def split_context_units(context: str) -> list[dict[str, str]]:
    """把上下文块拆成句子单元，并保留所属 chunk_id。"""

    units: list[dict[str, str]] = []
    for line in context.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        chunk_id = ""
        if stripped.startswith("["):
            end = stripped.find("]")
            if end != -1:
                chunk_id = stripped[1:end]
                stripped = stripped[end + 1 :].strip()
        for sentence in split_sentences(stripped):
            units.append({"sentence": sentence, "chunk_id": chunk_id})
    return units


def extract_query(prompt: str) -> str:
    """抽取上下文块之后的提问文本。"""

    end = prompt.find(_CONTEXT_CLOSE)
    if end == -1:
        return prompt
    return prompt[end + len(_CONTEXT_CLOSE) :].strip()


__all__ = ["ExtractiveMockLLM", "split_sentences", "split_context_units", "extract_context", "extract_query"]
__author__ = "晨星"
