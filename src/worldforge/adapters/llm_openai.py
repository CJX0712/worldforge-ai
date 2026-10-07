"""OpenAI Chat Completions 后端（可选 extras: llm-openai）。

作者：晨星
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator, Sequence

from worldforge.errors import BackendUnavailable, LLMError
from worldforge.types import Completion, Message


class OpenAILLM:
    """OpenAI 兼容聊天补全，HTTP 走标准库 urllib。"""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
    ) -> None:
        key = os.environ.get(api_key_env, "")
        if not key:
            raise BackendUnavailable(f"environment variable {api_key_env} is empty")
        self._url = api_base.rstrip("/") + "/v1/chat/completions"
        self._key = key
        self._model = model_id
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        return self.chat(
            [Message(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Completion:
        body: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            body["stop"] = list(stop)
        if seed is not None:
            body["seed"] = seed
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMError(f"openai chat failed with status {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"openai chat unreachable: {exc}") from exc
        choice = data["choices"][0]
        return Completion(
            text=choice["message"]["content"],
            model_id=data.get("model", self._model),
            finish_reason=choice.get("finish_reason", "stop"),
            usage={k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int)},
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Iterator[str]:
        # 一次性取回再切块：单次往返保证增量顺序与最终 chat() 结果一致
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens, seed=seed).text


__all__ = ["OpenAILLM"]
__author__ = "晨星"
