"""Ollama 本地推理后端（可选 extras: llm-ollama）。

作者：晨星
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence

from worldforge.errors import LLMError
from worldforge.types import Completion, Message


class OllamaLLM:
    """Ollama `/api/chat` 封装。

    不使用 httpx：它默认 trust_env=True，会把本机流量也送进系统 SOCKS 代理，
    访问 127.0.0.1 会被报 WinError 10054。标准库 urllib 不读代理环境变量。
    """

    def __init__(self, model_id: str = "qwen2.5:0.5b", api_base: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> None:
        self._url = api_base.rstrip("/") + "/api/chat"
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
        options: dict[str, object] = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        body: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        if stop:
            body["options"] = {**options, "stop": list(stop)}
        request = urllib.request.Request(
            self._url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMError(f"ollama chat failed with status {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"ollama unreachable: {exc}") from exc
        return Completion(
            text=data.get("message", {}).get("content", ""),
            model_id=data.get("model", self._model),
            finish_reason="stop" if data.get("done", True) else "length",
            usage={"prompt_tokens": int(data.get("prompt_eval_count", 0)), "completion_tokens": int(data.get("eval_count", 0))},
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Iterator[str]:
        body: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if seed is not None:
            body["options"] = {**body["options"], "seed": seed}  # type: ignore[dict-item]
        request = urllib.request.Request(
            self._url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise LLMError(f"ollama stream unreachable: {exc}") from exc


__all__ = ["OllamaLLM"]
__author__ = "晨星"
