"""llama.cpp 本地推理后端（可选 extras: llm-llamacpp）。

CPU-only 环境里唯一能真跑大模型的路径，因此是本仓库推荐的本地推理后端。

作者：晨星
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from worldforge.errors import BackendUnavailable, LLMError
from worldforge.types import Completion, Message


class LlamaCppLLM:
    """llama-cpp-python 封装。

    注意发行名是 `llama-cpp-python`，import 名是 `llama_cpp`，用 importlib 查
    发行名会误报缺失。
    """

    def __init__(
        self,
        model_path: str = "",
        model_id: str = "local-gguf",
        n_ctx: int = 4096,
        n_threads: int = 4,
    ) -> None:
        try:
            from llama_cpp import Llama  # noqa: PLC0415 - 惰性导入是本文件的核心约束
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise BackendUnavailable(f"llama-cpp-python is not installed: {exc}") from exc
        resolved = model_path or self._discover()
        if not resolved:
            raise BackendUnavailable("no GGUF model found; set WORLDFORGE_LLM_MODEL to a file path")
        self._llama = Llama(model_path=resolved, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
        self._model = model_id
        self._path = resolved

    @staticmethod
    def _discover() -> str:
        candidates = sorted(Path("models").glob("*.gguf"))
        return str(candidates[0]) if candidates else ""

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
        try:
            out = self._llama.create_chat_completion(
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=list(stop) if stop else None,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - 第三方推理后端异常种类不可控
            raise LLMError(f"llama.cpp completion failed: {exc}") from exc
        choice = out["choices"][0]
        usage = out.get("usage", {})
        return Completion(
            text=choice["message"]["content"] or "",
            model_id=self._model,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Iterator[str]:
        try:
            for piece in self._llama.create_chat_completion(
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                seed=seed,
            ):
                delta = piece["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"llama.cpp stream failed: {exc}") from exc


__all__ = ["LlamaCppLLM"]
__author__ = "晨星"
