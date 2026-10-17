"""OpenAI 兼容嵌入后端（可选 extras: vector-embed-openai）。

只用标准库 urllib，不引入 openai SDK，避免为一个 HTTP 调用拖进一整棵依赖树。

作者：晨星
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence

from worldforge.core.mathx import l2_normalize
from worldforge.errors import BackendUnavailable


class OpenAIEmbedder:
    """OpenAI `/v1/embeddings` 兼容端点。"""

    def __init__(
        self,
        model_id: str = "text-embedding-3-small",
        api_base: str = "https://api.openai.com",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 20.0,
    ) -> None:
        key = os.environ.get(api_key_env, "")
        if not key:
            raise BackendUnavailable(f"environment variable {api_key_env} is empty")
        self._url = api_base.rstrip("/") + "/v1/embeddings"
        self._key = key
        self._model = model_id
        self._dim = 0
        self._timeout = timeout

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self._model, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise BackendUnavailable(f"openai embeddings request failed: {exc}") from exc
        vectors = [l2_normalize(item["embedding"]) for item in body["data"]]
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


__all__ = ["OpenAIEmbedder"]
__author__ = "晨星"
