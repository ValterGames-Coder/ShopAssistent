import json
import math

import httpx

from .config import Settings
from .models import Decision, Hit


class ModelError(RuntimeError):
    pass


def validate_vectors(value, count: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != count or not value:
        raise ModelError("Wrong embedding count")
    dimension = None
    for vector in value:
        if not isinstance(vector, list) or not 1 <= len(vector) <= 16000:
            raise ModelError("Invalid embedding dimension")
        dimension = dimension or len(vector)
        if len(vector) != dimension or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in vector
        ) or not any(vector):
            raise ModelError("Invalid embedding vector")
    return value


class Ollama:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(settings.model_timeout_seconds, connect=10),
            limits=httpx.Limits(max_connections=4),
        )

    async def close(self):
        await self.client.aclose()

    async def _post(self, path: str, payload: dict) -> dict:
        response = await self.client.post(path, json=payload)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict) or value.get("error"):
            raise ModelError("Model service returned an invalid response")
        return value

    async def embed(self, texts: list[str]) -> list[list[float]]:
        data = await self._post("/api/embed", {
            "model": self.settings.embedding_model, "input": texts,
            "truncate": False, "keep_alive": "5m",
        })
        return validate_vectors(data.get("embeddings"), len(texts))

    async def decide(self, question: str, hits: list[Hit], prompt: str) -> Decision:
        # Sources have their own message; the user cannot add trusted documents.
        context = [{"id": h.document.id, "text": h.document.text} for h in hits]
        schema = Decision.model_json_schema()
        if context:
            schema["properties"]["source_ids"]["items"] = {
                "type": "string", "enum": [d["id"] for d in context]
            }
        data = await self._post("/api/chat", {
            "model": self.settings.chat_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "system", "content": "Доверенные фрагменты базы знаний:\n" + json.dumps(context, ensure_ascii=False)},
                {"role": "user", "content": question},
            ],
            "format": schema, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 512, "num_ctx": 4096},
            "keep_alive": "5m",
        })
        if data.get("done") is not True or data.get("done_reason") == "length":
            raise ModelError("Incomplete model response")
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ModelError("Missing model response")
        return Decision.model_validate_json(message["content"])
