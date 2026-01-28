from __future__ import annotations

import json
import logging

import httpx

from common.config import SLMSettings

logger = logging.getLogger(__name__)


async def chat_completion(
    messages: list[dict[str, str]],
    settings: SLMSettings | None = None,
) -> str:
    """Call Ollama /api/chat and return the assistant message content."""
    settings = settings or SLMSettings()
    url = f"{settings.ollama_url}/api/chat"

    payload = {
        "model": settings.model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": settings.temperature,
            "num_predict": settings.max_tokens,
        },
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
