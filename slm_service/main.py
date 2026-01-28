from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException

from common.config import SLMSettings
from common.schemas import AnalyzeRequest, AnalyzeResponse, RiskItem
from slm_service.ollama_client import chat_completion
from slm_service.prompts import SYSTEM_PROMPT, build_user_prompt, format_transcript

logger = logging.getLogger(__name__)

settings = SLMSettings()
app = FastAPI(title="SLM Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze-transcript", response_model=AnalyzeResponse)
async def analyze_transcript(req: AnalyzeRequest):
    formatted = format_transcript([s.model_dump() for s in req.segments])
    user_prompt = build_user_prompt(
        formatted,
        meeting_type=req.context.meeting_type,
        department=req.context.department,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await chat_completion(messages, settings)
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("SLM returned invalid JSON: %s", raw)
        raise HTTPException(status_code=502, detail="SLM returned invalid JSON")
    except Exception:
        logger.exception("SLM call failed")
        raise HTTPException(status_code=502, detail="SLM service unavailable")

    return AnalyzeResponse(
        stream_id=req.stream_id,
        summary=result.get("summary", ""),
        key_points=result.get("key_points", []),
        action_items=result.get("action_items", []),
        risks=[RiskItem(**r) for r in result.get("risks", [])],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
