"""End-to-end tests — require running services or are skipped."""

import json
import os

import pytest

E2E = os.environ.get("RUN_E2E", "").lower() in ("1", "true", "yes")
pytestmark = pytest.mark.skipif(not E2E, reason="E2E tests disabled (set RUN_E2E=1)")


@pytest.mark.asyncio
async def test_gateway_stream():
    import websockets
    import numpy as np

    uri = os.environ.get("GATEWAY_WS_URL", "ws://localhost:8000/audio")
    async with websockets.connect(uri) as ws:
        # Send start
        await ws.send(json.dumps({
            "type": "start",
            "stream_id": "e2e-test",
            "sample_rate": 16000,
            "encoding": "pcm_s16le",
            "channels": 1,
        }))

        # Send 2 seconds of silence as audio
        silence = np.zeros(32000, dtype=np.int16).tobytes()
        chunk_size = 16000 * 2  # 1s chunks in bytes
        for i in range(0, len(silence), chunk_size):
            await ws.send(silence[i : i + chunk_size])

        # Send end
        await ws.send(json.dumps({"type": "end", "stream_id": "e2e-test"}))

        # Collect responses
        messages = []
        async for msg in ws:
            data = json.loads(msg)
            messages.append(data)
            if data.get("type") == "transcript_complete":
                break

        assert any(m["type"] == "transcript_complete" for m in messages)


@pytest.mark.asyncio
async def test_slm_analyze():
    import httpx

    url = os.environ.get("SLM_URL", "http://localhost:8002/analyze-transcript")
    payload = {
        "stream_id": "e2e-test",
        "segments": [
            {
                "status": "final",
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "We need to discuss the quarterly targets.",
                "speaker": "SPEAKER_00",
            }
        ],
        "context": {"meeting_type": "standup"},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "risks" in data
