from __future__ import annotations

import asyncio
import json
import logging

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from common.config import GatewaySettings
from common.schemas import (
    ClientMessageType,
    ErrorMessage,
    StartMessage,
)
from gateway.audio_utils import normalize_audio
from gateway.session import SessionManager

logger = logging.getLogger(__name__)

settings = GatewaySettings()
app = FastAPI(title="Smart Transcriptor Gateway")
manager = SessionManager(max_sessions=settings.max_sessions)


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": manager.active_count}


@app.websocket("/audio")
async def audio_endpoint(ws: WebSocket):
    await ws.accept()
    stream_id: str | None = None
    try:
        # Expect a start message first (text frame)
        raw = await ws.receive_text()
        msg = json.loads(raw)
        if msg.get("type") != ClientMessageType.start:
            await ws.send_text(ErrorMessage(stream_id="", detail="Expected start message").model_dump_json())
            await ws.close()
            return

        start = StartMessage(**msg)
        stream_id = start.stream_id

        session = await manager.create(
            stream_id=stream_id,
            client_ws=ws,
            sample_rate=start.sample_rate,
            channels=start.channels,
            encoding=start.encoding,
            language=start.language,
        )

        # Open upstream connection to ASR service
        asr_ws = await websockets.connect(
            settings.asr_ws_url,
            ping_interval=30,
            ping_timeout=300,
            close_timeout=300,
        )
        session.asr_ws = asr_ws

        # Forward start message to ASR
        await asr_ws.send(raw)

        # Relay ASR responses back to client in background
        relay_task = asyncio.create_task(_relay_asr_to_client(asr_ws, ws, stream_id))

        # Main loop: receive audio/end from client
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == ClientMessageType.end:
                        await asr_ws.send(message["text"])
                        # Wait for ASR to finish processing and relay task to complete
                        try:
                            await relay_task
                        except asyncio.CancelledError:
                            pass
                        break
                elif "bytes" in message:
                    audio = normalize_audio(
                        message["bytes"],
                        input_sample_rate=session.sample_rate,
                        input_channels=session.channels,
                        input_encoding=session.encoding,
                    )
                    await asr_ws.send(audio)
        finally:
            relay_task.cancel()
            try:
                await relay_task
            except asyncio.CancelledError:
                pass
            await asr_ws.close()

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", stream_id)
    except RuntimeError as exc:
        logger.warning("Session error: %s", exc)
        await ws.send_text(ErrorMessage(stream_id=stream_id or "", detail=str(exc)).model_dump_json())
    except Exception:
        logger.exception("Unexpected error in audio endpoint")
    finally:
        if stream_id:
            await manager.remove(stream_id)


async def _relay_asr_to_client(asr_ws, client_ws: WebSocket, stream_id: str):
    """Forward messages from ASR service back to the client."""
    try:
        async for message in asr_ws:
            if isinstance(message, str):
                await client_ws.send_text(message)
            else:
                await client_ws.send_bytes(message)
    except websockets.ConnectionClosed:
        logger.info("ASR connection closed for %s", stream_id)
    except Exception:
        logger.exception("Relay error for %s", stream_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
