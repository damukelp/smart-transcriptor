from __future__ import annotations

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from common.config import ASRSettings
from common.schemas import (
    ClientMessageType,
    SegmentMessage,
    SegmentStatus,
    ServerMessageType,
    TranscriptCompleteMessage,
    TranscriptSegment,
    ErrorMessage,
    StartMessage,
)
from asr_service.session import StreamSession
from asr_service.transcriber import transcribe_chunk, get_model

logger = logging.getLogger(__name__)

settings = ASRSettings()
app = FastAPI(title="ASR Service")


@app.on_event("startup")
async def startup():
    get_model(settings)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/stream")
async def stream_endpoint(ws: WebSocket):
    await ws.accept()
    session: StreamSession | None = None

    try:
        # Expect start message
        raw = await ws.receive_text()
        msg = json.loads(raw)
        if msg.get("type") != ClientMessageType.start:
            await ws.send_text(ErrorMessage(stream_id="", detail="Expected start message").model_dump_json())
            await ws.close()
            return

        start = StartMessage(**msg)
        session = StreamSession(
            stream_id=start.stream_id,
            settings=settings,
            language=start.language,
        )
        logger.info("ASR session started: %s", start.stream_id)

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                session.add_audio(message["bytes"])
                logger.info("Audio received: buffer=%d samples", len(session._audio_buffer))

                # Process complete chunks (transcription only, diarization at end)
                while session.has_chunk():
                    chunk, offset = session.pop_chunk()
                    logger.info("Transcribing chunk at offset=%.1fs", offset)
                    segments = transcribe_chunk(chunk, offset, session.language)
                    logger.info("Transcribed %d segments", len(segments))

                    for seg in segments:
                        ts = TranscriptSegment(
                            status=SegmentStatus.partial,
                            segment_id=session.next_segment_id(),
                            start_time=seg.start_time,
                            end_time=seg.end_time,
                            text=seg.text,
                            speaker=None,
                            confidence=seg.confidence,
                        )
                        session.all_partial_segments.append(ts)
                        await ws.send_text(
                            SegmentMessage(stream_id=start.stream_id, segment=ts).model_dump_json()
                        )

            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == ClientMessageType.end:
                    break

        # Flush remaining audio and run diarization once on full buffer
        if session:
            remainder = session.flush()
            if remainder:
                chunk, offset = remainder
                segments = transcribe_chunk(chunk, offset, session.language)
                for seg in segments:
                    ts = TranscriptSegment(
                        status=SegmentStatus.final,
                        segment_id=session.next_segment_id(),
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        text=seg.text,
                        speaker=None,
                        confidence=seg.confidence,
                    )
                    session.all_partial_segments.append(ts)

            # Run diarization once on the full audio buffer
            logger.info("Running diarization for %s...", session.stream_id)
            diarization = session.diarizer.diarize()
            logger.info("Diarization complete: %d turns", len(diarization))

            # Assign speakers to all segments
            all_segments = []
            for seg in session.all_partial_segments:
                speaker = session.diarizer.assign_speaker(
                    seg.start_time, seg.end_time, diarization
                )
                all_segments.append(TranscriptSegment(
                    status=SegmentStatus.final,
                    segment_id=seg.segment_id,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=seg.text,
                    speaker=speaker,
                    confidence=seg.confidence,
                ))

            complete = TranscriptCompleteMessage(
                stream_id=session.stream_id,
                segments=all_segments,
            )
            await ws.send_text(complete.model_dump_json())

    except WebSocketDisconnect:
        logger.info("ASR client disconnected: %s", session.stream_id if session else "unknown")
    except Exception as exc:
        logger.exception("ASR stream error: %s", exc)
        try:
            if session:
                await ws.send_text(
                    ErrorMessage(stream_id=session.stream_id, detail="Internal ASR error").model_dump_json()
                )
        except Exception:
            pass
    finally:
        logger.info("ASR session ended: %s", session.stream_id if session else "unknown")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
