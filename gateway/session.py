from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class Session:
    stream_id: str
    client_ws: WebSocket
    asr_ws: object | None = None  # websockets client connection
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm_s16le"
    language: str | None = None


class SessionManager:
    def __init__(self, max_sessions: int = 10) -> None:
        self._max = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self, stream_id: str, client_ws: WebSocket, **kwargs) -> Session:
        async with self._lock:
            if len(self._sessions) >= self._max:
                raise RuntimeError(f"Max sessions ({self._max}) reached")
            if stream_id in self._sessions:
                raise RuntimeError(f"Session {stream_id} already exists")
            session = Session(stream_id=stream_id, client_ws=client_ws, **kwargs)
            self._sessions[stream_id] = session
            logger.info("Session created: %s (%d active)", stream_id, len(self._sessions))
            return session

    async def remove(self, stream_id: str) -> None:
        async with self._lock:
            self._sessions.pop(stream_id, None)
            logger.info("Session removed: %s (%d active)", stream_id, len(self._sessions))

    def get(self, stream_id: str) -> Session | None:
        return self._sessions.get(stream_id)

    @property
    def active_count(self) -> int:
        return len(self._sessions)
