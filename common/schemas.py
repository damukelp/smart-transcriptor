from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


# --- WebSocket messages: client ↔ gateway ↔ ASR ---

class ClientMessageType(str, Enum):
    start = "start"
    audio = "audio"
    end = "end"


class StartMessage(BaseModel):
    type: ClientMessageType = ClientMessageType.start
    stream_id: str
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    channels: int = 1
    language: Optional[str] = None
    diarize: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


class AudioMessage(BaseModel):
    type: ClientMessageType = ClientMessageType.audio
    stream_id: str
    # audio payload sent as binary frame, not in JSON


class EndMessage(BaseModel):
    type: ClientMessageType = ClientMessageType.end
    stream_id: str


class SegmentStatus(str, Enum):
    partial = "partial"
    final = "final"


class TranscriptSegment(BaseModel):
    status: SegmentStatus
    segment_id: int
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None


class ServerMessageType(str, Enum):
    segment = "segment"
    transcript_complete = "transcript_complete"
    error = "error"


class SegmentMessage(BaseModel):
    type: ServerMessageType = ServerMessageType.segment
    stream_id: str
    segment: TranscriptSegment


class TranscriptCompleteMessage(BaseModel):
    type: ServerMessageType = ServerMessageType.transcript_complete
    stream_id: str
    segments: list[TranscriptSegment]
    speaker_map: dict[str, str] = {}


class ErrorMessage(BaseModel):
    type: ServerMessageType = ServerMessageType.error
    stream_id: str
    detail: str


# --- SLM request / response ---

class MeetingContext(BaseModel):
    meeting_type: str = "general"
    department: Optional[str] = None
    participants: list[str] = []


class AnalyzeRequest(BaseModel):
    stream_id: str
    segments: list[TranscriptSegment]
    context: MeetingContext = MeetingContext()


class RiskItem(BaseModel):
    category: str
    description: str
    severity: str  # low / medium / high


class AnalyzeResponse(BaseModel):
    stream_id: str
    summary: str
    key_points: list[str]
    action_items: list[str]
    risks: list[RiskItem]
