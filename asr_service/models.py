"""Internal models for ASR service processing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    text: str
    start_time: float
    end_time: float
    confidence: float = 0.0


@dataclass
class DiarizedSegment:
    text: str
    start_time: float
    end_time: float
    speaker: str
    confidence: float = 0.0
