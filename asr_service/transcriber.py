from __future__ import annotations

import logging
import numpy as np
from faster_whisper import WhisperModel

from common.config import ASRSettings
from asr_service.models import ChunkResult

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


def get_model(settings: ASRSettings | None = None) -> WhisperModel:
    global _model
    if _model is None:
        settings = settings or ASRSettings()
        logger.info("Loading faster-whisper model: %s", settings.model_size)
        _model = WhisperModel(
            settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        logger.info("Model loaded")
    return _model


def transcribe_chunk(
    audio: np.ndarray,
    offset: float,
    language: str | None = None,
) -> list[ChunkResult]:
    """Transcribe a numpy audio array (16kHz float32) and return segments."""
    model = get_model()
    segments, info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        beam_size=5,
    )
    results: list[ChunkResult] = []
    for seg in segments:
        results.append(
            ChunkResult(
                text=seg.text.strip(),
                start_time=round(offset + seg.start, 3),
                end_time=round(offset + seg.end, 3),
                confidence=round(seg.avg_logprob, 4) if seg.avg_logprob else 0.0,
            )
        )
    return results
