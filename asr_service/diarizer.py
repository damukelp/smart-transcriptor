from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_pipeline = None
_pipeline_failed = False


def get_pipeline(hf_token: str = ""):
    """Lazily load pyannote speaker-diarization pipeline."""
    global _pipeline, _pipeline_failed
    if _pipeline_failed:
        return None
    if _pipeline is None:
        try:
            from pyannote.audio import Pipeline

            import os
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            logger.info("Loading pyannote diarization pipeline")
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
            )
            logger.info("Diarization pipeline loaded")
        except Exception:
            logger.warning("pyannote not available; diarization disabled", exc_info=True)
            _pipeline_failed = True
    return _pipeline


class SlidingWindowDiarizer:
    """Maintains a sliding window of audio for session-consistent diarization."""

    def __init__(self, window_seconds: float = 15.0, sample_rate: int = 16000, hf_token: str = ""):
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.hf_token = hf_token
        self._buffer = np.array([], dtype=np.float32)
        self._offset = 0.0  # start time of the buffer
        self._speaker_embeddings: dict[str, np.ndarray] = {}
        self._next_speaker_id = 0

    @property
    def buffer_duration(self) -> float:
        return len(self._buffer) / self.sample_rate

    def add_audio(self, audio: np.ndarray) -> None:
        self._buffer = np.concatenate([self._buffer, audio])
        max_samples = int(self.window_seconds * self.sample_rate)
        if len(self._buffer) > max_samples:
            trim = len(self._buffer) - max_samples
            self._offset += trim / self.sample_rate
            self._buffer = self._buffer[trim:]

    def diarize(self) -> dict[tuple[float, float], str]:
        """Run diarization on the current window. Returns {(start, end): speaker_label}."""
        pipeline = get_pipeline(self.hf_token)
        if pipeline is None or len(self._buffer) < self.sample_rate:
            return {}

        try:
            import torch

            waveform = torch.from_numpy(self._buffer).unsqueeze(0)
            audio_input = {"waveform": waveform, "sample_rate": self.sample_rate}
            output = pipeline(audio_input)

            result: dict[tuple[float, float], str] = {}
            for turn, _, speaker in output.itertracks(yield_label=True):
                abs_start = round(self._offset + turn.start, 3)
                abs_end = round(self._offset + turn.end, 3)
                result[(abs_start, abs_end)] = speaker
            return result
        except Exception:
            logger.exception("Diarization failed")
            return {}

    def assign_speaker(self, start: float, end: float, diarization: dict[tuple[float, float], str]) -> Optional[str]:
        """Find the best matching speaker for a transcript segment."""
        if not diarization:
            return None

        best_speaker = None
        best_overlap = 0.0
        for (d_start, d_end), speaker in diarization.items():
            overlap_start = max(start, d_start)
            overlap_end = min(end, d_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        return best_speaker
