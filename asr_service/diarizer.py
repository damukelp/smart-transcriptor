from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_pipeline = None
_pipeline_failed = False


def get_pipeline(hf_token: str = "", clustering_threshold: float | None = None):
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

            # Tune clustering threshold for better speaker separation
            if clustering_threshold is not None:
                params = _pipeline.parameters(instantiated=True)
                params["clustering"]["threshold"] = clustering_threshold
                _pipeline.instantiate(params)
                logger.info("Diarization clustering threshold set to %.3f", clustering_threshold)

            logger.info("Diarization pipeline loaded")
        except Exception:
            logger.warning("pyannote not available; diarization disabled", exc_info=True)
            _pipeline_failed = True
    return _pipeline


class SlidingWindowDiarizer:
    """Maintains audio buffer for session-consistent diarization."""

    MIN_OVERLAP_SECONDS = 0.3  # Minimum overlap to assign a speaker

    def __init__(
        self,
        window_seconds: float = 15.0,
        sample_rate: int = 16000,
        hf_token: str = "",
        clustering_threshold: float | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.hf_token = hf_token
        self.clustering_threshold = clustering_threshold
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._buffer = np.array([], dtype=np.float32)
        self._offset = 0.0  # start time of the buffer

    @property
    def buffer_duration(self) -> float:
        return len(self._buffer) / self.sample_rate

    def add_audio(self, audio: np.ndarray) -> None:
        self._buffer = np.concatenate([self._buffer, audio])

    def diarize(self) -> dict[tuple[float, float], str]:
        """Run diarization on the current buffer. Returns {(start, end): speaker_label}."""
        pipeline = get_pipeline(self.hf_token, self.clustering_threshold)
        if pipeline is None or len(self._buffer) < self.sample_rate:
            return {}

        try:
            import torch

            waveform = torch.from_numpy(self._buffer).unsqueeze(0)
            audio_input = {"waveform": waveform, "sample_rate": self.sample_rate}

            # Build kwargs for speaker count hints
            kwargs: dict = {}
            if self.min_speakers is not None:
                kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                kwargs["max_speakers"] = self.max_speakers

            if kwargs:
                logger.info("Diarization speaker hints: %s", kwargs)

            output = pipeline(audio_input, **kwargs)

            result: dict[tuple[float, float], str] = {}
            if hasattr(output, "itertracks"):
                for turn, _, speaker in output.itertracks(yield_label=True):
                    abs_start = round(self._offset + turn.start, 3)
                    abs_end = round(self._offset + turn.end, 3)
                    result[(abs_start, abs_end)] = speaker
            elif hasattr(output, "speaker_diarization"):
                for turn, speaker in output.speaker_diarization:
                    abs_start = round(self._offset + turn.start, 3)
                    abs_end = round(self._offset + turn.end, 3)
                    result[(abs_start, abs_end)] = speaker
            return result
        except Exception:
            logger.exception("Diarization failed")
            return {}

    def assign_speaker(self, start: float, end: float, diarization: dict[tuple[float, float], str]) -> Optional[str]:
        """Find the best matching speaker for a transcript segment.

        Uses overlap-weighted matching with a minimum overlap threshold
        to avoid spurious assignments at segment boundaries.
        """
        if not diarization:
            return None

        seg_duration = end - start
        best_speaker = None
        best_overlap = 0.0

        for (d_start, d_end), speaker in diarization.items():
            overlap_start = max(start, d_start)
            overlap_end = min(end, d_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        # Require minimum overlap to avoid wrong assignments at boundaries
        if best_overlap < self.MIN_OVERLAP_SECONDS and seg_duration > self.MIN_OVERLAP_SECONDS:
            return None

        return best_speaker
