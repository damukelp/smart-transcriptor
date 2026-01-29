from __future__ import annotations

import numpy as np

from common.config import ASRSettings
from asr_service.diarizer import SlidingWindowDiarizer


class StreamSession:
    """Per-stream state: accumulates audio and tracks segments."""

    def __init__(self, stream_id: str, settings: ASRSettings, language: str | None = None):
        self.stream_id = stream_id
        self.settings = settings
        self.language = language
        self.sample_rate = 16000
        self.chunk_samples = int(settings.chunk_duration_s * self.sample_rate)

        self._audio_buffer = np.array([], dtype=np.float32)
        self._processed_time = 0.0
        self._segment_counter = 0

        self.diarizer = SlidingWindowDiarizer(
            window_seconds=settings.diarize_window_s,
            sample_rate=self.sample_rate,
            hf_token=settings.hf_token,
        )

        self.final_segments: list[dict] = []
        self.all_partial_segments: list = []

    def add_audio(self, pcm_bytes: bytes) -> None:
        """Append raw 16-bit PCM audio to the buffer."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer = np.concatenate([self._audio_buffer, audio])
        self.diarizer.add_audio(audio)

    def has_chunk(self) -> bool:
        return len(self._audio_buffer) >= self.chunk_samples

    def pop_chunk(self) -> tuple[np.ndarray, float]:
        """Pop a chunk of audio for transcription. Returns (audio, offset)."""
        chunk = self._audio_buffer[: self.chunk_samples]
        self._audio_buffer = self._audio_buffer[self.chunk_samples:]
        offset = self._processed_time
        self._processed_time += len(chunk) / self.sample_rate
        return chunk, offset

    def flush(self) -> tuple[np.ndarray, float] | None:
        """Return remaining audio if any."""
        if len(self._audio_buffer) > 0:
            chunk = self._audio_buffer
            self._audio_buffer = np.array([], dtype=np.float32)
            offset = self._processed_time
            self._processed_time += len(chunk) / self.sample_rate
            return chunk, offset
        return None

    def next_segment_id(self) -> int:
        sid = self._segment_counter
        self._segment_counter += 1
        return sid
