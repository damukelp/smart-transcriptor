import numpy as np
import pytest

from asr_service.models import ChunkResult, DiarizedSegment
from asr_service.session import StreamSession
from common.config import ASRSettings


class TestStreamSession:
    @pytest.fixture
    def session(self):
        settings = ASRSettings(chunk_duration_s=0.5, hf_token="")
        return StreamSession(stream_id="test", settings=settings)

    def test_add_audio_and_has_chunk(self, session):
        # 0.5s at 16kHz = 8000 samples, 16-bit = 16000 bytes
        pcm = np.zeros(8000, dtype=np.int16).tobytes()
        session.add_audio(pcm)
        assert session.has_chunk()

    def test_pop_chunk_returns_correct_offset(self, session):
        pcm = np.zeros(8000, dtype=np.int16).tobytes()
        session.add_audio(pcm)
        chunk, offset = session.pop_chunk()
        assert offset == 0.0
        assert len(chunk) == 8000
        assert not session.has_chunk()

    def test_flush_returns_remainder(self, session):
        pcm = np.zeros(4000, dtype=np.int16).tobytes()
        session.add_audio(pcm)
        assert not session.has_chunk()
        result = session.flush()
        assert result is not None
        chunk, offset = result
        assert len(chunk) == 4000

    def test_segment_id_increments(self, session):
        assert session.next_segment_id() == 0
        assert session.next_segment_id() == 1
        assert session.next_segment_id() == 2


class TestModels:
    def test_chunk_result(self):
        r = ChunkResult(text="hello", start_time=0.0, end_time=1.0)
        assert r.text == "hello"

    def test_diarized_segment(self):
        d = DiarizedSegment(text="hi", start_time=0.0, end_time=1.0, speaker="SPEAKER_00")
        assert d.speaker == "SPEAKER_00"
