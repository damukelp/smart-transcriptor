import pytest
from gateway.audio_utils import normalize_audio, _ffmpeg_format
from gateway.session import SessionManager


class TestAudioUtils:
    def test_passthrough_when_already_correct_format(self):
        pcm = b"\x00\x01" * 100
        result = normalize_audio(pcm, input_sample_rate=16000, input_channels=1, input_encoding="pcm_s16le")
        assert result == pcm

    def test_ffmpeg_format_mapping(self):
        assert _ffmpeg_format("pcm_s16le") == "s16le"
        assert _ffmpeg_format("wav") == "wav"
        assert _ffmpeg_format("unknown") == "s16le"


class TestSessionManager:
    @pytest.fixture
    def manager(self):
        return SessionManager(max_sessions=2)

    @pytest.mark.asyncio
    async def test_create_and_remove(self, manager):
        # Use a mock websocket
        session = await manager.create("s1", client_ws=None)
        assert session.stream_id == "s1"
        assert manager.active_count == 1
        await manager.remove("s1")
        assert manager.active_count == 0

    @pytest.mark.asyncio
    async def test_max_sessions_enforced(self, manager):
        await manager.create("s1", client_ws=None)
        await manager.create("s2", client_ws=None)
        with pytest.raises(RuntimeError, match="Max sessions"):
            await manager.create("s3", client_ws=None)

    @pytest.mark.asyncio
    async def test_duplicate_stream_id_rejected(self, manager):
        await manager.create("s1", client_ws=None)
        with pytest.raises(RuntimeError, match="already exists"):
            await manager.create("s1", client_ws=None)
