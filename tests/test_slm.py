import json
import pytest

from slm_service.prompts import SYSTEM_PROMPT, build_user_prompt, format_transcript
from common.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    MeetingContext,
    RiskItem,
    TranscriptSegment,
    SegmentStatus,
)


class TestPrompts:
    def test_format_transcript(self):
        segments = [
            {"start_time": 0.0, "speaker": "SPEAKER_00", "text": "Hello"},
            {"start_time": 2.5, "speaker": "SPEAKER_01", "text": "Hi there"},
        ]
        result = format_transcript(segments)
        assert "SPEAKER_00: Hello" in result
        assert "[2.5s] SPEAKER_01: Hi there" in result

    def test_format_transcript_unknown_speaker(self):
        segments = [{"start_time": 0.0, "text": "No speaker"}]
        result = format_transcript(segments)
        assert "UNKNOWN: No speaker" in result

    def test_build_user_prompt_includes_context(self):
        prompt = build_user_prompt("transcript text", meeting_type="standup", department="Engineering")
        assert "standup" in prompt
        assert "Engineering" in prompt
        assert "transcript text" in prompt

    def test_system_prompt_mentions_json(self):
        assert "JSON" in SYSTEM_PROMPT


class TestSchemas:
    def test_analyze_request_roundtrip(self):
        req = AnalyzeRequest(
            stream_id="s1",
            segments=[
                TranscriptSegment(
                    status=SegmentStatus.final,
                    segment_id=0,
                    start_time=0.0,
                    end_time=1.0,
                    text="Hello",
                    speaker="SPEAKER_00",
                )
            ],
            context=MeetingContext(meeting_type="standup"),
        )
        data = req.model_dump()
        assert data["stream_id"] == "s1"
        assert len(data["segments"]) == 1

    def test_analyze_response_roundtrip(self):
        resp = AnalyzeResponse(
            stream_id="s1",
            summary="A meeting.",
            key_points=["Point 1"],
            action_items=["Do something"],
            risks=[RiskItem(category="PoSH", description="Concern", severity="medium")],
        )
        data = json.loads(resp.model_dump_json())
        assert data["risks"][0]["severity"] == "medium"
