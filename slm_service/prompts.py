from __future__ import annotations

from typing import Optional

SYSTEM_PROMPT = """\
You are an expert meeting analyst specializing in workplace compliance and HR policy.
Analyze the provided transcript and produce a structured JSON response.

You MUST respond with valid JSON matching this schema:
{
  "summary": "string — concise meeting summary",
  "key_points": ["string — key discussion points"],
  "action_items": ["string — action items with owners if identifiable"],
  "risks": [
    {
      "category": "string — e.g. PoSH, HR Policy, Legal",
      "description": "string — description of the risk",
      "severity": "low | medium | high"
    }
  ]
}

Pay special attention to:
- Prevention of Sexual Harassment (PoSH) compliance issues
- HR policy violations or concerns
- Inappropriate language or behavior
- Power dynamics and potential coercion
- Any legally sensitive statements

If no risks are found, return an empty risks array.
"""


def build_user_prompt(
    formatted_transcript: str,
    meeting_type: str = "general",
    department: Optional[str] = None,
) -> str:
    context_parts = [f"Meeting type: {meeting_type}"]
    if department:
        context_parts.append(f"Department: {department}")

    return f"""\
{chr(10).join(context_parts)}

Transcript:
{formatted_transcript}

Analyze this transcript and respond with the JSON structure specified."""


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or "UNKNOWN"
        text = seg.get("text", "")
        start = seg.get("start_time", 0.0)
        lines.append(f"[{start:.1f}s] {speaker}: {text}")
    return "\n".join(lines)
