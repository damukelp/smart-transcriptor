from __future__ import annotations

import subprocess


def normalize_audio(
    data: bytes,
    input_sample_rate: int = 16000,
    input_channels: int = 1,
    input_encoding: str = "pcm_s16le",
) -> bytes:
    """Convert incoming audio to 16kHz mono 16-bit PCM.

    If the audio is already in the target format, return as-is.
    Otherwise shell out to ffmpeg for conversion.
    """
    if input_sample_rate == 16000 and input_channels == 1 and input_encoding == "pcm_s16le":
        return data

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", _ffmpeg_format(input_encoding),
        "-ar", str(input_sample_rate),
        "-ac", str(input_channels),
        "-i", "pipe:0",
        "-f", "s16le",
        "-ar", "16000",
        "-ac", "1",
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=data, capture_output=True, check=True)
    return result.stdout


def _ffmpeg_format(encoding: str) -> str:
    mapping = {
        "pcm_s16le": "s16le",
        "pcm_f32le": "f32le",
        "wav": "wav",
        "ogg": "ogg",
        "mp3": "mp3",
    }
    return mapping.get(encoding, "s16le")
