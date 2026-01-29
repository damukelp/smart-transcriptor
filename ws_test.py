import asyncio
import json
import sys
import wave
import numpy as np
import websockets


async def test(wav_path=None):
    uri = "ws://localhost:8080/audio"
    async with websockets.connect(uri, close_timeout=2) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "stream_id": "live-test",
            "sample_rate": 16000,
            "encoding": "pcm_s16le",
            "channels": 1,
        }))
        print("Sent start")

        if wav_path:
            with wave.open(wav_path, "rb") as wf:
                print(f"WAV: {wf.getnchannels()}ch, {wf.getframerate()}Hz, {wf.getnframes()} frames")
                data = wf.readframes(wf.getnframes())
        else:
            t = np.linspace(0, 3, 16000 * 3, dtype=np.float32)
            tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
            data = tone.tobytes()

        chunk_size = 16000 * 2  # 1s of 16-bit mono
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        for i in range(0, len(data), chunk_size):
            await ws.send(data[i:i + chunk_size])
            print(f"Sent chunk {i // chunk_size + 1}/{total_chunks}")

        await ws.send(json.dumps({"type": "end", "stream_id": "live-test"}))
        print("Sent end, waiting...\n")

        async for msg in ws:
            resp = json.loads(msg)
            print(json.dumps(resp, indent=2))
            if resp.get("type") == "transcript_complete":
                break

    print("\nDone.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        asyncio.run(test(path))
    except websockets.exceptions.ConnectionClosedError:
        pass
