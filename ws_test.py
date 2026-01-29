import asyncio
import json
import numpy as np
import websockets


async def test():
    uri = "ws://localhost:8080/audio"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "stream_id": "live-test",
            "sample_rate": 16000,
            "encoding": "pcm_s16le",
            "channels": 1,
        }))
        print("Sent start")

        t = np.linspace(0, 3, 16000 * 3, dtype=np.float32)
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        data = tone.tobytes()
        chunk_size = 16000 * 2

        for i in range(0, len(data), chunk_size):
            await ws.send(data[i:i + chunk_size])
            print("Sent chunk", i // chunk_size + 1)

        await ws.send(json.dumps({"type": "end", "stream_id": "live-test"}))
        print("Sent end, waiting...")

        async for msg in ws:
            resp = json.loads(msg)
            print(json.dumps(resp, indent=2))
            if resp.get("type") == "transcript_complete":
                break

    print("\nDone.")


asyncio.run(test())
