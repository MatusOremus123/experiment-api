"""
End-to-end smoke test for the Experiment API.

Uses FastAPI's in-process TestClient (no running uvicorn needed). It:
  1. generates tiny-but-valid sample motion CSV / audio WAV / video H.264 files,
  2. creates an experiment -> exercise,
  3. starts recording,
  4. POSTs the three files to /recording/stop (triggers the pipeline ONCE),
  5. GETs /data and asserts the stored, merged features come back,
  6. GETs /export and checks the CSV.

Run:  py test_flow.py
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
from fastapi.testclient import TestClient

import server


def make_motion_csv() -> bytes:
    rows = ["time,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"]
    for i in range(100):  # 2 s at 50 Hz
        t = i / 50.0
        ax = 1.0 + 0.5 * np.sin(2 * np.pi * 2 * t)  # ~2 Hz "steps"
        rows.append(f"{t:.3f},{ax:.4f},0.1,9.81,{0.2*np.cos(t):.4f},0.0,0.0")
    return ("\n".join(rows) + "\n").encode()


def make_audio_wav() -> bytes:
    buf = io.BytesIO()
    sr = 48000
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit is enough for the placeholder
        wf.setframerate(sr)
        samples = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sr // 10) / sr) * 32767).astype(np.int16)
        wf.writeframes(b"".join(struct.pack("<h", int(s)) for s in samples))
    return buf.getvalue()


def make_video_h264() -> bytes:
    # Not a decodable stream — the placeholder video extractor only stats the file.
    return b"\x00\x00\x00\x01" + b"\x67\x42" + b"\x00" * 4096


def main() -> None:
    # `with` triggers the app's lifespan (which creates the DB tables).
    with TestClient(server.app) as client:
        # 1. create experiment
        r = client.post("/experiments", json={"patientNumber": "P-001", "age": 67, "height": 172, "weight": 80})
        r.raise_for_status()
        exp = r.json()
        print("experiment:", exp["id"])

        # 2. create exercise
        r = client.post(f"/experiments/{exp['id']}/exercises", json={"properties": {"trial": "1"}})
        r.raise_for_status()
        ex = r.json()
        print("exercise:  ", ex["id"], "status=", ex["recordingStatus"], "hasData=", ex["hasData"])

        # 3. start recording
        r = client.post(f"/exercises/{ex['id']}/recording/start")
        r.raise_for_status()
        print("started:   ", r.json()["recordingStatus"])

        # 4. stop + upload the three streams (pipeline runs ONCE here)
        files = {
            "motion": ("motion_123.csv", make_motion_csv(), "text/csv"),
            "audio": ("audio_123.wav", make_audio_wav(), "audio/wav"),
            "video": ("video_123.h264", make_video_h264(), "application/octet-stream"),
        }
        r = client.post(f"/exercises/{ex['id']}/recording/stop", files=files)
        r.raise_for_status()
        print("stopped:   ", r.json()["recordingStatus"], "hasData=", r.json()["hasData"])

        # 5. GET stored data (no reprocessing)
        r = client.get(f"/exercises/{ex['id']}/data")
        r.raise_for_status()
        data = r.json()
        print("\n--- /data ---")
        import json
        print(json.dumps(data, indent=2))

        assert "motion" in data["features"], "motion features missing"
        assert "audio" in data["features"], "audio features missing"
        assert "video" in data["features"], "video features missing"
        assert data["features"]["motion"]["sample_count"] == 100

        # 6. CSV export
        r = client.get(f"/exercises/{ex['id']}/export")
        r.raise_for_status()
        print("\n--- /export (CSV) ---")
        print(r.text)

    print("\nOK — end-to-end flow passed.")


if __name__ == "__main__":
    main()
