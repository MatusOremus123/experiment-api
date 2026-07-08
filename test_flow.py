"""
End-to-end smoke test for the Experiment API.

Uses FastAPI's in-process TestClient (no running uvicorn needed). It drives the
full flow against a REAL sample walk from the srh-ss26-iot-project repo:

  1. create experiment -> exercise
  2. start recording
  3. POST the real motion/audio/video files to /recording/stop (pipeline runs ONCE)
  4. GET /data and confirm the stored, merged features come back
  5. GET /export and print the CSV

It finds a sample walk automatically from SRH_PROJECT_PATH (or the sibling
srh-ss26-iot-project1 folder). Any stream whose dependency isn't installed on
this machine (e.g. mediapipe) is reported under `errors` rather than failing the
whole run — so the test asserts on the streams that actually produced features.

Run:  py test_flow.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import pipeline
import server


def find_sample_triple() -> dict[str, Path]:
    """Locate one motion+audio+video triple from the srh sample data."""
    root = pipeline.SRH_PROJECT_PATH / "collected_sample_data"
    if not root.exists():
        raise SystemExit(
            f"No sample data at {root}. Set SRH_PROJECT_PATH to your srh-ss26-iot-project clone."
        )
    for motion in sorted(root.rglob("motion_*.csv")):
        stamp = motion.stem.split("motion_", 1)[1]
        audio = motion.with_name(f"audio_{stamp}.wav")
        video = motion.with_name(f"video_{stamp}.h264")
        if audio.exists() and video.exists():
            return {"motion": motion, "audio": audio, "video": video}
    raise SystemExit(f"No complete motion/audio/video triple found under {root}")


def main() -> None:
    triple = find_sample_triple()
    print("using sample walk:", triple["motion"].parent.name, "/", triple["motion"].stem.split("motion_")[1])

    # `with` triggers the app's lifespan (which creates the DB tables).
    with TestClient(server.app) as client:
        # 1. create experiment + exercise
        r = client.post("/experiments", json={"patientNumber": "P-001", "age": 67, "height": 172, "weight": 80})
        r.raise_for_status()
        exp = r.json()
        print("experiment:", exp["id"])

        r = client.post(
            f"/experiments/{exp['id']}/exercises",
            json={"properties": {"condition": "normal", "repetition": "1"}},
        )
        r.raise_for_status()
        ex = r.json()
        print("exercise:  ", ex["id"], "status=", ex["recordingStatus"], "hasData=", ex["hasData"])

        # 2. start recording
        r = client.post(f"/exercises/{ex['id']}/recording/start")
        r.raise_for_status()
        print("started:   ", r.json()["recordingStatus"])

        # 3. stop + upload the three REAL streams (pipeline runs ONCE here)
        files = {
            "motion": (triple["motion"].name, triple["motion"].read_bytes(), "text/csv"),
            "audio": (triple["audio"].name, triple["audio"].read_bytes(), "audio/wav"),
            "video": (triple["video"].name, triple["video"].read_bytes(), "application/octet-stream"),
        }
        r = client.post(f"/exercises/{ex['id']}/recording/stop", files=files)
        r.raise_for_status()
        print("stopped:   ", r.json()["recordingStatus"], "hasData=", r.json()["hasData"])

        # 4. GET stored data (no reprocessing) — openapi ExerciseData shape (hybrid)
        r = client.get(f"/exercises/{ex['id']}/data")
        r.raise_for_status()
        data = r.json()
        print("\n--- /data (truncated mouthOpening.values) ---")
        preview = dict(data)
        if preview.get("mouthOpening", {}).get("values"):
            n = len(preview["mouthOpening"]["values"])
            preview["mouthOpening"] = {**preview["mouthOpening"],
                                       "values": preview["mouthOpening"]["values"][:3] + [f"... ({n} frames)"]}
        print(json.dumps(preview, indent=2))

        produced = set(data["features"])
        failed = set(data["errors"])
        print("\nstreams with features:", sorted(produced))
        if failed:
            print("streams with errors:  ", sorted(failed), "(likely a missing dependency on this machine)")

        # openapi ExerciseData shape present
        for key in ("mouthOpening", "soundPressure", "footSpeed", "aggregates"):
            assert key in data, f"ExerciseData missing '{key}'"
        # At least the motion pipeline (numpy/pandas/scipy only) must succeed.
        assert "motion" in produced, f"motion features missing; errors={data['errors']}"
        assert "step_count" in data["features"]["motion"], "motion feature keys look wrong"
        # Fixed-distance derivations populated the aggregates.
        assert data["aggregates"]["averages"]["stepLength"] is not None, "stepLength aggregate missing"
        assert data["aggregates"]["averages"]["footSpeed"] is not None, "footSpeed (walking speed) aggregate missing"
        print("aggregates.averages:", json.dumps(data["aggregates"]["averages"]))

        # 5. CSV export — per experiment (one row per exercise)
        r = client.get(f"/experiments/{exp['id']}/export")
        r.raise_for_status()
        print("\n--- /experiments/{id}/export (CSV) ---")
        print(r.text)

    print("\nOK — end-to-end flow passed.")


if __name__ == "__main__":
    main()
