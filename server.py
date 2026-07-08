"""
Experiment API — FastAPI server.

Ingests IoT walking-trial data (motion CSV + audio WAV + video H.264) from a
Raspberry Pi, runs the feature-extraction pipeline ONCE on upload, stores the
merged feature dict in SQLite, and serves it to the frontend team.

Endpoint shape follows the professor's reference spec:
    https://github.com/davidlinner/experiment-api  (openapi.yaml)

Run:
    uvicorn server:app --reload

Interactive docs (for the frontend team): http://localhost:8000/docs
"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import database as db

# The three existing feature-extraction functions. They live in the repo root
# as `<stream>_features.py` modules. (Currently PLACEHOLDER stand-ins — replace
# those three files with the real pipeline; the imports below stay the same.)
from step_features import extract_step_features
from audio_features import extract_audio_features
from video_features import extract_video_features

DATA_DIR = os.environ.get("EXPERIMENT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)
    db.init_db()
    yield


app = FastAPI(
    title="Experiment API (Parkinson's Gait)",
    description=(
        "Ingest and serve processed IoT walking-trial data. Recordings are "
        "processed once on /recording/stop and stored; GET endpoints never "
        "reprocess."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the frontend to call from a browser (course project: allow all).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# Request models
# ==========================================================================

class ExperimentInput(BaseModel):
    patientNumber: Optional[str] = None
    height: Optional[float] = Field(default=None, description="cm")
    age: Optional[int] = Field(default=None, description="years")
    weight: Optional[float] = Field(default=None, description="kg")
    properties: dict = Field(default_factory=dict)


class ExerciseInput(BaseModel):
    properties: dict = Field(default_factory=dict)


def _new_id() -> str:
    return uuid.uuid4().hex


# ==========================================================================
# Experiments
# ==========================================================================

@app.post("/experiments", status_code=201, tags=["experiments"])
def create_experiment(body: ExperimentInput) -> dict:
    exp_id = _new_id()
    created = db.now_iso()
    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO experiments
               (id, patient_number, height, age, weight, properties, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (exp_id, body.patientNumber, body.height, body.age, body.weight,
             json.dumps(body.properties), created),
        )
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
    return db.experiment_to_dict(row)


@app.get("/experiments", tags=["experiments"])
def list_experiments(
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
) -> dict:
    offset = (page - 1) * pageSize
    with db.get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM experiments").fetchone()["c"]
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (pageSize, offset),
        ).fetchall()
    return {
        "items": [db.experiment_to_dict(r) for r in rows],
        "page": page,
        "pageSize": pageSize,
        "total": total,
    }


@app.get("/experiments/{experiment_id}", tags=["experiments"])
def get_experiment(experiment_id: str) -> dict:
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return db.experiment_to_dict(row)


@app.delete("/experiments/{experiment_id}", status_code=204, tags=["experiments"])
def delete_experiment(experiment_id: str) -> None:
    with db.get_conn() as conn:
        cur = conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Experiment not found")
    # 204: no body


# ==========================================================================
# Exercises
# ==========================================================================

@app.post("/experiments/{experiment_id}/exercises", status_code=201, tags=["exercises"])
def create_exercise(experiment_id: str, body: ExerciseInput) -> dict:
    with db.get_conn() as conn:
        exp = conn.execute("SELECT 1 FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        ex_id = _new_id()
        conn.execute(
            """INSERT INTO exercises
               (id, experiment_id, recording_status, properties, created_at)
               VALUES (?, ?, 'idle', ?, ?)""",
            (ex_id, experiment_id, json.dumps(body.properties), db.now_iso()),
        )
        row = conn.execute("SELECT * FROM exercises WHERE id = ?", (ex_id,)).fetchone()
        has_data = db.exercise_has_data(conn, ex_id)
    return db.exercise_to_dict(row, has_data)


@app.get("/experiments/{experiment_id}/exercises", tags=["exercises"])
def list_exercises_for_experiment(experiment_id: str) -> list:
    with db.get_conn() as conn:
        exp = conn.execute("SELECT 1 FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        rows = conn.execute(
            "SELECT * FROM exercises WHERE experiment_id = ? ORDER BY created_at DESC",
            (experiment_id,),
        ).fetchall()
        return [db.exercise_to_dict(r, db.exercise_has_data(conn, r["id"])) for r in rows]


@app.get("/exercises/{exercise_id}", tags=["exercises"])
def get_exercise(exercise_id: str) -> dict:
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Exercise not found")
        has_data = db.exercise_has_data(conn, exercise_id)
    return db.exercise_to_dict(row, has_data)


# ==========================================================================
# Recording control
# ==========================================================================

@app.post("/exercises/{exercise_id}/recording/start", tags=["recording"])
def start_recording(exercise_id: str) -> dict:
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Exercise not found")
        if db.exercise_has_data(conn, exercise_id):
            raise HTTPException(status_code=409, detail="Exercise already has data")
        if row["recording_status"] == "recording":
            raise HTTPException(status_code=409, detail="Recording already in progress")
        conn.execute(
            "UPDATE exercises SET recording_status = 'recording', recording_started_at = ? WHERE id = ?",
            (db.now_iso(), exercise_id),
        )
        row = conn.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,)).fetchone()
    return db.exercise_to_dict(row, False)


@app.post("/exercises/{exercise_id}/recording/stop", tags=["recording"])
async def stop_recording(
    exercise_id: str,
    motion: UploadFile = File(..., description="motion CSV (accel + gyro, 50 Hz)"),
    audio: UploadFile = File(..., description="audio WAV (48kHz mono, 32-bit)"),
    video: UploadFile = File(..., description="video H.264 (1280x720)"),
) -> dict:
    """
    THE KEY ENDPOINT. Accepts a multipart upload of the three synchronized
    streams, saves them to the data folder, runs all three extraction
    functions ONCE, merges their dicts, and stores the result in the DB.

    Processing is fault-tolerant: if one stream fails, whatever succeeded is
    still stored, along with a per-stream error note under `errors`.
    """
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Exercise not found")

    started_at = row["recording_started_at"]
    ended_at = db.now_iso()

    # --- 1. Persist uploaded files -----------------------------------------
    ex_dir = os.path.join(DATA_DIR, exercise_id)
    os.makedirs(ex_dir, exist_ok=True)
    saved = {}
    for label, upload, default_name in (
        ("motion", motion, "motion.csv"),
        ("audio", audio, "audio.wav"),
        ("video", video, "video.h264"),
    ):
        dest = os.path.join(ex_dir, upload.filename or default_name)
        with open(dest, "wb") as fh:
            fh.write(await upload.read())
        saved[label] = dest

    # --- 2. Run the three extractors (fault tolerant) ----------------------
    features: dict = {}
    errors: dict = {}

    extractors = (
        ("motion", extract_step_features, saved["motion"]),
        ("audio", extract_audio_features, saved["audio"]),
        ("video", extract_video_features, saved["video"]),
    )
    for label, fn, path in extractors:
        try:
            result = fn(path)
            # Namespace each stream's features so keys never collide on merge.
            features[label] = result
        except Exception as exc:  # noqa: BLE001 — store the note, keep going
            errors[label] = f"{type(exc).__name__}: {exc}"

    # --- 3. Store processed results (ONCE) ---------------------------------
    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO exercise_data
                   (exercise_id, started_at, ended_at, features, files, errors, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(exercise_id) DO UPDATE SET
                   started_at = excluded.started_at,
                   ended_at   = excluded.ended_at,
                   features   = excluded.features,
                   files      = excluded.files,
                   errors     = excluded.errors,
                   created_at = excluded.created_at""",
            (exercise_id, started_at, ended_at,
             json.dumps(features), json.dumps(saved), json.dumps(errors), db.now_iso()),
        )
        conn.execute(
            "UPDATE exercises SET recording_status = 'stopped', recording_ended_at = ? WHERE id = ?",
            (ended_at, exercise_id),
        )
        row = conn.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,)).fetchone()
    return db.exercise_to_dict(row, has_data=True)


# ==========================================================================
# Data retrieval  (reads stored JSON — NEVER reprocesses)
# ==========================================================================

@app.get("/exercises/{exercise_id}/data", tags=["data"])
def get_exercise_data(exercise_id: str) -> dict:
    with db.get_conn() as conn:
        ex = conn.execute("SELECT 1 FROM exercises WHERE id = ?", (exercise_id,)).fetchone()
        if ex is None:
            raise HTTPException(status_code=404, detail="Exercise not found")
        row = conn.execute("SELECT * FROM exercise_data WHERE exercise_id = ?", (exercise_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No data recorded for this exercise")
    return db.exercise_data_to_dict(row)


@app.delete("/exercises/{exercise_id}/data", status_code=204, tags=["data"])
def delete_exercise_data(exercise_id: str) -> None:
    with db.get_conn() as conn:
        cur = conn.execute("DELETE FROM exercise_data WHERE exercise_id = ?", (exercise_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="No data recorded for this exercise")
        conn.execute(
            "UPDATE exercises SET recording_status = 'idle', recording_ended_at = NULL WHERE id = ?",
            (exercise_id,),
        )


@app.get("/exercises/{exercise_id}/export", tags=["data"])
def export_exercise_data(exercise_id: str) -> StreamingResponse:
    """Flatten the stored feature dict to a two-column (feature,value) CSV."""
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM exercise_data WHERE exercise_id = ?", (exercise_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No data recorded for this exercise")

    features = json.loads(row["features"]) if row["features"] else {}

    def _flatten(d: dict, prefix: str = "") -> dict:
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                flat.update(_flatten(v, prefix=f"{key}."))
            else:
                flat[key] = v
        return flat

    flat = _flatten(features)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["feature", "value"])
    for k, v in flat.items():
        writer.writerow([k, v])
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="exercise_{exercise_id}.csv"'},
    )


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "Experiment API", "docs": "/docs"}
