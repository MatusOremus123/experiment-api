"""
SQLite persistence layer for the Experiment API.

Single-file database (experiment.db). Data model:
    experiments (a patient)  ->  exercises (a walking trial)  ->  exercise_data (processed features)

IMPORTANT design point (required by the brief): recordings are processed
exactly ONCE, on /recording/stop, and the resulting feature dict is stored
in `exercise_data.features` as JSON. GET endpoints only read that stored
JSON — they never reprocess.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

DB_PATH = os.environ.get("EXPERIMENT_DB", os.path.join(os.path.dirname(__file__), "experiment.db"))


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (matches the reference spec's date-time)."""
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id             TEXT PRIMARY KEY,
                patient_number TEXT,
                height         REAL,
                age            INTEGER,
                weight         REAL,
                properties     TEXT NOT NULL DEFAULT '{}',   -- JSON object
                created_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exercises (
                id                   TEXT PRIMARY KEY,
                experiment_id        TEXT NOT NULL,
                recording_status     TEXT NOT NULL DEFAULT 'idle',  -- idle | recording | stopped
                recording_started_at TEXT,
                recording_ended_at   TEXT,
                properties           TEXT NOT NULL DEFAULT '{}',    -- JSON object
                created_at           TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS exercise_data (
                exercise_id  TEXT PRIMARY KEY,
                started_at   TEXT,
                ended_at     TEXT,
                payload      TEXT NOT NULL DEFAULT '{}',   -- assembled ExerciseData (openapi shape + extras), JSON
                files        TEXT NOT NULL DEFAULT '{}',   -- saved upload paths, JSON
                created_at   TEXT NOT NULL,
                FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE
            );
            """
        )


# --------------------------------------------------------------------------
# Row serialization helpers  (SQLite row -> API-shaped dict, camelCase to
# match the professor's openapi.yaml)
# --------------------------------------------------------------------------

def _loads(value: Optional[str]) -> Any:
    return json.loads(value) if value else {}


def experiment_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "patientNumber": row["patient_number"],
        "height": row["height"],
        "age": row["age"],
        "weight": row["weight"],
        "properties": _loads(row["properties"]),
        "createdAt": row["created_at"],
    }


def exercise_to_dict(row: sqlite3.Row, has_data: bool) -> dict:
    return {
        "id": row["id"],
        "experimentId": row["experiment_id"],
        "recordingStatus": row["recording_status"],
        "recordingStartedAt": row["recording_started_at"],
        "recordingEndedAt": row["recording_ended_at"],
        "hasData": has_data,
        "properties": _loads(row["properties"]),
        "createdAt": row["created_at"],
    }


def exercise_data_to_dict(row: sqlite3.Row) -> dict:
    """
    Serve the ExerciseData that was assembled and stored ONCE at recording/stop
    (see pipeline.process_recording). This only reads stored JSON — no reprocessing.

    The payload follows the professor's openapi ExerciseData shape (mouthOpening /
    soundPressure / footSpeed / aggregates) as a hybrid: fields we can produce are
    filled, the rest are empty/null with reasons under `_notes`. Our raw scalar
    features are also included verbatim under `features`. The exact final field
    mapping remains a pending TEAM DECISION.
    """
    payload = _loads(row["payload"])
    return {
        "exerciseId": row["exercise_id"],
        "startedAt": row["started_at"],
        "endedAt": row["ended_at"],
        **payload,  # mouthOpening, soundPressure, footSpeed, aggregates, features, errors, _notes
        "files": _loads(row["files"]),
    }


# --------------------------------------------------------------------------
# Small data-access helpers
# --------------------------------------------------------------------------

def exercise_has_data(conn: sqlite3.Connection, exercise_id: str) -> bool:
    cur = conn.execute("SELECT 1 FROM exercise_data WHERE exercise_id = ?", (exercise_id,))
    return cur.fetchone() is not None
