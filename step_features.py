"""
Motion / gait feature extraction.

======================================================================
!!! PLACEHOLDER MODULE — REPLACE WITH YOUR REAL IMPLEMENTATION !!!
======================================================================
The build brief said `extract_step_features(motion_csv_path) -> dict`
already exists in the repo root. This repository was empty when the
server was scaffolded, so this file provides a *lightweight, dependency-free
stand-in* that reads the motion CSV and returns a plausible gait-feature
dict. It exists only so the API + DB + pipeline wiring can be tested
end to end.

To integrate the real pipeline: drop your real `extract_step_features`
here (keep the function name and the `(path) -> dict` signature) and the
server will pick it up with no other changes.

Expected input CSV columns:
    time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z   (50 Hz)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_step_features(motion_csv_path: str) -> dict:
    df = pd.read_csv(motion_csv_path)

    # Basic sanity / timing
    n = len(df)
    if "time" in df.columns and n > 1:
        duration_s = float(df["time"].iloc[-1] - df["time"].iloc[0])
    else:
        duration_s = n / 50.0  # assume 50 Hz
    sample_rate = (n - 1) / duration_s if duration_s > 0 else 50.0

    accel_cols = [c for c in ("accel_x", "accel_y", "accel_z") if c in df.columns]
    gyro_cols = [c for c in ("gyro_x", "gyro_y", "gyro_z") if c in df.columns]

    accel_mag = (
        np.sqrt((df[accel_cols] ** 2).sum(axis=1)).to_numpy()
        if accel_cols
        else np.zeros(n)
    )
    gyro_mag = (
        np.sqrt((df[gyro_cols] ** 2).sum(axis=1)).to_numpy()
        if gyro_cols
        else np.zeros(n)
    )

    # Very rough peak-count "step" estimate on the accel magnitude signal.
    mean_a = float(accel_mag.mean()) if accel_mag.size else 0.0
    std_a = float(accel_mag.std()) if accel_mag.size else 0.0
    threshold = mean_a + 0.5 * std_a
    above = accel_mag > threshold
    # count rising edges
    steps = int(np.sum((~above[:-1]) & (above[1:]))) if accel_mag.size > 1 else 0
    cadence_spm = (steps / duration_s * 60.0) if duration_s > 0 else 0.0

    return {
        "sample_count": n,
        "duration_s": round(duration_s, 3),
        "sample_rate_hz": round(float(sample_rate), 2),
        "estimated_steps": steps,
        "cadence_spm": round(cadence_spm, 2),
        "step_regularity": round(1.0 - (std_a / mean_a) if mean_a else 0.0, 4),
        "mean_accel_magnitude": round(mean_a, 4),
        "mean_rotation": round(float(gyro_mag.mean()) if gyro_mag.size else 0.0, 4),
        "_note": "PLACEHOLDER step features — replace step_features.py with the real pipeline",
    }
