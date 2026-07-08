"""
Bridge to the srh-ss26-iot-project feature-extraction pipeline, plus assembly
of the openapi-shaped ExerciseData payload.

The extraction code (extract_step_features / extract_audio_features /
extract_video_features) lives in its OWN repo, not here. This API imports those
functions from that repo by adding it to sys.path — it does NOT copy them.
Importing in place also lets extract_video_features find its
models/face_landmarker.task (path relative to that file).

Point SRH_PROJECT_PATH at your clone of srh-ss26-iot-project; it defaults to a
sibling folder named `srh-ss26-iot-project1`.

--------------------------------------------------------------------------
Data-shape note (hybrid, per team decision):
The srh extractors return ~14 SCALAR features. The professor's openapi.yaml
ExerciseData wants raw per-sample SIGNALS (mouthOpening / soundPressure /
footSpeed) plus aggregates. We fill what we actually have:
  * aggregates      — from the 14 scalars + fixed-distance derivations
  * mouthOpening    — real per-frame vertical series (horizontal not produced
                      yet -> null); decoded ONCE via the video module's own
                      mouth_opening_series helper
  * soundPressure   — EMPTY for now: our audio loudness is dBFS (relative), not
                      calibrated SPL. Calibrated SPL is expected from the Pi's
                      spl.csv, which the current 3-file upload does not include.
  * footSpeed       — EMPTY for now: per-sample foot speed is not derived by the
                      current motion pipeline.
The raw 14-feature dict is also returned verbatim under `features`.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np

# Fixed indoor straight route (Session 5 architecture: "14 m", exact distance).
# Walking speed and step length are derived from it.
ROUTE_DISTANCE_M = float(os.environ.get("ROUTE_DISTANCE_M", "14.0"))

_DEFAULT_SRH = Path(__file__).resolve().parent.parent / "srh-ss26-iot-project1"
SRH_PROJECT_PATH = Path(os.environ.get("SRH_PROJECT_PATH", _DEFAULT_SRH))

# Make the srh repo's root importable (its extract_* modules are top-level scripts).
if str(SRH_PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(SRH_PROJECT_PATH))


def _err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _num(x) -> float | None:
    """JSON-safe number: turn NaN/inf into None."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


# --------------------------------------------------------------------------
# Per-stream extraction (fault tolerant: a failing stream is recorded, others run)
# --------------------------------------------------------------------------

def _run_motion(path: str, features: dict, errors: dict) -> None:
    try:
        mod = importlib.import_module("extract_step_features")
        feats = mod.extract_step_features(path)
        # Fixed-distance derivations (only these two need the 14 m constant).
        dur = feats.get("duration_s")
        steps = feats.get("step_count")
        if dur and dur > 0:
            feats["walking_speed_cms"] = ROUTE_DISTANCE_M * 100.0 / dur
        if steps:
            feats["step_length_cm"] = ROUTE_DISTANCE_M * 100.0 / steps
        features["motion"] = feats
    except Exception as exc:  # noqa: BLE001
        errors["motion"] = _err(exc)


def _run_audio(path: str, features: dict, errors: dict) -> None:
    try:
        mod = importlib.import_module("extract_audio_features")
        features["audio"] = mod.extract_audio_features(path)
    except Exception as exc:  # noqa: BLE001
        errors["audio"] = _err(exc)


def _run_video(path: str, features: dict, series: dict, errors: dict) -> None:
    """Decode the video ONCE: derive both the scalar features and the raw
    vertical mouth-opening series from a single mouth_opening_series() pass."""
    try:
        mod = importlib.import_module("extract_video_features")
        fps = float(getattr(mod, "DEFAULT_FPS", 30.0))
        opening = mod.mouth_opening_series(path)  # per-frame vertical opening (may hold NaN)
        features["video"] = {
            "n_frames": int(len(opening)),
            "n_face_detected": int(np.sum(~np.isnan(opening))),
            "fps_assumed": fps,
            "mean_mouth_opening": _num(mod.mean_mouth_opening(opening)),
            "mouth_opening_rate": _num(mod.opening_rate(opening, fps)),
            "opening_variability": _num(mod.opening_variability(opening)),
            "opening_trend": _num(mod.opening_trend(opening, fps)),
        }
        # ExerciseData.mouthOpening: [[vertical, horizontal], ...]; horizontal not
        # produced by the current pipeline -> null, flagged in the payload notes.
        series["mouthOpening"] = {
            "values": [[_num(v), None] for v in opening],
            "sampleRate": fps,
        }
    except Exception as exc:  # noqa: BLE001
        errors["video"] = _err(exc)
        # Best-effort fallback to the top-level scalar function (no series).
        try:
            mod = importlib.import_module("extract_video_features")
            features["video"] = mod.extract_video_features(path)
            errors.pop("video", None)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Assembly into the openapi ExerciseData shape
# --------------------------------------------------------------------------

def _build_aggregates(features: dict, mouth_series: dict | None) -> dict:
    motion = features.get("motion", {})
    audio = features.get("audio", {})
    video = features.get("video", {})

    step_length_cm = _num(motion.get("step_length_cm"))
    walking_speed_cms = _num(motion.get("walking_speed_cms"))
    mouth_v = _num(video.get("mean_mouth_opening"))
    sound = _num(audio.get("mean_loudness"))  # NOTE: dBFS (relative), not calibrated SPL

    # Median of the mouth-opening series where we have it (single-trial medians).
    mouth_v_median = None
    if mouth_series and mouth_series.get("values"):
        verts = [row[0] for row in mouth_series["values"] if row[0] is not None]
        if verts:
            mouth_v_median = float(np.median(verts))

    return {
        "stepLengths": {
            # Fixed distance / step_count gives a single average step length.
            "values": [step_length_cm] if step_length_cm is not None else [],
            "unit": "cm",
        },
        "averages": {
            "mouthOpeningVertical": mouth_v,
            "mouthOpeningHorizontal": None,   # not produced yet
            "soundPressure": sound,           # dBFS (relative) — see notes
            "footSpeed": walking_speed_cms,   # avg foot speed ~= walking speed
            "stepLength": step_length_cm,
        },
        "medians": {
            "mouthOpeningVertical": mouth_v_median,
            "mouthOpeningHorizontal": None,
            "soundPressure": None,            # no SPL series to take a median of yet
            "footSpeed": None,
            "stepLength": step_length_cm,     # single value
        },
    }


PENDING_NOTES = {
    "soundPressure": "empty: our audio loudness is dBFS (relative), not calibrated "
                     "SPL (Pa/dB). Calibrated SPL is expected from the Pi's spl.csv.",
    "mouthOpeningHorizontal": "null: only vertical mouth opening is produced today.",
    "footSpeed": "empty: per-sample foot speed is not derived by the current motion "
                 "pipeline; aggregate footSpeed uses walking speed (14 m / duration).",
    "aggregates.averages.soundPressure": "value is mean loudness in dBFS, not calibrated SPL.",
    "distance": f"walking speed and step length assume a fixed {ROUTE_DISTANCE_M:g} m route.",
}


def process_recording(paths: dict[str, str]) -> dict:
    """
    Run the pipeline ONCE on the three uploaded files and assemble the stored
    payload. Returns a dict ready to persist and (with ids/timestamps added) to
    serve directly from GET /exercises/{id}/data — no reprocessing on GET.
    """
    features: dict = {}
    series: dict = {}
    errors: dict = {}

    _run_motion(paths.get("motion"), features, errors)
    _run_audio(paths.get("audio"), features, errors)
    _run_video(paths.get("video"), features, series, errors)

    audio = features.get("audio", {})
    mouth = series.get("mouthOpening")

    return {
        # --- openapi ExerciseData signal blocks ---
        "mouthOpening": mouth or {"values": [], "sampleRate": None},
        "soundPressure": {
            "values": [],
            "unit": "dB",
            "sampleRate": _num(audio.get("sample_rate")),
        },
        "footSpeed": {"values": [], "unit": "cm/s", "sampleRate": None},
        "aggregates": _build_aggregates(features, mouth),
        # --- our raw scalar features, verbatim (hybrid extra) ---
        "features": features,
        "errors": errors,
        "_notes": PENDING_NOTES,
    }
