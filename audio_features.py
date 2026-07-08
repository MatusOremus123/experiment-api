"""
Audio feature extraction.

======================================================================
!!! PLACEHOLDER MODULE — REPLACE WITH YOUR REAL IMPLEMENTATION !!!
======================================================================
Real signature (per the build brief):
    extract_audio_features(audio_wav_path) -> dict
    (loudness, sound pressure dB, "Ba" events)

This stand-in uses only the Python stdlib `wave` module so the server
can run and be tested without librosa installed. Replace this file with
the real librosa-based implementation, keeping the function name and the
`(path) -> dict` signature.

Expected input: 48 kHz mono, 32-bit WAV.
"""

from __future__ import annotations

import wave

import numpy as np


def extract_audio_features(audio_wav_path: str) -> dict:
    with wave.open(audio_wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if n_channels > 1 and samples.size:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    max_amp = float(np.iinfo(dtype).max) if samples.size else 1.0
    norm = samples / max_amp if max_amp else samples
    duration_s = n_frames / sample_rate if sample_rate else 0.0

    rms = float(np.sqrt(np.mean(norm ** 2))) if norm.size else 0.0
    # dBFS relative to full scale (stand-in for sound-pressure dB)
    sound_pressure_db = 20.0 * np.log10(rms) if rms > 0 else -120.0

    return {
        "sample_rate_hz": sample_rate,
        "channels": n_channels,
        "duration_s": round(duration_s, 3),
        "rms_amplitude": round(rms, 6),
        "sound_pressure_db": round(float(sound_pressure_db), 2),
        "ba_event_count": 0,  # real pipeline detects "Ba" syllable events
        "_note": "PLACEHOLDER audio features — replace audio_features.py with the real pipeline",
    }
