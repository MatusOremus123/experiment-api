"""
Video feature extraction (mouth-opening via MediaPipe).

======================================================================
!!! PLACEHOLDER MODULE — REPLACE WITH YOUR REAL IMPLEMENTATION !!!
======================================================================
Real signature (per the build brief):
    extract_video_features(video_h264_path) -> dict
    (mouth-opening features via MediaPipe / OpenCV)

Raw H.264 (.h264) has no reliable container metadata, and MediaPipe /
OpenCV are heavy to install, so this stand-in does NOT decode the video.
It only inspects the file on disk and returns a plausible feature dict so
the end-to-end API + DB flow can be tested. Replace this file with the
real MediaPipe implementation, keeping the function name and the
`(path) -> dict` signature.
"""

from __future__ import annotations

import os


def extract_video_features(video_h264_path: str) -> dict:
    size_bytes = os.path.getsize(video_h264_path) if os.path.exists(video_h264_path) else 0

    return {
        "file_size_bytes": size_bytes,
        "frames_analyzed": 0,          # real pipeline counts decoded frames
        "faces_detected": 0,
        "mouth_opening_vertical_mean": None,    # normalized to frame height
        "mouth_opening_horizontal_mean": None,  # normalized to frame width
        "_note": "PLACEHOLDER video features — replace video_features.py with the real MediaPipe pipeline",
    }
