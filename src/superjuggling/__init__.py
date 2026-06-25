"""Superjuggling — juggling ball tracker.

Public API. The tracking core (models / config / report) imports with no heavy
dependencies; the video pipeline stages require the optional ``cv`` extra
(``uv sync --extra cv``).
"""

from .config import Config
from .models import (
    BallCandidate,
    CandidateFrame,
    TrackingSummary,
    Trajectory,
    VideoMeta,
)

__all__ = [
    "BallCandidate",
    "CandidateFrame",
    "Config",
    "TrackingSummary",
    "Trajectory",
    "VideoMeta",
]
