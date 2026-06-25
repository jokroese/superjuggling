"""Superjuggling — juggling ball tracker.

Public API. The tracking core (models / config / report) imports with no heavy
dependencies; the video pipeline stages require the optional ``cv`` extra
(``uv sync --extra cv``).
"""

from .config import Config
from .evaluation import CandidateEvaluationSummary
from .labels import GroundTruthLabel
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
    "CandidateEvaluationSummary",
    "GroundTruthLabel",
    "TrackingSummary",
    "Trajectory",
    "VideoMeta",
]
