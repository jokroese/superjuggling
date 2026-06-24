"""Superjuggling — juggling consistency tracker.

Public API. The analytical core (events / metrics / report / models / config)
imports with no heavy dependencies; the video pipeline stages require the
optional ``cv`` extra (``uv sync --extra cv``).
"""

from .config import Config
from .metrics import MetricsReport, compute_metrics
from .models import (
    CatchEvent,
    DropEvent,
    Hand,
    HandTrack,
    ThrowEvent,
    Timelines,
    Trajectory,
    VideoMeta,
)

__all__ = [
    "CatchEvent",
    "Config",
    "DropEvent",
    "Hand",
    "HandTrack",
    "MetricsReport",
    "ThrowEvent",
    "Timelines",
    "Trajectory",
    "VideoMeta",
    "compute_metrics",
]
