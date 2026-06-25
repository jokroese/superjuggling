"""Core ball-tracking data models.

These are deliberately CV-free: detector outputs are normalised into centre
candidates, linkers turn candidates into trajectories, and reports/overlays
inspect those trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VideoMeta:
    """Resolution / fps / duration of the ingested clip."""

    path: str
    fps: float
    total_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0


@dataclass(frozen=True)
class BallCandidate:
    """A centre-point candidate for a juggling prop in one frame.

    Boxes are detector artefacts; downstream linking and event extraction care
    about prop centres. This is the stable boundary between detection and
    tracking/linking.
    """

    frame_index: int
    t: float
    x: float
    y: float
    score: float
    radius_px: float | None = None
    source: str = "detector"
    class_id: int | None = None


@dataclass(frozen=True)
class CandidateFrame:
    """All centre candidates for a single video frame."""

    frame_index: int
    t: float
    candidates: list[BallCandidate] = field(default_factory=list)


@dataclass
class Trajectory:
    """A single tracked ball's centre position over time."""

    track_id: int
    t: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    confidence: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        n = len(self.t)
        if not (len(self.x) == len(self.y) == n):
            msg = "Trajectory t/x/y arrays must be the same length"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrackingSummary:
    """Run-level diagnostics for ball tracking quality."""

    frames: int
    candidates: int
    trajectories: int
    trajectory_points: int
    estimated_props: int
    backend: str
    mean_track_points: float
    median_track_points: float
