"""Core data models shared across pipeline stages.

These are deliberately CV-free: trajectories and event timelines are plain
dataclasses / NumPy arrays so the event-extraction and metrics stages can be
exercised against synthetic fixtures without any video dependency (see the
tech spec, §4.5 / §10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Hand(str, Enum):
    """Which hand an event is attributed to."""

    LEFT = "L"
    RIGHT = "R"
    UNKNOWN = "?"


@dataclass(frozen=True)
class VideoMeta:
    """Resolution / fps / duration of the ingested clip (tech spec §4.1)."""

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
class FlightSegment:
    """A short fitted prop flight segment.

    Coefficients are expressed in centred time ``u = t - t_ref``:

    - x(u) = coeff_x[0] * u + coeff_x[1]
    - y(u) = coeff_y[0] * u² + coeff_y[1] * u + coeff_y[2]
    """

    segment_id: int
    track_id: int | None
    t: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    confidence: NDArray[np.float64] | None = None
    t_ref: float = 0.0
    coeff_x: tuple[float, float] | None = None
    coeff_y: tuple[float, float, float] | None = None
    rms_error_px: float | None = None


@dataclass
class Trajectory:
    """A single tracked prop's centre position over time.

    Arrays are parallel and indexed by sample; ``t`` is in **seconds**.
    Image ``y`` increases downward, so a throw apex is a local *minimum* of
    ``y`` (tech spec §4.4).
    """

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


@dataclass
class HandTrack:
    """Per-frame wrist positions for one hand (from pose keypoints)."""

    hand: Hand
    t: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]


@dataclass(frozen=True)
class ThrowEvent:
    """A throw, located at the trajectory apex (tech spec §4.4)."""

    t: float
    track_id: int
    hand: Hand
    apex_x: float
    apex_y: float
    height_px: float  # apex height relative to the hand line (positive = higher)


@dataclass(frozen=True)
class CatchEvent:
    t: float
    track_id: int
    hand: Hand


@dataclass(frozen=True)
class DropEvent:
    t: float
    track_id: int


@dataclass
class Timelines:
    """The event timelines that feed the metrics engine (tech spec §5)."""

    throws: list[ThrowEvent] = field(default_factory=list)
    catches: list[CatchEvent] = field(default_factory=list)
    drops: list[DropEvent] = field(default_factory=list)

    def throws_for(self, hand: Hand) -> list[ThrowEvent]:
        return [e for e in self.throws if e.hand == hand]
