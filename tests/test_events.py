"""Tests for event extraction against synthetic trajectories (tech spec §4.4)."""

from __future__ import annotations

import numpy as np

from superjuggling.config import EventConfig
from superjuggling.events import extract_throws
from superjuggling.models import Hand, Trajectory
from superjuggling.pipeline import analyze_trajectories, estimate_prop_count
from superjuggling.models import VideoMeta


def _parabolic_arc(
    t0: float, duration: float, apex_y: float, base_y: float, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """A single throw: y dips to a minimum (apex) then returns (image y down)."""
    n = int(duration * fps)
    t = t0 + np.arange(n) / fps
    # Parabola opening downward in 'height', i.e. upward in y at the ends.
    mid = t0 + duration / 2
    norm = (t - mid) / (duration / 2)
    y = apex_y + (base_y - apex_y) * (norm**2)
    return t, y


def test_single_apex_detected() -> None:
    fps = 60.0
    t, y = _parabolic_arc(0.0, 1.0, apex_y=200.0, base_y=600.0, fps=fps)
    x = np.full_like(t, 700.0)
    traj = Trajectory(track_id=1, t=t, x=x, y=y)
    throws = extract_throws(traj, EventConfig(), hand_line_y=600.0, hands=None)
    assert len(throws) == 1
    # Apex near the middle of the arc.
    assert abs(throws[0].t - 0.5) < 0.05
    # Height above the hand line ≈ 600 - 200 = 400.
    assert abs(throws[0].height_px - 400.0) < 20.0


def test_multiple_throws_detected() -> None:
    fps = 60.0
    ts, ys, xs = [], [], []
    for i in range(5):
        t, y = _parabolic_arc(i * 1.0, 1.0, 200.0, 600.0, fps)
        ts.append(t)
        ys.append(y)
        xs.append(np.full_like(t, 700.0))
    traj = Trajectory(
        track_id=1,
        t=np.concatenate(ts),
        x=np.concatenate(xs),
        y=np.concatenate(ys),
    )
    throws = extract_throws(traj, EventConfig(), hand_line_y=600.0, hands=None)
    assert len(throws) == 5


def test_hand_assignment_unknown_without_pose() -> None:
    fps = 60.0
    t, y = _parabolic_arc(0.0, 1.0, 200.0, 600.0, fps)
    x = np.full_like(t, 700.0)
    traj = Trajectory(track_id=1, t=t, x=x, y=y)
    throws = extract_throws(traj, EventConfig(), 600.0, hands=None)
    assert throws[0].hand == Hand.UNKNOWN


def test_estimate_prop_count() -> None:
    meta = VideoMeta("x.mp4", fps=60.0, total_frames=300, width=1920, height=1080)
    # Three tracks active throughout the early window.
    trajs = [
        Trajectory(
            track_id=i,
            t=np.linspace(0, 5, 50),
            x=np.full(50, 500.0 + i),
            y=np.full(50, 300.0),
        )
        for i in range(3)
    ]
    assert estimate_prop_count(trajs, meta) == 3


def test_analyze_trajectories_end_to_end() -> None:
    fps = 60.0
    meta = VideoMeta("x.mp4", fps=fps, total_frames=300, width=1920, height=1080)
    ts, ys, xs = [], [], []
    for i in range(6):
        t, y = _parabolic_arc(i * 0.5, 0.5, 200.0, 600.0, fps)
        ts.append(t)
        ys.append(y)
        xs.append(np.full_like(t, 700.0))
    traj = Trajectory(
        track_id=1,
        t=np.concatenate(ts),
        x=np.concatenate(xs),
        y=np.concatenate(ys),
    )
    result = analyze_trajectories(meta, [traj], hands=None)
    assert len(result.timelines.throws) == 6
    assert result.metrics.overall_consistency >= 0.0
