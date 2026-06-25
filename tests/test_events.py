"""Tests for event extraction against synthetic trajectories (tech spec §4.4)."""

from __future__ import annotations

import numpy as np

from superjuggling.config import EventConfig, SegmentConfig
from superjuggling.events import _fit_apex_window, extract_events, extract_throws
from superjuggling.models import Hand, Trajectory, VideoMeta
from superjuggling.pipeline import analyze_trajectories, estimate_prop_count
from superjuggling.segments import segment_trajectories_into_flights


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


def test_windowed_apex_fit_uses_local_quadratic_window() -> None:
    fps = 60.0
    t, y = _parabolic_arc(0.0, 1.0, apex_y=200.0, base_y=600.0, fps=fps)
    x = 700.0 + 40.0 * (t - 0.5)

    # Add deterministic detector jitter. The local fit should still recover
    # the underlying apex more accurately than trusting the raw sample.
    y = y + 4.0 * np.sin(np.arange(len(y)) * 1.7)
    x = x + 3.0 * np.cos(np.arange(len(x)) * 1.3)

    traj = Trajectory(track_id=1, t=t, x=x, y=y)
    peak_index = int(np.argmin(y))
    fitted = _fit_apex_window(traj, peak_index, EventConfig())

    assert fitted is not None
    t_apex, y_apex, x_apex = fitted
    assert abs(t_apex - 0.5) < 0.03
    assert abs(y_apex - 200.0) < 8.0
    assert abs(x_apex - 700.0) < 8.0


def test_windowed_apex_fit_falls_back_to_three_point_for_sparse_window() -> None:
    fps = 30.0
    t, y = _parabolic_arc(0.0, 1.0, apex_y=200.0, base_y=600.0, fps=fps)
    x = np.full_like(t, 700.0)
    traj = Trajectory(track_id=1, t=t, x=x, y=y)

    cfg = EventConfig(apex_fit_half_window_s=0.01, apex_fit_min_points=5)
    fitted = _fit_apex_window(traj, int(np.argmin(y)), cfg)

    assert fitted is None

    throws = extract_throws(traj, cfg, hand_line_y=600.0, hands=None)
    assert len(throws) == 1
    assert abs(throws[0].t - 0.5) < 0.05


def test_windowed_apex_fit_rejects_noisy_bad_fit() -> None:
    fps = 60.0
    t, y = _parabolic_arc(0.0, 1.0, apex_y=200.0, base_y=600.0, fps=fps)
    x = np.full_like(t, 700.0)

    # Break the local neighbourhood so the quality gate should refuse the
    # windowed fit rather than inventing a confident-but-wrong apex.
    centre = int(np.argmin(y))
    y[centre - 3 : centre + 4] = np.array(
        [260.0, 180.0, 250.0, 170.0, 245.0, 185.0, 255.0]
    )

    traj = Trajectory(track_id=1, t=t, x=x, y=y)
    cfg = EventConfig(apex_fit_max_rms_px=3.0)
    fitted = _fit_apex_window(traj, int(np.argmin(y)), cfg)

    assert fitted is None


def test_extract_events_prefers_flight_segments_when_available() -> None:
    fps = 60.0
    meta = VideoMeta("x.mp4", fps=fps, total_frames=60, width=1280, height=720)
    t, y = _parabolic_arc(0.0, 1.0, apex_y=200.0, base_y=600.0, fps=fps)
    x = 700.0 + 20.0 * (t - 0.5)
    traj = Trajectory(track_id=1, t=t, x=x, y=y)
    segments = segment_trajectories_into_flights([traj], cfg=SegmentConfig())

    timelines = extract_events([traj], None, meta, EventConfig(), segments=segments)

    assert len(timelines.throws) == 1
    assert abs(timelines.throws[0].t - 0.5) < 0.02
