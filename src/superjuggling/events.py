"""Stage 4 — Event extraction (tech spec §4.4).

Turns smoothed prop trajectories into discrete, timestamped throw / catch /
drop events. Pure NumPy + SciPy: no CV dependency, so it is unit-testable
against synthetic trajectories.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .config import EventConfig
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


def _parabolic_vertex(
    x0: float, x1: float, x2: float, y0: float, y1: float, y2: float
) -> tuple[float, float]:
    """Vertex (x, y) of the parabola through three evenly-or-unevenly spaced
    points, used for sub-frame apex refinement (tech spec §4.4 step 4)."""
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return x1, y1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a == 0:
        return x1, y1
    vx = -b / (2 * a)
    c = y1 - a * x1 * x1 - b * x1
    vy = a * vx * vx + b * vx + c
    return vx, vy


def _hand_line(hands: dict[Hand, HandTrack] | None, default_y: float) -> float:
    """Approximate the y of the hand line (mean wrist height)."""
    if not hands:
        return default_y
    ys = [float(np.nanmean(h.y)) for h in hands.values() if len(h.y)]
    return float(np.mean(ys)) if ys else default_y


def _assign_hand(apex_x: float, t: float, hands: dict[Hand, HandTrack] | None) -> Hand:
    """Attribute a throw to the nearest wrist at the apex time (tech spec
    §4.4 'hand assignment'). Falls back to side-of-frame when no pose data."""
    if not hands:
        return Hand.UNKNOWN
    best: tuple[float, Hand] | None = None
    for hand, track in hands.items():
        if not len(track.t):
            continue
        idx = int(np.argmin(np.abs(track.t - t)))
        dist = abs(track.x[idx] - apex_x)
        if best is None or dist < best[0]:
            best = (dist, hand)
    return best[1] if best else Hand.UNKNOWN


def extract_throws(
    traj: Trajectory,
    cfg: EventConfig,
    hand_line_y: float,
    hands: dict[Hand, HandTrack] | None,
) -> list[ThrowEvent]:
    """Find throw apexes for one trajectory (local minima of y)."""
    if len(traj.t) < 3:
        return []
    dt = float(np.median(np.diff(traj.t))) if len(traj.t) > 1 else 1.0
    distance = max(1, int(round(cfg.min_inter_throw_s / dt))) if dt > 0 else 1

    # Apex = local minimum of y; find_peaks works on maxima, so negate.
    peaks, _ = find_peaks(
        -traj.y, prominence=cfg.apex_min_prominence, distance=distance
    )

    throws: list[ThrowEvent] = []
    for p in peaks:
        t_apex, y_apex = float(traj.t[p]), float(traj.y[p])
        x_apex = float(traj.x[p])
        if cfg.parabolic_refine and 0 < p < len(traj.t) - 1:
            t_apex, y_apex = _parabolic_vertex(
                float(traj.t[p - 1]),
                float(traj.t[p]),
                float(traj.t[p + 1]),
                float(traj.y[p - 1]),
                float(traj.y[p]),
                float(traj.y[p + 1]),
            )
        throws.append(
            ThrowEvent(
                t=t_apex,
                track_id=traj.track_id,
                hand=_assign_hand(x_apex, t_apex, hands),
                apex_x=x_apex,
                apex_y=y_apex,
                # Image y grows downward: a higher throw is a smaller y, so
                # height above the hand line is (hand_line_y - apex_y).
                height_px=hand_line_y - y_apex,
            )
        )
    return throws


def extract_catches(
    traj: Trajectory,
    cfg: EventConfig,
    hand_line_y: float,
    hands: dict[Hand, HandTrack] | None,
) -> list[CatchEvent]:
    """A catch is the trajectory returning to the hand line — a local maximum
    of y (tech spec §4.4 'catch & dwell')."""
    if len(traj.t) < 3:
        return []
    dt = float(np.median(np.diff(traj.t))) if len(traj.t) > 1 else 1.0
    distance = max(1, int(round(cfg.min_inter_throw_s / dt))) if dt > 0 else 1
    peaks, _ = find_peaks(traj.y, prominence=cfg.apex_min_prominence, distance=distance)
    catches: list[CatchEvent] = []
    for p in peaks:
        # Only count maxima that actually reach the hand line region.
        if traj.y[p] < hand_line_y - cfg.apex_min_prominence:
            continue
        catches.append(
            CatchEvent(
                t=float(traj.t[p]),
                track_id=traj.track_id,
                hand=_assign_hand(float(traj.x[p]), float(traj.t[p]), hands),
            )
        )
    return catches


def detect_drops(
    trajectories: list[Trajectory],
    cfg: EventConfig,
    meta: VideoMeta,
) -> list[DropEvent]:
    """Infer drops from tracks that terminate in the floor zone (§4.4)."""
    floor_y = meta.height * cfg.floor_zone_fraction
    end_t = meta.duration_s
    drops: list[DropEvent] = []
    for traj in trajectories:
        if not len(traj.t):
            continue
        last_t = float(traj.t[-1])
        last_y = float(traj.y[-1])
        # Track ended well before the clip did (it was lost) and its last
        # known position is low in the frame → a drop.
        ended_early = (end_t - last_t) > (cfg.lost_after_frames / meta.fps)
        if ended_early and last_y >= floor_y:
            drops.append(DropEvent(t=last_t, track_id=traj.track_id))
    return drops


def extract_events(
    trajectories: list[Trajectory],
    hands: dict[Hand, HandTrack] | None,
    meta: VideoMeta,
    cfg: EventConfig,
) -> Timelines:
    """Run the full event-extraction stage over all trajectories."""
    default_hand_line = meta.height * 0.8
    hand_line_y = _hand_line(hands, default_hand_line)

    timelines = Timelines()
    for traj in trajectories:
        timelines.throws.extend(extract_throws(traj, cfg, hand_line_y, hands))
        timelines.catches.extend(extract_catches(traj, cfg, hand_line_y, hands))
    timelines.drops.extend(detect_drops(trajectories, cfg, meta))

    timelines.throws.sort(key=lambda e: e.t)
    timelines.catches.sort(key=lambda e: e.t)
    timelines.drops.sort(key=lambda e: e.t)
    return timelines


def smooth(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Simple centred moving-average smoother for trajectory denoising."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")
