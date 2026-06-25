"""Stage 4 — Event extraction (tech spec §4.4).

Turns smoothed prop trajectories into discrete, timestamped throw / catch /
drop events. Pure NumPy + SciPy: no CV dependency, so it is unit-testable
against synthetic trajectories.

Detected apex candidates are refined with a short-window quadratic fit, which
stabilises apex timing/height while avoiding brittle whole-flight assumptions.
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


def _confidence_weights(
    traj: Trajectory,
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return normalised positive weights from detection confidence.

    Missing confidence falls back to equal weights. The square root is applied
    later when forming the weighted least-squares design matrix.
    """
    n = int(np.sum(mask))
    if traj.confidence is None:
        return np.ones(n, dtype=np.float64)

    weights = np.asarray(traj.confidence[mask], dtype=np.float64)
    valid = np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.ones(n, dtype=np.float64)

    weights = np.where(valid, weights, 0.0)
    max_weight = float(np.max(weights))
    if max_weight <= 0:
        return np.ones(n, dtype=np.float64)
    return weights / max_weight


def _weighted_lstsq(
    design: NDArray[np.float64],
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Solve a small weighted least-squares problem.

    ``np.linalg.lstsq`` is enough here because the models are linear in their
    coefficients: y = a*u² + b*u + c and x = m*u + k.
    """
    try:
        sqrt_w = np.sqrt(weights)
        lhs = design * sqrt_w[:, None]
        rhs = values * sqrt_w
        coeffs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return np.asarray(coeffs, dtype=np.float64)


def _fit_apex_window(
    traj: Trajectory,
    peak_index: int,
    cfg: EventConfig,
) -> tuple[float, float, float] | None:
    """Refine one apex candidate with a short local quadratic fit.

    The peak detector proposes a local y-minimum. Around that candidate, fit:

    - y(u) = a*u² + b*u + c for the vertical ballistic shape
    - x(u) = m*u + k for lateral interpolation at the refined apex time

    where u = t - t_peak. Centring keeps the tiny polynomial system well
    conditioned. The fit is accepted only when it has positive curvature, the
    vertex remains inside the local window, and the vertical RMS residual is
    small enough.
    """
    if cfg.apex_fit_half_window_s <= 0:
        return None

    t_peak = float(traj.t[peak_index])
    mask = np.abs(traj.t - t_peak) <= cfg.apex_fit_half_window_s
    if int(np.sum(mask)) < cfg.apex_fit_min_points:
        return None

    t_win = np.asarray(traj.t[mask], dtype=np.float64)
    x_win = np.asarray(traj.x[mask], dtype=np.float64)
    y_win = np.asarray(traj.y[mask], dtype=np.float64)
    weights = _confidence_weights(traj, mask)

    finite = np.isfinite(t_win) & np.isfinite(x_win) & np.isfinite(y_win)
    if int(np.sum(finite)) < cfg.apex_fit_min_points:
        return None

    t_win = t_win[finite]
    x_win = x_win[finite]
    y_win = y_win[finite]
    weights = weights[finite]

    u = t_win - t_peak
    y_design = np.column_stack((u * u, u, np.ones_like(u)))
    y_coeffs = _weighted_lstsq(y_design, y_win, weights)
    if y_coeffs is None:
        return None

    a, b, c = [float(v) for v in y_coeffs]
    # Image y grows downward, so a visible throw apex is a local minimum and
    # therefore needs positive curvature in y(t).
    if a <= 0:
        return None

    vertex_u = -b / (2.0 * a)
    if abs(vertex_u) > cfg.apex_fit_half_window_s:
        return None
    if not (float(np.min(u)) <= vertex_u <= float(np.max(u))):
        return None

    y_pred = y_design @ y_coeffs
    rms = float(np.sqrt(np.mean((y_pred - y_win) ** 2)))
    if rms > cfg.apex_fit_max_rms_px:
        return None

    x_design = np.column_stack((u, np.ones_like(u)))
    x_coeffs = _weighted_lstsq(x_design, x_win, weights)
    if x_coeffs is None:
        return None

    m, k = [float(v) for v in x_coeffs]
    t_apex = t_peak + vertex_u
    y_apex = a * vertex_u * vertex_u + b * vertex_u + c
    x_apex = m * vertex_u + k
    return t_apex, y_apex, x_apex


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
        if cfg.parabolic_refine:
            fitted = _fit_apex_window(traj, int(p), cfg)
            if fitted is not None:
                t_apex, y_apex, x_apex = fitted
            elif 0 < p < len(traj.t) - 1:
                # Conservative fallback for sparse windows: preserve the old
                # 3-point sub-frame refinement behaviour.
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
