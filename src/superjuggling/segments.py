"""Flight-segment fitting.

This module converts linked centre trajectories into short fitted ballistic
segments. Event extraction can then prefer segment apexes while retaining the
older trajectory-minimum fallback.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import SegmentConfig
from .models import FlightSegment, Trajectory


def _split_indices_by_gap(t: np.ndarray, max_gap_s: float) -> list[slice]:
    if len(t) == 0:
        return []
    starts = [0]
    for i, dt in enumerate(np.diff(t), start=1):
        if float(dt) > max_gap_s:
            starts.append(i)
    stops = [*starts[1:], len(t)]
    return [slice(a, b) for a, b in zip(starts, stops, strict=False)]


def fit_flight_segment(
    segment_id: int,
    track_id: int | None,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    confidence: np.ndarray | None,
    cfg: SegmentConfig,
) -> FlightSegment | None:
    """Fit x-linear/y-quadratic model to one candidate segment."""
    if len(t) < cfg.min_points:
        return None

    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < cfg.min_points:
        return None

    t = t[finite]
    x = x[finite]
    y = y[finite]
    conf = (
        None if confidence is None else np.asarray(confidence, dtype=np.float64)[finite]
    )

    t_ref = float(np.mean(t))
    u = t - t_ref
    x_design = np.column_stack((u, np.ones_like(u)))
    y_design = np.column_stack((u * u, u, np.ones_like(u)))

    if conf is None:
        weights = np.ones(len(t), dtype=np.float64)
    else:
        weights = np.where(np.isfinite(conf) & (conf > 0), conf, 0.0)
        if float(np.max(weights, initial=0.0)) <= 0:
            weights = np.ones(len(t), dtype=np.float64)
        else:
            weights = weights / float(np.max(weights))

    try:
        sw = np.sqrt(weights)
        x_coeff, *_ = np.linalg.lstsq(x_design * sw[:, None], x * sw, rcond=None)
        y_coeff, *_ = np.linalg.lstsq(y_design * sw[:, None], y * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None

    x_pred = x_design @ x_coeff
    y_pred = y_design @ y_coeff
    rms = float(np.sqrt(np.mean((x_pred - x) ** 2 + (y_pred - y) ** 2)))
    if rms > cfg.max_rms_px:
        return None

    return FlightSegment(
        segment_id=segment_id,
        track_id=track_id,
        t=t,
        x=x,
        y=y,
        confidence=conf,
        t_ref=t_ref,
        coeff_x=(float(x_coeff[0]), float(x_coeff[1])),
        coeff_y=(float(y_coeff[0]), float(y_coeff[1]), float(y_coeff[2])),
        rms_error_px=rms,
    )


def segment_trajectories_into_flights(
    trajectories: list[Trajectory],
    cfg: SegmentConfig,
) -> list[FlightSegment]:
    """Split trajectories on large gaps and fit a flight model to each piece."""
    if not cfg.enabled:
        return []

    segments: list[FlightSegment] = []
    next_id = 1
    for traj in trajectories:
        for slc in _split_indices_by_gap(traj.t, cfg.max_gap_s):
            conf = None if traj.confidence is None else traj.confidence[slc]
            fitted = fit_flight_segment(
                segment_id=next_id,
                track_id=traj.track_id,
                t=traj.t[slc],
                x=traj.x[slc],
                y=traj.y[slc],
                confidence=conf,
                cfg=cfg,
            )
            if fitted is not None:
                segments.append(fitted)
                next_id += 1
    return segments


def segment_apex(segment: FlightSegment) -> tuple[float, float, float] | None:
    """Return fitted apex ``(t, x, y)`` for a valid segment."""
    if segment.coeff_x is None or segment.coeff_y is None:
        return None
    a, b, c = segment.coeff_y
    if a <= 0:
        return None
    u = -b / (2.0 * a)
    t_apex = segment.t_ref + u
    if not (float(np.min(segment.t)) <= t_apex <= float(np.max(segment.t))):
        return None
    m, k = segment.coeff_x
    x_apex = m * u + k
    y_apex = a * u * u + b * u + c
    return float(t_apex), float(x_apex), float(y_apex)


def write_segments_csv(segments: list[FlightSegment], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "segment_id",
                "track_id",
                "t_start",
                "t_end",
                "n_points",
                "rms_error_px",
                "t_apex",
                "x_apex",
                "y_apex",
                "status",
            ],
        )
        writer.writeheader()
        for seg in segments:
            apex = segment_apex(seg)
            writer.writerow(
                {
                    "segment_id": seg.segment_id,
                    "track_id": seg.track_id,
                    "t_start": float(np.min(seg.t)),
                    "t_end": float(np.max(seg.t)),
                    "n_points": len(seg.t),
                    "rms_error_px": seg.rms_error_px,
                    "t_apex": None if apex is None else apex[0],
                    "x_apex": None if apex is None else apex[1],
                    "y_apex": None if apex is None else apex[2],
                    "status": "ok" if apex is not None else "no_apex",
                }
            )
    return out_path
