"""Stage 5 — Metrics engine (tech spec §5).

Consumes event timelines and emits consistency metrics. Pure Python / NumPy,
no CV dependency, so it is unit-testable against synthetic timelines.

All consistency metrics are reported as coefficient of variation
(CV = std / mean) plus an inverted 0–100 score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ScoreWeights
from .models import DropEvent, Hand, ThrowEvent, Timelines


def cv(values: NDArray[np.float64]) -> float:
    """Coefficient of variation (std / |mean|). NaN-safe; 0.0 if undefined."""
    if len(values) < 2:
        return 0.0
    mean = float(np.mean(values))
    if mean == 0:
        return 0.0
    return float(np.std(values, ddof=1) / abs(mean))


def cv_to_score(coeff: float) -> float:
    """Invert a CV into a 0–100 consistency score: 100 * (1 - clamp(CV))."""
    return round(100.0 * (1.0 - min(max(coeff, 0.0), 1.0)), 1)


def _intervals(times: list[float]) -> NDArray[np.float64]:
    arr = np.asarray(sorted(times), dtype=np.float64)
    return np.diff(arr) if len(arr) > 1 else np.array([], dtype=np.float64)


@dataclass
class RhythmMetrics:
    cadence_hz: float
    iti_cv: float
    dwell_cv: float
    lr_tempo_balance: float


@dataclass
class SpatialMetrics:
    height_cv: float
    apex_lateral_cv: float
    pattern_width_px: float
    envelope_drift: float
    height_cadence_coherence: float


@dataclass
class SymmetryMetrics:
    lr_height_symmetry: float
    lr_placement_symmetry: float
    throw_balance: float


@dataclass
class FailureMetrics:
    drops: int
    drop_rate_per_min: float
    longest_clean_run_s: float
    recovery_s: float


@dataclass
class Scores:
    rhythm: float
    spatial: float
    symmetry: float
    failure: float


@dataclass
class MetricsReport:
    rhythm: RhythmMetrics
    spatial: SpatialMetrics
    symmetry: SymmetryMetrics
    failure: FailureMetrics
    scores: Scores
    overall_consistency: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rhythm": asdict(self.rhythm),
            "spatial": asdict(self.spatial),
            "symmetry": asdict(self.symmetry),
            "failure": asdict(self.failure),
            "overall_consistency": self.overall_consistency,
            "scores": asdict(self.scores),
        }


# --- §5.1 Rhythm -----------------------------------------------------------


def compute_rhythm(timelines: Timelines) -> RhythmMetrics:
    throw_times = [e.t for e in timelines.throws]
    iti = _intervals(throw_times)

    duration = (throw_times[-1] - throw_times[0]) if len(throw_times) > 1 else 0.0
    cadence = (len(throw_times) - 1) / duration if duration > 0 else 0.0

    dwell = _hand_dwell_times(timelines)

    l_iti = _intervals([e.t for e in timelines.throws_for(Hand.LEFT)])
    r_iti = _intervals([e.t for e in timelines.throws_for(Hand.RIGHT)])
    if len(l_iti) and len(r_iti):
        lm, rm = float(np.mean(l_iti)), float(np.mean(r_iti))
        balance = min(lm, rm) / max(lm, rm) if max(lm, rm) > 0 else 1.0
    else:
        balance = 1.0

    return RhythmMetrics(
        cadence_hz=round(cadence, 3),
        iti_cv=round(cv(iti), 4),
        dwell_cv=round(cv(dwell), 4),
        lr_tempo_balance=round(balance, 4),
    )


def _hand_dwell_times(timelines: Timelines) -> NDArray[np.float64]:
    """Catch-to-next-throw dwell per track (tech spec §4.4 'catch & dwell')."""
    dwell: list[float] = []
    throws_by_track: dict[int, list[float]] = {}
    for e in timelines.throws:
        throws_by_track.setdefault(e.track_id, []).append(e.t)
    for catch in timelines.catches:
        future = [t for t in throws_by_track.get(catch.track_id, []) if t > catch.t]
        if future:
            dwell.append(min(future) - catch.t)
    return np.asarray(dwell, dtype=np.float64)


# --- §5.2 Spatial ----------------------------------------------------------


def compute_spatial(timelines: Timelines) -> SpatialMetrics:
    heights = np.asarray([e.height_px for e in timelines.throws], dtype=np.float64)
    apex_x = np.asarray([e.apex_x for e in timelines.throws], dtype=np.float64)

    width = float(np.ptp(apex_x)) if len(apex_x) else 0.0
    drift = _envelope_drift(timelines)
    coherence = _height_cadence_coherence(timelines)

    return SpatialMetrics(
        height_cv=round(cv(heights), 4),
        apex_lateral_cv=round(_per_hand_lateral_cv(timelines), 4),
        pattern_width_px=round(width, 1),
        envelope_drift=round(drift, 4),
        height_cadence_coherence=round(coherence, 4),
    )


def _per_hand_lateral_cv(timelines: Timelines) -> float:
    cvs: list[float] = []
    for hand in (Hand.LEFT, Hand.RIGHT):
        xs = np.asarray(
            [e.apex_x for e in timelines.throws_for(hand)], dtype=np.float64
        )
        if len(xs) >= 2:
            cvs.append(cv(xs))
    if cvs:
        return float(np.mean(cvs))
    # No hand split (e.g. no pose data) — fall back to global lateral CV.
    xs = np.asarray([e.apex_x for e in timelines.throws], dtype=np.float64)
    return cv(xs)


def _envelope_drift(timelines: Timelines) -> float:
    """How much the apex centroid 'walks' over the clip, normalised by the
    pattern width (tech spec §5.2 'pattern-envelope stability')."""
    throws = sorted(timelines.throws, key=lambda e: e.t)
    if len(throws) < 4:
        return 0.0
    xs = np.asarray([e.apex_x for e in throws], dtype=np.float64)
    half = len(xs) // 2
    drift = abs(float(np.mean(xs[half:])) - float(np.mean(xs[:half])))
    span = float(np.ptp(xs))
    return drift / span if span > 0 else 0.0


def _height_cadence_coherence(timelines: Timelines) -> float:
    """Correlation between throw height and the following interval: physics
    predicts higher throws take longer (tech spec §5.2)."""
    throws = sorted(timelines.throws, key=lambda e: e.t)
    if len(throws) < 3:
        return 0.0
    heights = np.asarray([e.height_px for e in throws[:-1]], dtype=np.float64)
    intervals = np.diff(np.asarray([e.t for e in throws], dtype=np.float64))
    if np.std(heights) == 0 or np.std(intervals) == 0:
        return 0.0
    return float(np.corrcoef(heights, intervals)[0, 1])


# --- §5.3 Symmetry ---------------------------------------------------------


def compute_symmetry(timelines: Timelines) -> SymmetryMetrics:
    left = timelines.throws_for(Hand.LEFT)
    right = timelines.throws_for(Hand.RIGHT)

    height_sym = _ratio([e.height_px for e in left], [e.height_px for e in right])

    n_left, n_right = len(left), len(right)
    total = n_left + n_right
    placement_sym = _placement_symmetry(timelines, left, right)

    return SymmetryMetrics(
        lr_height_symmetry=round(height_sym, 4),
        lr_placement_symmetry=round(placement_sym, 4),
        throw_balance=round((n_left / total) if total else 0.0, 4),
    )


def _ratio(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 1.0
    ma, mb = float(np.mean(a)), float(np.mean(b))
    return min(ma, mb) / max(ma, mb) if max(ma, mb) > 0 else 1.0


def _placement_symmetry(
    timelines: Timelines,
    left: list[ThrowEvent],
    right: list[ThrowEvent],
) -> float:
    """Mirror distance of L vs. R apex x about the pattern centre, normalised
    to a 0–1 symmetry score (tech spec §5.3)."""
    if not left or not right:
        return 1.0
    all_x = [e.apex_x for e in timelines.throws]
    centre = float(np.mean(all_x))
    lx = centre - float(np.mean([e.apex_x for e in left]))
    rx = float(np.mean([e.apex_x for e in right])) - centre
    denom = abs(lx) + abs(rx)
    if denom == 0:
        return 1.0
    return 1.0 - abs(lx - rx) / denom


# --- §5.4 Failure & endurance ---------------------------------------------


def compute_failure(timelines: Timelines, duration_s: float) -> FailureMetrics:
    drops = sorted(timelines.drops, key=lambda d: d.t)
    n_drops = len(drops)
    rate = (n_drops / duration_s * 60.0) if duration_s > 0 else 0.0

    longest = _longest_clean_run(drops, duration_s)
    recovery = _recovery_time(timelines, drops)

    return FailureMetrics(
        drops=n_drops,
        drop_rate_per_min=round(rate, 3),
        longest_clean_run_s=round(longest, 2),
        recovery_s=round(recovery, 2),
    )


def _longest_clean_run(drops: list[DropEvent], duration_s: float) -> float:
    if not drops:
        return duration_s
    boundaries = [0.0, *[d.t for d in drops], duration_s]
    return max(boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1))


def _recovery_time(timelines: Timelines, drops: list[DropEvent]) -> float:
    """Mean time from each drop to the next throw (re-established pattern)."""
    if not drops:
        return 0.0
    throw_times = sorted(e.t for e in timelines.throws)
    recoveries: list[float] = []
    for d in drops:
        nxt = [t for t in throw_times if t > d.t]
        if nxt:
            recoveries.append(min(nxt) - d.t)
    return float(np.mean(recoveries)) if recoveries else 0.0


# --- §5.5 Composite --------------------------------------------------------


def compute_metrics(
    timelines: Timelines,
    duration_s: float,
    weights: ScoreWeights | None = None,
) -> MetricsReport:
    """Run the full metrics engine and assemble the composite score."""
    weights = weights or ScoreWeights()

    rhythm = compute_rhythm(timelines)
    spatial = compute_spatial(timelines)
    symmetry = compute_symmetry(timelines)
    failure = compute_failure(timelines, duration_s)

    rhythm_score = cv_to_score(rhythm.iti_cv)
    spatial_score = cv_to_score(spatial.height_cv)
    symmetry_score = round(100.0 * symmetry.lr_height_symmetry, 1)
    # Failure score: penalise drops and reward long clean runs.
    failure_score = round(
        100.0 * (failure.longest_clean_run_s / duration_s if duration_s else 1.0),
        1,
    )

    scores = Scores(
        rhythm=rhythm_score,
        spatial=spatial_score,
        symmetry=symmetry_score,
        failure=failure_score,
    )

    overall = (
        weights.rhythm * rhythm_score
        + weights.spatial * spatial_score
        + weights.symmetry * symmetry_score
        + weights.failure * failure_score
    ) / (weights.rhythm + weights.spatial + weights.symmetry + weights.failure)

    return MetricsReport(
        rhythm=rhythm,
        spatial=spatial,
        symmetry=symmetry,
        failure=failure,
        scores=scores,
        overall_consistency=round(overall, 1),
    )
