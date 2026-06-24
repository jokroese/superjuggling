"""Tests for the metrics engine against synthetic timelines (tech spec §5)."""

from __future__ import annotations

import numpy as np

from superjuggling.metrics import compute_metrics, cv, cv_to_score
from superjuggling.models import (
    CatchEvent,
    DropEvent,
    Hand,
    ThrowEvent,
    Timelines,
)


def test_cv_zero_for_constant() -> None:
    assert cv(np.array([5.0, 5.0, 5.0])) == 0.0


def test_cv_positive_for_variance() -> None:
    assert cv(np.array([1.0, 2.0, 3.0])) > 0.0


def test_cv_to_score_inverts_and_clamps() -> None:
    assert cv_to_score(0.0) == 100.0
    assert cv_to_score(1.0) == 0.0
    assert cv_to_score(2.0) == 0.0  # clamped


def _alternating_timeline(
    n: int, interval: float, height: float, jitter: float = 0.0
) -> Timelines:
    """A perfectly steady alternating L/R cascade."""
    throws: list[ThrowEvent] = []
    for i in range(n):
        hand = Hand.RIGHT if i % 2 else Hand.LEFT
        x = 900.0 if hand == Hand.RIGHT else 600.0
        throws.append(
            ThrowEvent(
                t=i * interval + (jitter * (i % 2)),
                track_id=i % 3,
                hand=hand,
                apex_x=x,
                apex_y=300.0,
                height_px=height,
            )
        )
    return Timelines(throws=throws)


def test_steady_pattern_scores_high() -> None:
    tl = _alternating_timeline(n=40, interval=0.25, height=240.0)
    report = compute_metrics(tl, duration_s=10.0)
    assert report.rhythm.iti_cv == 0.0
    assert report.spatial.height_cv == 0.0
    assert report.scores.rhythm == 100.0
    assert report.overall_consistency > 90.0


def test_jittery_rhythm_scores_lower() -> None:
    steady = compute_metrics(_alternating_timeline(40, 0.25, 240.0), duration_s=10.0)
    jittery = compute_metrics(
        _alternating_timeline(40, 0.25, 240.0, jitter=0.1), duration_s=10.0
    )
    assert jittery.rhythm.iti_cv > steady.rhythm.iti_cv
    assert jittery.scores.rhythm < steady.scores.rhythm


def test_cadence_matches_interval() -> None:
    tl = _alternating_timeline(n=21, interval=0.25, height=240.0)
    report = compute_metrics(tl, duration_s=5.0)
    # 20 intervals of 0.25s → 4 throws/s.
    assert abs(report.rhythm.cadence_hz - 4.0) < 1e-6


def test_throw_balance_even() -> None:
    tl = _alternating_timeline(n=40, interval=0.25, height=240.0)
    report = compute_metrics(tl, duration_s=10.0)
    assert abs(report.symmetry.throw_balance - 0.5) < 1e-6


def test_drops_reduce_failure_score() -> None:
    tl = _alternating_timeline(n=40, interval=0.25, height=240.0)
    tl.drops = [DropEvent(t=3.0, track_id=1), DropEvent(t=6.0, track_id=2)]
    report = compute_metrics(tl, duration_s=10.0)
    assert report.failure.drops == 2
    assert report.failure.drop_rate_per_min == 12.0
    # Longest clean run is one of the gaps (0-3, 3-6, 6-10) → 4.0s.
    assert report.failure.longest_clean_run_s == 4.0


def test_dwell_cv_from_catches() -> None:
    throws = [
        ThrowEvent(0.5, 1, Hand.LEFT, 600.0, 300.0, 240.0),
        ThrowEvent(1.0, 1, Hand.LEFT, 600.0, 300.0, 240.0),
    ]
    catches = [
        CatchEvent(0.4, 1, Hand.LEFT),
        CatchEvent(0.9, 1, Hand.LEFT),
    ]
    tl = Timelines(throws=throws, catches=catches)
    report = compute_metrics(tl, duration_s=2.0)
    # Both dwell times are 0.1s → constant → CV 0.
    assert report.rhythm.dwell_cv == 0.0


def test_empty_timeline_is_safe() -> None:
    report = compute_metrics(Timelines(), duration_s=10.0)
    assert report.rhythm.cadence_hz == 0.0
    assert report.failure.drops == 0
