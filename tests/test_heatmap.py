from __future__ import annotations

import numpy as np

from superjuggling.config import HeatmapConfig
from superjuggling.heatmap import (
    MultiFrameHeatmapDetector,
    heatmap_to_candidates,
    motion_heatmap,
    pad_window,
)


def _frame_with_dot(x: int, y: int, size: int = 64) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 255
    return frame


def test_motion_heatmap_highlights_moving_dot() -> None:
    frames = [
        _frame_with_dot(20, 30),
        _frame_with_dot(24, 30),
        _frame_with_dot(28, 30),
        _frame_with_dot(32, 30),
        _frame_with_dot(36, 30),
    ]

    heatmap = motion_heatmap(frames, HeatmapConfig())

    assert heatmap.shape == (64, 64)
    assert float(np.max(heatmap)) > 0.5


def test_heatmap_to_candidates_peak_picks_local_maxima() -> None:
    heatmap = np.zeros((64, 64), dtype=np.float64)
    heatmap[30, 20] = 0.9
    heatmap[10, 10] = 0.4

    candidates = heatmap_to_candidates(
        heatmap,
        frame_index=6,
        fps=60.0,
        cfg=HeatmapConfig(min_score=0.2, threshold_percentile=99.0),
    )

    assert candidates
    assert candidates[0].frame_index == 6
    assert candidates[0].t == 0.1
    assert candidates[0].x == 20.0
    assert candidates[0].y == 30.0
    assert candidates[0].source == "heatmap"


def test_multiframe_heatmap_detector_returns_centre_candidates() -> None:
    detector = MultiFrameHeatmapDetector(
        HeatmapConfig(min_score=0.1, threshold_percentile=99.0)
    )
    frames = [
        _frame_with_dot(20, 30),
        _frame_with_dot(24, 30),
        _frame_with_dot(28, 30),
        _frame_with_dot(32, 30),
        _frame_with_dot(36, 30),
    ]

    candidates = detector(frames, centre_frame_index=2, fps=60.0)

    assert candidates
    assert all(c.source == "heatmap" for c in candidates)


def test_pad_window_clamps_at_boundaries() -> None:
    frames = [_frame_with_dot(i + 10, 20) for i in range(3)]

    window = pad_window(frames, centre=0, radius=2)

    assert len(window) == 5
    assert window[0] is frames[0]
    assert window[1] is frames[0]
    assert window[-1] is frames[2]
