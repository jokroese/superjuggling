"""Multi-frame heatmap candidate detection.

TrackNet-style ball trackers use several consecutive frames and predict a
ball-location heatmap. This module implements the same interface with a simple
model-free motion heatmap: temporal differences around the centre frame are
blurred, normalised, peak-picked and converted into ``BallCandidate`` objects.

It is intentionally conservative and inspectable. A learned heatmap model can
replace ``motion_heatmap`` later without changing the candidate/linking layers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, maximum_filter

from .config import HeatmapConfig
from .models import BallCandidate


def _to_gray(frame: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Convert RGB/BGR/grayscale frame to float grayscale in [0, 1]."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        gray = arr.astype(np.float64)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # OpenCV/supervision frames are usually BGR. The exact channel order is
        # not important for motion differencing, so use a stable luminance-ish
        # average weighted toward green.
        gray = (
            0.114 * arr[..., 0].astype(np.float64)
            + 0.587 * arr[..., 1].astype(np.float64)
            + 0.299 * arr[..., 2].astype(np.float64)
        )
    else:
        msg = f"unsupported frame shape for heatmap detection: {arr.shape}"
        raise ValueError(msg)
    return gray / 255.0 if gray.max(initial=0.0) > 1.0 else gray


def motion_heatmap(
    frames: Sequence[NDArray[np.uint8]],
    cfg: HeatmapConfig,
) -> NDArray[np.float64]:
    """Return a normalised centre-frame motion heatmap.

    The heatmap is the max absolute difference between the centre frame and its
    temporal neighbours, smoothed and normalised to [0, 1].
    """
    if len(frames) < 3 or len(frames) % 2 == 0:
        msg = "motion_heatmap expects an odd window with at least 3 frames"
        raise ValueError(msg)

    centre_idx = len(frames) // 2
    centre = _to_gray(frames[centre_idx])
    diffs: list[NDArray[np.float64]] = []
    for i, frame in enumerate(frames):
        if i == centre_idx:
            continue
        other = _to_gray(frame)
        if other.shape != centre.shape:
            msg = "all heatmap frames must have the same shape"
            raise ValueError(msg)
        diffs.append(np.abs(centre - other))

    heatmap = np.max(np.stack(diffs, axis=0), axis=0)
    if cfg.blur_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=cfg.blur_sigma)

    # Robust normalisation avoids one hot pixel flattening the map.
    lo = float(np.percentile(heatmap, 50.0))
    hi = float(np.percentile(heatmap, 99.9))
    if hi <= lo:
        return np.zeros_like(heatmap, dtype=np.float64)
    heatmap = np.clip((heatmap - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(heatmap, dtype=np.float64)


def heatmap_to_candidates(
    heatmap: NDArray[np.float64],
    *,
    frame_index: int,
    fps: float,
    cfg: HeatmapConfig,
    source: str = "heatmap",
) -> list[BallCandidate]:
    """Peak-pick a normalised heatmap into centre candidates."""
    if heatmap.ndim != 2:
        msg = "heatmap_to_candidates expects a 2D heatmap"
        raise ValueError(msg)
    if heatmap.size == 0:
        return []

    threshold = max(
        cfg.min_score,
        float(np.percentile(heatmap, cfg.threshold_percentile)),
    )

    size = max(1, int(cfg.nms_radius_px) * 2 + 1)
    local_max = maximum_filter(heatmap, size=size, mode="nearest")
    mask = (heatmap >= threshold) & (heatmap == local_max)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []

    scores = heatmap[ys, xs]
    order = np.argsort(scores)[::-1][: cfg.max_candidates_per_frame]
    t = frame_index / fps if fps else float(frame_index)

    candidates: list[BallCandidate] = []
    for idx in order:
        y = float(ys[idx])
        x = float(xs[idx])
        score = float(scores[idx])
        candidates.append(
            BallCandidate(
                frame_index=frame_index,
                t=t,
                x=x,
                y=y,
                score=score,
                radius_px=float(max(3, cfg.nms_radius_px)),
                source=source,
            )
        )
    return candidates


class MultiFrameHeatmapDetector:
    """Model-free multi-frame heatmap detector.

    ``frames`` must be an odd-length window centred on ``centre_frame_index``.
    """

    def __init__(self, cfg: HeatmapConfig) -> None:
        self.cfg = cfg

    @property
    def window_size(self) -> int:
        return self.cfg.window_radius * 2 + 1

    def __call__(
        self,
        frames: Sequence[NDArray[np.uint8]],
        centre_frame_index: int,
        fps: float,
    ) -> list[BallCandidate]:
        if len(frames) != self.window_size:
            msg = (
                f"expected {self.window_size} frames for heatmap detector, "
                f"got {len(frames)}"
            )
            raise ValueError(msg)
        heatmap = motion_heatmap(frames, self.cfg)
        return heatmap_to_candidates(
            heatmap,
            frame_index=centre_frame_index,
            fps=fps,
            cfg=self.cfg,
        )


def pad_window(
    frames: Sequence[NDArray[np.uint8]],
    centre: int,
    radius: int,
) -> list[NDArray[np.uint8]]:
    """Return an odd frame window, clamping at sequence boundaries."""
    if not frames:
        return []
    out: list[NDArray[np.uint8]] = []
    last = len(frames) - 1
    for idx in range(centre - radius, centre + radius + 1):
        out.append(frames[min(max(idx, 0), last)])
    return out


def detect_heatmap_candidate_frames(
    frames: Sequence[NDArray[np.uint8]],
    fps: float,
    cfg: HeatmapConfig,
) -> list[list[BallCandidate]]:
    """Detect heatmap candidates for every frame in a loaded clip."""
    detector = MultiFrameHeatmapDetector(cfg)
    return [
        detector(pad_window(frames, i, cfg.window_radius), i, fps)
        for i in range(len(frames))
    ]
