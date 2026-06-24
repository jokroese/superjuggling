"""Stage 3 — Tracking & smoothing (tech spec §4.3).

Wraps ``sv.ByteTrack`` (persistent prop IDs) and ``sv.DetectionsSmoother``
(temporal smoothing), accumulating per-track ``(t, x, y)`` trajectories.

Requires the ``cv`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ._optional import require
from .config import TrackingConfig
from .models import Trajectory


class TrackAccumulator:
    """Streams per-frame detections through ByteTrack + smoothing and
    accumulates one ``Trajectory`` per ``tracker_id`` (§4.3)."""

    def __init__(self, cfg: TrackingConfig, fps: float) -> None:
        sv = require("supervision")
        self._fps = fps
        self._tracker = sv.ByteTrack(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            frame_rate=int(round(fps)) or 30,
        )
        self._smoother = sv.DetectionsSmoother(length=cfg.smoother_length)
        self._buf: dict[int, list[tuple[float, float, float, float]]] = {}
        self._frame_idx = 0

    def update(self, detections: Any) -> None:
        """Advance one frame of detections."""
        tracked = self._tracker.update_with_detections(detections)
        tracked = self._smoother.update_with_detections(tracked)
        t = self._frame_idx / self._fps if self._fps else float(self._frame_idx)
        xyxy = tracked.xyxy
        ids = tracked.tracker_id
        conf = (
            tracked.confidence if tracked.confidence is not None else np.ones(len(xyxy))
        )
        if ids is not None:
            for box, tid, c in zip(xyxy, ids, conf, strict=False):
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                self._buf.setdefault(int(tid), []).append((t, cx, cy, float(c)))
        self._frame_idx += 1

    def trajectories(self) -> list[Trajectory]:
        out: list[Trajectory] = []
        for tid, samples in self._buf.items():
            arr = np.asarray(samples, dtype=np.float64)
            out.append(
                Trajectory(
                    track_id=tid,
                    t=arr[:, 0],
                    x=arr[:, 1],
                    y=arr[:, 2],
                    confidence=arr[:, 3],
                )
            )
        return out


def accumulate(
    detections_per_frame: Iterable[Any],
    cfg: TrackingConfig,
    fps: float,
) -> list[Trajectory]:
    """Convenience: run a sequence of per-frame detections to trajectories."""
    acc = TrackAccumulator(cfg, fps)
    for det in detections_per_frame:
        acc.update(det)
    return acc.trajectories()
