"""Centre-point candidate generation, filtering, fusion and debug output.

The current detector produces boxes, which are normalised into centre candidates.
Everything downstream consumes ``BallCandidate`` objects.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .config import CandidateConfig
from .models import BallCandidate, CandidateFrame


@runtime_checkable
class CandidateDetector(Protocol):
    """A detector that emits centre-point ball candidates."""

    def __call__(
        self,
        frame: NDArray[np.uint8],
        frame_index: int,
        fps: float,
    ) -> list[BallCandidate]: ...


@runtime_checkable
class TemporalCandidateDetector(Protocol):
    """A detector that emits candidates for the centre frame of a short buffer."""

    def __call__(
        self,
        frames: list[NDArray[np.uint8]],
        centre_frame_index: int,
        fps: float,
    ) -> list[BallCandidate]: ...


def detections_to_candidates(
    detections: Any,
    frame_index: int,
    fps: float,
    source: str = "yolo",
) -> list[BallCandidate]:
    """Convert ``sv.Detections`` boxes into centre-point candidates."""
    if detections is None or not len(detections):
        return []

    t = frame_index / fps if fps else float(frame_index)
    xyxy = np.asarray(detections.xyxy, dtype=np.float64)
    confidence = detections.confidence
    class_id = detections.class_id

    if confidence is None:
        confidence = np.ones(len(xyxy), dtype=np.float64)
    if class_id is None:
        class_id = [None] * len(xyxy)

    out: list[BallCandidate] = []
    for box, score, cls in zip(xyxy, confidence, class_id, strict=False):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        out.append(
            BallCandidate(
                frame_index=frame_index,
                t=t,
                x=(x1 + x2) / 2.0,
                y=(y1 + y2) / 2.0,
                score=float(score),
                radius_px=0.25 * (w + h),
                source=source,
                class_id=None if cls is None else int(cls),
            )
        )
    return out


def filter_candidates(
    candidates: list[BallCandidate],
    cfg: CandidateConfig,
) -> list[BallCandidate]:
    """Apply score and optional ROI filtering."""
    out = [cand for cand in candidates if cand.score >= cfg.min_score]
    if cfg.roi is None:
        return out

    x1, y1, x2, y2 = cfg.roi
    return [cand for cand in out if x1 <= cand.x <= x2 and y1 <= cand.y <= y2]


def fuse_candidates(
    candidates: list[BallCandidate],
    distance_px: float,
) -> list[BallCandidate]:
    """Weighted centre-space NMS/fusion.

    This lets multiple candidate sources coexist without making downstream
    linkers care whether a candidate came from YOLO, sliced inference, or a
    temporal/blob detector.
    """
    if not candidates:
        return []

    remaining = sorted(candidates, key=lambda cand: cand.score, reverse=True)
    fused: list[BallCandidate] = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        keep: list[BallCandidate] = []
        for cand in remaining:
            dist = float(np.hypot(cand.x - seed.x, cand.y - seed.y))
            if dist <= distance_px:
                cluster.append(cand)
            else:
                keep.append(cand)
        remaining = keep

        weights = np.asarray([max(c.score, 1e-6) for c in cluster], dtype=np.float64)
        xs = np.asarray([c.x for c in cluster], dtype=np.float64)
        ys = np.asarray([c.y for c in cluster], dtype=np.float64)
        radii = [c.radius_px for c in cluster if c.radius_px is not None]
        best = max(cluster, key=lambda cand: cand.score)
        total = float(np.sum(weights))
        sources = "+".join(sorted({c.source for c in cluster}))

        fused.append(
            BallCandidate(
                frame_index=best.frame_index,
                t=best.t,
                x=float(np.sum(weights * xs) / total),
                y=float(np.sum(weights * ys) / total),
                score=best.score,
                radius_px=float(np.mean(radii)) if radii else None,
                source=f"fused:{sources}" if len(cluster) > 1 else best.source,
                class_id=best.class_id,
            )
        )

    return sorted(fused, key=lambda cand: cand.score, reverse=True)


def make_candidate_frame(
    frame_index: int,
    fps: float,
    candidates: list[BallCandidate],
) -> CandidateFrame:
    return CandidateFrame(
        frame_index=frame_index,
        t=frame_index / fps if fps else float(frame_index),
        candidates=candidates,
    )


def write_candidates_csv(
    frames: list[CandidateFrame],
    out_path: Path,
) -> Path:
    """Write candidate debug artefact."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame_index",
                "t",
                "x",
                "y",
                "score",
                "radius_px",
                "source",
                "class_id",
            ],
        )
        writer.writeheader()
        for frame in frames:
            for cand in frame.candidates:
                writer.writerow(
                    {
                        "frame_index": cand.frame_index,
                        "t": cand.t,
                        "x": cand.x,
                        "y": cand.y,
                        "score": cand.score,
                        "radius_px": cand.radius_px,
                        "source": cand.source,
                        "class_id": cand.class_id,
                    }
                )
    return out_path
