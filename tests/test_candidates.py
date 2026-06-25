from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from superjuggling.candidates import (
    detections_to_candidates,
    filter_candidates,
    fuse_candidates,
)
from superjuggling.config import CandidateConfig
from superjuggling.models import BallCandidate


@dataclass
class FakeDetections:
    xyxy: np.ndarray
    confidence: np.ndarray | None = None
    class_id: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.xyxy)


def test_detections_to_candidates_uses_box_centres() -> None:
    dets = FakeDetections(
        xyxy=np.asarray([[10.0, 20.0, 30.0, 60.0]]),
        confidence=np.asarray([0.75]),
        class_id=np.asarray([32]),
    )

    candidates = detections_to_candidates(dets, frame_index=12, fps=60.0)

    assert len(candidates) == 1
    assert candidates[0].t == 0.2
    assert candidates[0].x == 20.0
    assert candidates[0].y == 40.0
    assert candidates[0].score == 0.75
    assert candidates[0].radius_px == 15.0
    assert candidates[0].class_id == 32


def test_detections_to_candidates_defaults_missing_confidence() -> None:
    dets = FakeDetections(xyxy=np.asarray([[0.0, 0.0, 10.0, 10.0]]))

    candidates = detections_to_candidates(dets, frame_index=0, fps=30.0)

    assert len(candidates) == 1
    assert candidates[0].score == 1.0


def test_fuse_candidates_empty() -> None:
    assert fuse_candidates([], distance_px=5.0) == []


def test_filter_candidates_applies_score_and_roi() -> None:
    candidates = [
        BallCandidate(0, 0.0, 50.0, 50.0, 0.9),
        BallCandidate(0, 0.0, 500.0, 50.0, 0.9),
        BallCandidate(0, 0.0, 50.0, 50.0, 0.01),
    ]
    cfg = CandidateConfig(min_score=0.05, roi=(0.0, 0.0, 100.0, 100.0))

    out = filter_candidates(candidates, cfg)

    assert len(out) == 1
    assert out[0].x == 50.0


def test_fuse_candidates_clusters_nearby_points() -> None:
    candidates = [
        BallCandidate(0, 0.0, 10.0, 10.0, 1.0, source="a"),
        BallCandidate(0, 0.0, 12.0, 10.0, 0.5, source="b"),
        BallCandidate(0, 0.0, 100.0, 100.0, 0.8, source="c"),
    ]

    fused = fuse_candidates(candidates, distance_px=5.0)

    assert len(fused) == 2
    assert abs(fused[0].x - 10.666) < 0.01
    assert fused[0].source == "fused:a+b"
    assert fused[0].score == 1.0
