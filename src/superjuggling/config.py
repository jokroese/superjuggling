"""Tunable configuration for the ball-tracking pipeline.

Keep this lean while the project is pre-release: expose only settings that
directly affect candidate generation, linking and tracking artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class IngestConfig:
    """Validation guardrails for ingest (tech spec §4.1)."""

    min_height: int = 720
    min_fps: float = 30.0
    recommended_fps: float = 60.0
    max_duration_s: float = 600.0


@dataclass
class DetectionConfig:
    """Detector thresholds (tech spec §4.2)."""

    model_path: str | None = None  # path to fine-tuned YOLO prop weights
    pose_model_path: str | None = None
    # Props are small and motion-blurred at the apex, so the threshold is low
    # and we lean on tracking + smoothing to suppress spurious detections.
    confidence: float = 0.2
    # Candidate source used by the tracking workbench. ``heatmap`` is a
    # multi-frame motion heatmap detector; ``hybrid`` fuses YOLO boxes and
    # heatmap peaks before linking.
    candidate_source: Literal["yolo", "heatmap", "hybrid"] = "hybrid"
    # Smoke-test escape hatch (tech spec §4.2): when no fine-tuned weights exist,
    # fall back to COCO ``yolov8n.pt`` filtered to the "sports ball" class. The
    # spec warns these detections are unreliable for fast props — plumbing only.
    allow_coco: bool = False
    coco_model_path: str = "yolov8n.pt"
    coco_ball_class_id: int = 32  # COCO "sports ball"


@dataclass
class CandidateConfig:
    """Centre-candidate filtering/fusion settings."""

    min_score: float = 0.05
    fusion_distance_px: float = 20.0
    roi: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2


@dataclass
class HeatmapConfig:
    """Multi-frame motion heatmap candidate settings.

    This is deliberately model-free: it gives us a TrackNet-style interface
    now, using short-window temporal difference heatmaps, and can later be
    replaced by a learned heatmap model without changing downstream linking.
    """

    # Number of neighbour frames on each side of the centre frame.
    window_radius: int = 2
    # Minimum normalised heatmap score for a peak candidate.
    min_score: float = 0.18
    # Percentile threshold adapts to lighting/background motion per frame.
    threshold_percentile: float = 99.4
    # Local-maximum suppression radius in heatmap pixels.
    nms_radius_px: int = 10
    max_candidates_per_frame: int = 12
    blur_sigma: float = 1.2


@dataclass
class LinkingConfig:
    """Candidate linking backend and association thresholds."""

    backend: Literal["bytetrack", "centre", "ballistic"] = "bytetrack"
    max_match_distance_px: float = 80.0
    max_gap_s: float = 0.18
    min_track_points: int = 3
    confidence_bonus_px: float = 12.0
    physics_residual_weight: float = 0.35


@dataclass
class TrackingConfig:
    """ByteTrack / smoothing tuning for fast small objects (tech spec §4.3)."""

    track_activation_threshold: float = 0.2
    lost_track_buffer: int = 60
    minimum_matching_threshold: float = 0.8
    smoother_length: int = 5


@dataclass
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
