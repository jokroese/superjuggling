"""Tunable configuration for the pipeline.

Everything that the tech spec calls out as "tuned" or "configurable" lives
here so it can be tweaked per clip / prop-count tier without touching logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # Smoke-test escape hatch (tech spec §4.2): when no fine-tuned weights exist,
    # fall back to COCO ``yolov8n.pt`` filtered to the "sports ball" class. The
    # spec warns these detections are unreliable for fast props — plumbing only.
    allow_coco: bool = False
    coco_model_path: str = "yolov8n.pt"
    coco_ball_class_id: int = 32  # COCO "sports ball"


@dataclass
class TrackingConfig:
    """ByteTrack / smoothing tuning for fast small objects (tech spec §4.3)."""

    track_activation_threshold: float = 0.2
    lost_track_buffer: int = 60
    minimum_matching_threshold: float = 0.8
    smoother_length: int = 5


@dataclass
class EventConfig:
    """Apex / catch / drop extraction parameters (tech spec §4.4)."""

    # Minimum vertical prominence (px) of an apex to count as a throw — filters
    # micro-jitter.
    apex_min_prominence: float = 15.0
    # Refractory period between consecutive throws on one trajectory (seconds).
    min_inter_throw_s: float = 0.15
    # Refine apex timing/height with a local quadratic fit. This is intentionally
    # short-window rather than whole-flight physics fitting: the tracker can
    # fragment or swap IDs, but the samples near a detected apex are usually the
    # most reliable part of the flight.
    parabolic_refine: bool = True
    # Half-window around a detected apex candidate used for the local quadratic
    # fit. At 60 fps, 0.10s gives up to ~13 samples; at 30 fps, ~7 samples.
    apex_fit_half_window_s: float = 0.10
    # Minimum samples required before using the windowed fit. Otherwise the
    # extractor falls back to the previous 3-point vertex refinement.
    apex_fit_min_points: int = 5
    # Reject noisy local fits rather than trusting a bad parabola.
    apex_fit_max_rms_px: float = 25.0
    # A drop is confirmed when a lost track's last centre sits in the bottom
    # fraction of the frame.
    floor_zone_fraction: float = 0.85
    # Frames a track must be missing before it is considered lost.
    lost_after_frames: int = 60


@dataclass
class ScoreWeights:
    """Weights for the composite consistency score (tech spec §5.5)."""

    rhythm: float = 0.3
    spatial: float = 0.3
    symmetry: float = 0.2
    failure: float = 0.2


@dataclass
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    events: EventConfig = field(default_factory=EventConfig)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
