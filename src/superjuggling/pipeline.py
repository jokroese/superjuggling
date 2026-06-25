"""Pipeline orchestration (tech spec §3).

Wires the stages together: ingest → detection → tracking → event extraction →
metrics → report (+ optional annotated video). The CV stages (ingest /
detection / tracking / annotate) require the ``cv`` extra; the analytical
stages (events / metrics / report) are pure and run anywhere.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import events as events_stage
from . import ingest as ingest_stage
from . import report as report_stage
from .config import Config
from .detection import YOLOPoseEstimator, YOLOPropDetector
from .metrics import MetricsReport, compute_metrics
from .models import Hand, HandTrack, Timelines, Trajectory, VideoMeta
from .tracking import TrackAccumulator


@dataclass
class AnalysisResult:
    meta: VideoMeta
    trajectories: list[Trajectory]
    timelines: Timelines
    metrics: MetricsReport
    count_estimate: int
    warnings: list[str]
    # Per-frame overlay data, populated only when the annotated-video output
    # is requested (see ``analyze_video(collect_overlay=True)``).
    frame_detections: list[Any] = field(default_factory=list)
    frame_keypoints: list[Any] = field(default_factory=list)
    # Raw detector output before ByteTrack / smoothing. Populated only when
    # overlay data is collected; used by diagnostic annotation overlays.
    frame_raw_detections: list[Any] = field(default_factory=list)


def estimate_prop_count(trajectories: list[Trajectory], meta: VideoMeta) -> int:
    """Baseline prop count = mode of active-track count over a stable early
    window (tech spec §8 'establishing prop count')."""
    if not trajectories:
        return 0
    window_end = min(meta.duration_s, 5.0)
    # Sample active-track count at evenly spaced instants in the window.
    instants = np.linspace(0.0, window_end, num=20)
    counts: list[int] = []
    for inst in instants:
        active = sum(
            1 for tr in trajectories if len(tr.t) and tr.t[0] <= inst <= tr.t[-1]
        )
        if active:
            counts.append(active)
    return Counter(counts).most_common(1)[0][0] if counts else 0


def analyze_trajectories(
    meta: VideoMeta,
    trajectories: list[Trajectory],
    hands: dict[Hand, HandTrack] | None,
    cfg: Config | None = None,
) -> AnalysisResult:
    """Run the pure analytical half of the pipeline (stages 4–5).

    This is the seam used by tests: feed synthetic trajectories straight in,
    no video required.
    """
    cfg = cfg or Config()
    timelines = events_stage.extract_events(trajectories, hands, meta, cfg.events)
    metrics = compute_metrics(timelines, meta.duration_s, cfg.weights)
    count = estimate_prop_count(trajectories, meta)
    return AnalysisResult(
        meta=meta,
        trajectories=trajectories,
        timelines=timelines,
        metrics=metrics,
        count_estimate=count,
        warnings=[],
    )


def analyze_video(
    path: str,
    cfg: Config | None = None,
    collect_overlay: bool = False,
) -> AnalysisResult:
    """Run the full video pipeline (stages 1–5). Requires the ``cv`` extra.

    When ``collect_overlay`` is set, per-frame tracked detections and pose
    keypoints are retained so an annotated video can be rendered (stage 6a).
    """
    cfg = cfg or Config()

    meta = ingest_stage.probe(path)
    warnings = ingest_stage.validate(meta, cfg.ingest)

    prop_detector = YOLOPropDetector(cfg.detection)
    pose_estimator = YOLOPoseEstimator(cfg.detection)
    accumulator = TrackAccumulator(cfg.tracking, meta.fps, keep_frames=collect_overlay)

    # Stage 2–3: stream frames through detection + tracking.
    keypoints: list[object] = []
    raw_detections: list[object] = []
    for frame in ingest_stage.frames(path):
        detections = prop_detector(frame)
        if collect_overlay:
            raw_detections.append(detections)
        accumulator.update(detections)
        kp = pose_estimator(frame)  # wrist keypoints; aggregation into hands is TODO
        if collect_overlay:
            keypoints.append(kp)

    trajectories = accumulator.trajectories()
    result = analyze_trajectories(meta, trajectories, None, cfg)
    result.warnings = warnings
    if collect_overlay:
        result.frame_detections = accumulator.frame_detections()
        result.frame_keypoints = keypoints
        result.frame_raw_detections = raw_detections
    return result


def run(
    path: str,
    out_dir: Path,
    cfg: Config | None = None,
    annotate: bool = False,
    debug_overlays: bool = False,
) -> AnalysisResult:
    """Full analyze + write report (+ optional annotated video)."""
    cfg = cfg or Config()
    collect_overlay = annotate or debug_overlays
    result = analyze_video(path, cfg, collect_overlay=collect_overlay)
    report_stage.write_report(
        result.meta,
        result.timelines,
        result.metrics,
        out_dir,
        n_tracks=len(result.trajectories),
        count_estimate=result.count_estimate,
    )
    if collect_overlay:
        from . import annotate as annotate_stage

        annotate_stage.annotate_video(
            result, out_dir / "annotated.mp4", cfg, debug_overlays=debug_overlays
        )
    return result
