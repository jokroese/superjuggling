"""Pipeline orchestration (tech spec §3).

Wires the stages together: ingest → detection → tracking → event extraction →
metrics → report (+ optional annotated video). The CV stages (ingest /
detection / tracking / annotate) require the ``cv`` extra; the analytical
stages (events / metrics / report) are pure and run anywhere.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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
) -> AnalysisResult:
    """Run the full video pipeline (stages 1–5). Requires the ``cv`` extra."""
    cfg = cfg or Config()

    meta = ingest_stage.probe(path)
    warnings = ingest_stage.validate(meta, cfg.ingest)

    prop_detector = YOLOPropDetector(cfg.detection)
    pose_estimator = YOLOPoseEstimator(cfg.detection)
    accumulator = TrackAccumulator(cfg.tracking, meta.fps)

    # Stage 2–3: stream frames through detection + tracking.
    # Pose/wrist trajectories are accumulated alongside (draft: kept minimal).
    for frame in ingest_stage.frames(path):
        detections = prop_detector(frame)
        accumulator.update(detections)
        pose_estimator(frame)  # wrist extraction wired; aggregation is TODO

    trajectories = accumulator.trajectories()
    result = analyze_trajectories(meta, trajectories, None, cfg)
    result.warnings = warnings
    return result


def run(
    path: str,
    out_dir: Path,
    cfg: Config | None = None,
    annotate: bool = False,
) -> AnalysisResult:
    """Full analyze + write report (+ optional annotated video)."""
    result = analyze_video(path, cfg)
    report_stage.write_report(
        result.meta,
        result.timelines,
        result.metrics,
        out_dir,
        n_tracks=len(result.trajectories),
        count_estimate=result.count_estimate,
    )
    if annotate:
        from . import annotate as annotate_stage

        annotate_stage.annotate_video(
            result.meta, result.metrics, out_dir / "annotated.mp4"
        )
    return result
