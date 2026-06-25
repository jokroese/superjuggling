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
from .candidates import (
    detections_to_candidates,
    filter_candidates,
    fuse_candidates,
    make_candidate_frame,
    write_candidates_csv,
)
from .config import Config
from .detection import YOLOPoseEstimator, YOLOPropDetector
from .linking import link_candidate_frames, write_links_csv
from .metrics import MetricsReport, compute_metrics
from .models import (
    CandidateFrame,
    FlightSegment,
    Hand,
    HandTrack,
    Timelines,
    Trajectory,
    VideoMeta,
)
from .runs import prepare_run_dir, write_run_sidecars
from .segments import segment_trajectories_into_flights, write_segments_csv
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
    # Output directory for this analysis run.
    out_dir: Path = Path()
    # Centre-candidate architecture artefacts.
    candidate_frames: list[CandidateFrame] = field(default_factory=list)
    flight_segments: list[FlightSegment] = field(default_factory=list)
    # Linker debug rows are intentionally typed Any here to keep AnalysisResult
    # independent from linker internals in the public surface.
    link_debug_rows: list[Any] = field(default_factory=list)


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
    segments: list[FlightSegment] | None = None,
) -> AnalysisResult:
    """Run the pure analytical half of the pipeline (stages 4–5).

    This is the seam used by tests: feed synthetic trajectories straight in,
    no video required.
    """
    cfg = cfg or Config()
    timelines = events_stage.extract_events(
        trajectories, hands, meta, cfg.events, segments
    )
    metrics = compute_metrics(timelines, meta.duration_s, cfg.weights)
    count = estimate_prop_count(trajectories, meta)
    return AnalysisResult(
        meta=meta,
        trajectories=trajectories,
        timelines=timelines,
        metrics=metrics,
        count_estimate=count,
        warnings=[],
        flight_segments=segments or [],
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

    if cfg.detection.mode in {"sliced", "temporal"}:
        msg = f"detection mode {cfg.detection.mode!r} is not implemented yet"
        raise NotImplementedError(msg)

    prop_detector = YOLOPropDetector(cfg.detection)
    pose_estimator = YOLOPoseEstimator(cfg.detection)
    accumulator = TrackAccumulator(cfg.tracking, meta.fps, keep_frames=collect_overlay)

    # Stage 2–3: stream frames through detection + tracking.
    keypoints: list[object] = []
    raw_detections: list[object] = []
    candidate_frames: list[CandidateFrame] = []
    for frame_idx, frame in enumerate(ingest_stage.frames(path)):
        detections = prop_detector(frame)
        candidates = fuse_candidates(
            filter_candidates(
                detections_to_candidates(detections, frame_idx, meta.fps),
                cfg.candidates,
            ),
            cfg.candidates.fusion_distance_px,
        )
        candidate_frames.append(make_candidate_frame(frame_idx, meta.fps, candidates))

        if collect_overlay:
            raw_detections.append(detections)

        if cfg.linking.backend == "bytetrack":
            accumulator.update(detections)

        kp = pose_estimator(frame)  # wrist keypoints; aggregation into hands is TODO
        if collect_overlay:
            keypoints.append(kp)

    link_rows: list[Any] = []
    if cfg.linking.backend == "bytetrack":
        trajectories = accumulator.trajectories()
    else:
        trajectories, link_rows = link_candidate_frames(candidate_frames, cfg.linking)

    segments = segment_trajectories_into_flights(trajectories, cfg.segments)
    result = analyze_trajectories(meta, trajectories, None, cfg, segments=segments)
    result.warnings = warnings
    result.candidate_frames = candidate_frames
    result.link_debug_rows = link_rows
    if collect_overlay:
        result.frame_detections = accumulator.frame_detections()
        result.frame_keypoints = keypoints
        result.frame_raw_detections = raw_detections
    return result


def run(
    path: str,
    out_dir: Path | None = None,
    cfg: Config | None = None,
    annotate: bool = False,
    debug_overlays: bool = False,
    runs_dir: Path = Path("runs"),
    overwrite: bool = False,
    command: str | None = None,
) -> AnalysisResult:
    """Full analyze + write report (+ optional annotated video)."""
    cfg = cfg or Config()
    run_dir, run_id, input_sha256 = prepare_run_dir(
        path=Path(path),
        out_dir=out_dir,
        runs_dir=runs_dir,
        overwrite=overwrite,
    )

    collect_overlay = annotate or debug_overlays
    result = analyze_video(path, cfg, collect_overlay=collect_overlay)
    result.out_dir = run_dir

    report_stage.write_report(
        result.meta,
        result.timelines,
        result.metrics,
        run_dir,
        n_tracks=len(result.trajectories),
        count_estimate=result.count_estimate,
    )

    annotated_path: Path | None = None
    if collect_overlay:
        from . import annotate as annotate_stage

        annotated_path = run_dir / "annotated.mp4"
        annotate_stage.annotate_video(
            result, annotated_path, cfg, debug_overlays=debug_overlays
        )

    if debug_overlays:
        write_candidates_csv(result.candidate_frames, run_dir / "debug_candidates.csv")
        write_segments_csv(result.flight_segments, run_dir / "debug_segments.csv")
        if result.link_debug_rows:
            write_links_csv(result.link_debug_rows, run_dir / "debug_links.csv")

    write_run_sidecars(
        run_dir=run_dir,
        run_id=run_id,
        input_path=Path(path),
        input_sha256=input_sha256,
        cfg=cfg,
        annotate=annotate,
        debug_overlays=debug_overlays,
        command=command,
        annotated_path=annotated_path,
    )
    return result
