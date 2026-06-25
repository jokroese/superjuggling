"""Ball-tracking pipeline orchestration.

Wires the stages together: ingest → detection → centre candidates → linking →
trajectories → report (+ optional annotated video). This pre-release pipeline
intentionally avoids event extraction and consistency scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
from .detection import YOLOPropDetector
from .linking import link_candidate_frames, write_links_csv
from .models import (
    CandidateFrame,
    TrackingSummary,
    Trajectory,
    VideoMeta,
)
from .runs import prepare_run_dir, write_run_sidecars
from .tracking import TrackAccumulator


@dataclass
class TrackingResult:
    meta: VideoMeta
    trajectories: list[Trajectory]
    summary: TrackingSummary
    count_estimate: int
    warnings: list[str]
    # Per-frame overlay data, populated only when the annotated-video output
    # is requested (see ``analyze_video(collect_overlay=True)``).
    frame_detections: list[Any] = field(default_factory=list)
    # Raw detector output before ByteTrack / smoothing. Populated only when
    # overlay data is collected; used by diagnostic annotation overlays.
    frame_raw_detections: list[Any] = field(default_factory=list)
    # Output directory for this analysis run.
    out_dir: Path = Path()
    candidate_frames: list[CandidateFrame] = field(default_factory=list)
    # Linker debug rows are intentionally typed Any here to keep TrackingResult
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


def summarize_tracking(
    *,
    candidate_frames: list[CandidateFrame],
    trajectories: list[Trajectory],
    count_estimate: int,
    backend: str,
) -> TrackingSummary:
    return report_stage.build_tracking_summary(
        candidate_frames=candidate_frames,
        trajectories=trajectories,
        count_estimate=count_estimate,
        backend=backend,
    )


def analyze_video(
    path: str,
    cfg: Config | None = None,
    collect_overlay: bool = False,
) -> TrackingResult:
    """Run the full ball-tracking pipeline. Requires the ``cv`` extra.

    When ``collect_overlay`` is set, per-frame tracked detections are retained
    so an annotated video can be rendered.
    """
    cfg = cfg or Config()

    meta = ingest_stage.probe(path)
    warnings = ingest_stage.validate(meta, cfg.ingest)

    prop_detector = YOLOPropDetector(cfg.detection)
    accumulator = TrackAccumulator(cfg.tracking, meta.fps, keep_frames=collect_overlay)

    # Stage 2–3: stream frames through detection + tracking.
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

    link_rows: list[Any] = []
    if cfg.linking.backend == "bytetrack":
        trajectories = accumulator.trajectories()
    else:
        trajectories, link_rows = link_candidate_frames(candidate_frames, cfg.linking)

    count = estimate_prop_count(trajectories, meta)
    summary = summarize_tracking(
        candidate_frames=candidate_frames,
        trajectories=trajectories,
        count_estimate=count,
        backend=cfg.linking.backend,
    )
    result = TrackingResult(
        meta=meta,
        trajectories=trajectories,
        summary=summary,
        count_estimate=count,
        warnings=warnings,
        candidate_frames=candidate_frames,
        link_debug_rows=link_rows,
    )
    if collect_overlay:
        result.frame_detections = accumulator.frame_detections()
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
) -> TrackingResult:
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
        result.candidate_frames,
        result.trajectories,
        result.summary,
        run_dir,
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
