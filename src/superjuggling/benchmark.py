"""Benchmark helpers for candidate-generation experiments.

This module runs a small matrix of candidate-generation configurations against
the same trusted label slice and writes a compact comparison table.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import report as report_stage
from .candidates import write_candidates_csv
from .config import Config
from .evaluation import (
    evaluate_candidate_csv,
    write_candidate_evaluation_json,
)
from .pipeline import analyze_video


CandidateMethod = Literal["yolo", "heatmap", "hybrid"]
TrackingBackend = Literal["bytetrack", "centre", "ballistic"]


@dataclass(frozen=True)
class CandidateBenchmarkSpec:
    """One candidate-generation variant to run."""

    name: str
    candidate_source: CandidateMethod
    tracking: TrackingBackend
    yolo_confidence: float | None = None


@dataclass(frozen=True)
class CandidateBenchmarkRow:
    """One row in the candidate benchmark summary."""

    name: str
    candidate_source: str
    tracking: str
    yolo_confidence: float | None
    run_dir: str
    labelled_frames: int
    visible_labels: int
    predicted_candidates: int
    matches: int
    false_negatives: int
    false_positives: int
    recall: float
    precision: float
    f1: float
    mean_error_px: float | None
    median_error_px: float | None
    held_recall: float | None
    free_recall: float | None


def parse_methods(value: str) -> list[CandidateMethod]:
    """Parse comma-separated candidate methods."""
    methods: list[CandidateMethod] = []
    valid = {"yolo", "heatmap", "hybrid"}
    for raw in value.split(","):
        method = raw.strip().lower()
        if not method:
            continue
        if method not in valid:
            msg = f"unknown candidate method: {method}"
            raise ValueError(msg)
        methods.append(method)  # type: ignore[arg-type]
    if not methods:
        msg = "at least one candidate method is required"
        raise ValueError(msg)
    return methods


def parse_float_list(value: str) -> list[float]:
    """Parse comma-separated floats."""
    values: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        parsed = float(item)
        if not 0.0 <= parsed <= 1.0:
            msg = f"confidence values must be between 0 and 1: {parsed}"
            raise ValueError(msg)
        values.append(parsed)
    if not values:
        msg = "at least one confidence value is required"
        raise ValueError(msg)
    return values


def _confidence_slug(value: float) -> str:
    return f"{int(round(value * 1000)):03d}"


def build_candidate_benchmark_specs(
    *,
    methods: list[CandidateMethod],
    yolo_confidences: list[float],
    tracking: TrackingBackend,
) -> list[CandidateBenchmarkSpec]:
    """Build the benchmark matrix.

    YOLO gets a confidence sweep. Heatmap has no YOLO threshold. Hybrid uses
    the default YOLO confidence for now, so it remains a method comparison
    rather than a second sweep dimension.
    """
    specs: list[CandidateBenchmarkSpec] = []
    for method in methods:
        if method == "yolo":
            for confidence in yolo_confidences:
                specs.append(
                    CandidateBenchmarkSpec(
                        name=f"yolo-conf-{_confidence_slug(confidence)}",
                        candidate_source="yolo",
                        tracking=tracking,
                        yolo_confidence=confidence,
                    )
                )
        elif method == "heatmap":
            specs.append(
                CandidateBenchmarkSpec(
                    name="heatmap",
                    candidate_source="heatmap",
                    tracking=tracking,
                )
            )
        elif method == "hybrid":
            specs.append(
                CandidateBenchmarkSpec(
                    name="hybrid",
                    candidate_source="hybrid",
                    tracking=tracking,
                )
            )
    return specs


def _prepare_variant_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        msg = f"benchmark output already exists and is not empty: {path}"
        raise FileExistsError(msg)
    path.mkdir(parents=True, exist_ok=True)


def _config_for_spec(spec: CandidateBenchmarkSpec, *, require_model: bool) -> Config:
    cfg = Config()
    cfg.detection.allow_coco = not require_model
    cfg.detection.candidate_source = spec.candidate_source
    cfg.linking.backend = spec.tracking
    if spec.yolo_confidence is not None:
        cfg.detection.confidence = spec.yolo_confidence
    return cfg


def run_candidate_benchmark(
    *,
    video_path: Path,
    labels_path: Path,
    out_dir: Path,
    specs: list[CandidateBenchmarkSpec],
    radius_px: float,
    overwrite: bool = False,
    require_model: bool = False,
) -> list[CandidateBenchmarkRow]:
    """Run each candidate benchmark spec and return summary rows."""
    if radius_px <= 0:
        msg = "radius_px must be > 0"
        raise ValueError(msg)
    if not specs:
        msg = "at least one benchmark spec is required"
        raise ValueError(msg)

    _prepare_variant_dir(out_dir, overwrite=overwrite)

    rows: list[CandidateBenchmarkRow] = []
    for spec in specs:
        run_dir = out_dir / spec.name
        _prepare_variant_dir(run_dir, overwrite=overwrite)

        cfg = _config_for_spec(spec, require_model=require_model)
        result = analyze_video(str(video_path), cfg, collect_overlay=False)
        result.out_dir = run_dir

        report_stage.write_report(
            result.meta,
            result.candidate_frames,
            result.trajectories,
            result.summary,
            run_dir,
        )
        candidates_path = write_candidates_csv(
            result.candidate_frames,
            run_dir / "debug_candidates.csv",
        )
        evaluation = evaluate_candidate_csv(
            candidates_path=candidates_path,
            labels_path=labels_path,
            radius_px=radius_px,
        )
        write_candidate_evaluation_json(
            evaluation,
            run_dir / "candidate_evaluation.json",
        )

        summary = evaluation.summary
        rows.append(
            CandidateBenchmarkRow(
                name=spec.name,
                candidate_source=spec.candidate_source,
                tracking=spec.tracking,
                yolo_confidence=spec.yolo_confidence,
                run_dir=str(run_dir),
                labelled_frames=summary.labelled_frames,
                visible_labels=summary.visible_labels,
                predicted_candidates=summary.predicted_candidates,
                matches=summary.matches,
                false_negatives=summary.false_negatives,
                false_positives=summary.false_positives,
                recall=summary.recall,
                precision=summary.precision,
                f1=summary.f1,
                mean_error_px=summary.mean_error_px,
                median_error_px=summary.median_error_px,
                held_recall=summary.held_recall,
                free_recall=summary.free_recall,
            )
        )

    write_candidate_benchmark_summary(rows, out_dir)
    return rows


def write_candidate_benchmark_summary(
    rows: list[CandidateBenchmarkRow],
    out_dir: Path,
) -> dict[str, Path]:
    """Write CSV and markdown benchmark summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_summary.csv"
    md_path = out_dir / "benchmark_summary.md"

    fieldnames = list(CandidateBenchmarkRow.__dataclass_fields__)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    lines = [
        "# Candidate benchmark summary",
        "",
        "| Variant | Candidates | Matches | Recall | Precision | F1 | Mean error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item.f1, reverse=True):
        mean_error = "n/a" if row.mean_error_px is None else f"{row.mean_error_px:.3f}"
        lines.append(
            f"| {row.name} | {row.predicted_candidates} | {row.matches} | "
            f"{row.recall:.3f} | {row.precision:.3f} | {row.f1:.3f} | "
            f"{mean_error} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return {"csv": csv_path, "markdown": md_path}
