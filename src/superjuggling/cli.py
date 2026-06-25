"""Command-line interface (tech spec §10).

superjuggling analyze run1.mp4 --out report/
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ._optional import MissingCVDependency
from .benchmark import (
    build_candidate_benchmark_specs,
    parse_float_list,
    parse_methods,
    run_candidate_benchmark,
)
from .config import Config
from .evaluation import (
    evaluate_candidate_csv,
    write_candidate_evaluation_json,
)
from .labels import (
    convert_cvat_video_to_csv,
    filter_labels_by_frame_range,
    read_cvat_video_labels,
)
from .pipeline import run


_VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


def _resolve_video_path(value: str, videos_dir: Path = Path("data/videos")) -> Path:
    """Resolve ergonomic video arguments.

    Accepted forms:

    - exact path: ``data/videos/juggling-short.mp4``
    - file in ``data/videos``: ``juggling-short.mp4``
    - stem in ``data/videos``: ``juggling-short``
    """
    path = Path(value)
    if path.exists():
        return path

    candidates: list[Path] = []
    if path.suffix:
        candidates.append(videos_dir / path.name)
    else:
        candidates.extend(videos_dir / f"{path.name}{ext}" for ext in _VIDEO_EXTENSIONS)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="superjuggling",
        description="Juggling ball tracker — turn a clip into trajectories.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze", help="Analyze a juggling clip and write a report."
    )
    analyze.add_argument(
        "video",
        help=(
            "Input video path, or a filename/stem in data/videos/ "
            "(e.g. juggling-short)."
        ),
    )
    analyze.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory for this run. Defaults to "
            "runs/<timestamp>_<video-stem>_<input-hash>/."
        ),
    )
    analyze.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory for auto-created run outputs (default: runs/).",
    )
    analyze.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing --out directory.",
    )
    analyze.add_argument(
        "--annotate",
        action="store_true",
        help="Render an annotated output video. This is already the default.",
    )
    analyze.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip rendering annotated.mp4 and only write report artefacts.",
    )
    analyze.add_argument(
        "--debug-overlays",
        action="store_true",
        help="Render diagnostic overlays in the annotated video. Implies --annotate.",
    )
    analyze.add_argument(
        "--allow-coco",
        action="store_true",
        help=(
            "Allow COCO 'sports ball' fallback when no fine-tuned prop model "
            "is set. This is already the default for the CLI."
        ),
    )
    analyze.add_argument(
        "--require-model",
        action="store_true",
        help="Disable COCO fallback and require fine-tuned prop weights.",
    )
    analyze.add_argument(
        "--tracking",
        choices=["bytetrack", "centre", "ballistic"],
        default="ballistic",
        help="Tracking/linking backend (default: ballistic).",
    )
    analyze.add_argument(
        "--candidate-source",
        choices=["yolo", "heatmap", "hybrid"],
        default="yolo",
        help=(
            "Candidate source: YOLO boxes, multi-frame motion heatmap, "
            "or fused hybrid candidates (default: yolo)."
        ),
    )
    analyze.add_argument(
        "--yolo-confidence",
        type=float,
        default=None,
        help="YOLO detector confidence threshold (default: 0.05). Lower values increase recall.",
    )
    analyze.add_argument(
        "--candidate-min-score",
        type=float,
        default=None,
        help="Minimum centre-candidate score before fusion/linking.",
    )
    analyze.add_argument(
        "--heatmap-window-radius",
        type=int,
        default=None,
        help="Neighbour frames on each side for heatmap detection.",
    )
    analyze.add_argument(
        "--heatmap-min-score",
        type=float,
        default=None,
        help="Minimum normalised heatmap score for peak candidates.",
    )

    convert = sub.add_parser(
        "convert-cvat",
        help="Convert a CVAT for video export into project-native label CSV.",
    )
    convert.add_argument(
        "cvat_export",
        type=Path,
        help="Path to CVAT for video XML or ZIP export.",
    )
    convert.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path, e.g. data/labels/juggling-short.csv.",
    )
    convert.add_argument(
        "--label",
        default="ball",
        help="CVAT label name to convert (default: ball).",
    )
    convert.add_argument(
        "--held-attribute",
        default="held",
        help="CVAT mutable attribute name for held/in-hand state (default: held).",
    )
    convert.add_argument(
        "--frame-start",
        type=int,
        default=None,
        help="First trusted frame to include, inclusive.",
    )
    convert.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="Last trusted frame to include, inclusive. For your first slice use 71.",
    )

    evaluate = sub.add_parser(
        "evaluate-candidates",
        help="Evaluate debug candidate detections against ground-truth labels.",
    )
    evaluate.add_argument(
        "candidates_csv",
        type=Path,
        help="Path to debug_candidates.csv from an analyze --debug-overlays run.",
    )
    evaluate.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Project-native ground-truth labels CSV.",
    )
    evaluate.add_argument(
        "--radius-px",
        type=float,
        default=15.0,
        help="Maximum centre distance for a match in pixels (default: 15).",
    )
    evaluate.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for machine-readable candidate_evaluation.json.",
    )

    benchmark = sub.add_parser(
        "benchmark-candidates",
        help="Run and evaluate multiple candidate-generation variants.",
    )
    benchmark.add_argument(
        "video",
        help=(
            "Input video path, or a filename/stem in data/videos/ "
            "(e.g. jobi-juggling2-0000-0071)."
        ),
    )
    benchmark.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Project-native ground-truth labels CSV.",
    )
    benchmark.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Benchmark output directory.",
    )
    benchmark.add_argument(
        "--methods",
        default="yolo,heatmap,hybrid",
        help="Comma-separated candidate methods: yolo,heatmap,hybrid.",
    )
    benchmark.add_argument(
        "--yolo-confidences",
        default="0.20,0.15,0.10,0.07,0.05",
        help="Comma-separated YOLO confidence sweep values.",
    )
    benchmark.add_argument(
        "--tracking",
        choices=["bytetrack", "centre", "ballistic"],
        default="ballistic",
        help="Tracking/linking backend for benchmark runs.",
    )
    benchmark.add_argument(
        "--radius-px",
        type=float,
        default=15.0,
        help="Maximum centre distance for candidate evaluation.",
    )
    benchmark.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing non-empty benchmark directories.",
    )
    benchmark.add_argument(
        "--require-model",
        action="store_true",
        help="Disable COCO fallback and require fine-tuned prop weights.",
    )
    return parser


def cmd_convert_cvat(args: argparse.Namespace) -> int:
    try:
        labels = read_cvat_video_labels(
            args.cvat_export,
            label_name=args.label,
            held_attribute=args.held_attribute,
        )
        labels = filter_labels_by_frame_range(
            labels,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
        convert_cvat_video_to_csv(
            args.cvat_export,
            args.out,
            label_name=args.label,
            held_attribute=args.held_attribute,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    visible = sum(1 for label in labels if label.visible)
    held = sum(1 for label in labels if label.held is True)
    unknown_held = sum(1 for label in labels if label.held is None)
    tracks = sorted({label.ball_id for label in labels})
    frames = sorted({label.frame_index for label in labels})

    print(f"Converted {len(labels)} labels from {args.cvat_export}")
    print(f"Tracks: {len(tracks)} ({', '.join(str(track) for track in tracks)})")
    if frames:
        print(f"Frames: {frames[0]}..{frames[-1]} ({len(frames)} labelled frames)")
    print(f"Visible labels: {visible}")
    print(f"Held labels: {held}")
    if unknown_held:
        print(f"Unknown held labels: {unknown_held}")
    print(f"Wrote {args.out}")
    return 0


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def cmd_evaluate_candidates(args: argparse.Namespace) -> int:
    try:
        evaluation = evaluate_candidate_csv(
            candidates_path=args.candidates_csv,
            labels_path=args.labels,
            radius_px=args.radius_px,
        )
        if args.out is not None:
            write_candidate_evaluation_json(evaluation, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = evaluation.summary
    print(f"Candidate evaluation @ {summary.radius_px:g}px")
    print(f"Labelled frames: {summary.labelled_frames}")
    print(f"Visible labels: {summary.visible_labels}")
    print(f"Predicted candidates on labelled frames: {summary.predicted_candidates}")
    print(f"Matches: {summary.matches}")
    print(f"False negatives: {summary.false_negatives}")
    print(f"False positives: {summary.false_positives}")
    print(f"Recall: {_fmt_rate(summary.recall)}")
    print(f"Precision: {_fmt_rate(summary.precision)}")
    print(f"F1: {_fmt_rate(summary.f1)}")
    print(
        "Mean matched error: "
        f"{'n/a' if summary.mean_error_px is None else f'{summary.mean_error_px:.3f}px'}"
    )
    print(
        "Median matched error: "
        f"{'n/a' if summary.median_error_px is None else f'{summary.median_error_px:.3f}px'}"
    )
    print(f"Held recall: {_fmt_rate(summary.held_recall)}")
    print(f"Free-flight recall: {_fmt_rate(summary.free_recall)}")
    if evaluation.source_counts:
        print("Candidate sources:")
        for source, count in sorted(evaluation.source_counts.items()):
            print(f"  {source}: {count}")
    if args.out is not None:
        print(f"Wrote {args.out}")
    return 0


def cmd_benchmark_candidates(args: argparse.Namespace) -> int:
    try:
        methods = parse_methods(args.methods)
        yolo_confidences = parse_float_list(args.yolo_confidences)
        specs = build_candidate_benchmark_specs(
            methods=methods,
            yolo_confidences=yolo_confidences,
            tracking=args.tracking,
        )
        rows = run_candidate_benchmark(
            video_path=_resolve_video_path(args.video),
            labels_path=args.labels,
            out_dir=args.out,
            specs=specs,
            radius_px=args.radius_px,
            overwrite=args.overwrite,
            require_model=args.require_model,
        )
    except MissingCVDependency as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Benchmark written to {args.out}/")
    print("")
    print("Variant                 Cand  Match  Recall  Prec    F1")
    print("----------------------  ----  -----  ------  ------  ------")
    for row in sorted(rows, key=lambda item: item.f1, reverse=True):
        print(
            f"{row.name:<22}  "
            f"{row.predicted_candidates:>4}  "
            f"{row.matches:>5}  "
            f"{row.recall:>6.3f}  "
            f"{row.precision:>6.3f}  "
            f"{row.f1:>6.3f}"
        )

    if rows:
        best_recall = max(rows, key=lambda item: item.recall)
        best_f1 = max(rows, key=lambda item: item.f1)
        print("")
        print(f"Best recall: {best_recall.name} ({best_recall.recall:.3f})")
        print(f"Best F1: {best_f1.name} ({best_f1.f1:.3f})")
    print("")
    print(f"Summary CSV: {args.out / 'benchmark_summary.csv'}")
    print(f"Summary Markdown: {args.out / 'benchmark_summary.md'}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = Config()
    # CLI default: make the first run useful without forcing users to configure
    # a detector first. Library/default Config can remain stricter later if
    # desired; the command-line tool optimises for a smooth local workflow.
    cfg.detection.allow_coco = not args.require_model
    cfg.linking.backend = args.tracking
    cfg.detection.candidate_source = args.candidate_source
    if args.yolo_confidence is not None:
        if not 0.0 <= args.yolo_confidence <= 1.0:
            print("error: --yolo-confidence must be between 0 and 1", file=sys.stderr)
            return 1
        cfg.detection.confidence = args.yolo_confidence
    if args.candidate_min_score is not None:
        cfg.candidates.min_score = args.candidate_min_score
    if args.heatmap_window_radius is not None:
        cfg.heatmap.window_radius = args.heatmap_window_radius
    if args.heatmap_min_score is not None:
        cfg.heatmap.min_score = args.heatmap_min_score

    video_path = _resolve_video_path(args.video)

    # Annotated video is the useful default for this project: it is both an
    # output artefact and the fastest way to understand detection/tracking
    # failures. Users can opt out for faster report-only runs.
    annotate = not args.no_annotate or args.annotate or args.debug_overlays

    command = " ".join(shlex.quote(part) for part in sys.argv)
    try:
        result = run(
            str(video_path),
            args.out,
            cfg,
            annotate=annotate,
            debug_overlays=args.debug_overlays,
            runs_dir=args.runs_dir,
            overwrite=args.overwrite,
            command=command,
        )
    except MissingCVDependency as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: choose a different --out or pass --overwrite.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.require_model:
            print(
                "hint: remove --require-model to use the default COCO fallback.",
                file=sys.stderr,
            )
        return 1
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"error: video not found: {args.video}", file=sys.stderr)
        if not Path(args.video).exists():
            print(
                "hint: pass a full path, or place the clip in data/videos/ "
                "and use its filename or stem.",
                file=sys.stderr,
            )
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    print(
        f"Analyzed {result.meta.path} "
        f"({result.meta.duration_s:.1f}s, ~{result.count_estimate} props)"
    )
    print(f"Trajectories: {result.summary.trajectories}")
    print(f"Candidate detections: {result.summary.candidates}")
    print(f"Run written to {result.out_dir}/")
    if annotate:
        print(f"Annotated video written to {result.out_dir / 'annotated.mp4'}")
    if cfg.detection.allow_coco and cfg.detection.model_path is None:
        print(
            "note: used COCO sports-ball fallback; fine-tuned prop weights are recommended"
        )
    print(f"YOLO confidence: {cfg.detection.confidence}")
    print(f"Tracking backend: {cfg.linking.backend}")
    print(f"Candidate source: {cfg.detection.candidate_source}")
    if args.debug_overlays:
        print("Diagnostic overlays enabled")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "convert-cvat":
        return cmd_convert_cvat(args)
    if args.command == "evaluate-candidates":
        return cmd_evaluate_candidates(args)
    if args.command == "benchmark-candidates":
        return cmd_benchmark_candidates(args)
    parser.print_help()
    return 1
