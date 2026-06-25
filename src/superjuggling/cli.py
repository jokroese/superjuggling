"""Command-line interface (tech spec §10).

superjuggling analyze run1.mp4 --out report/
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ._optional import MissingCVDependency
from .config import Config
from .labels import convert_cvat_video_to_csv, read_cvat_video_labels
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
        default="hybrid",
        help=(
            "Candidate source: YOLO boxes, multi-frame motion heatmap, "
            "or fused hybrid candidates (default: hybrid)."
        ),
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
    return parser


def cmd_convert_cvat(args: argparse.Namespace) -> int:
    try:
        labels = read_cvat_video_labels(
            args.cvat_export,
            label_name=args.label,
            held_attribute=args.held_attribute,
        )
        convert_cvat_video_to_csv(
            args.cvat_export,
            args.out,
            label_name=args.label,
            held_attribute=args.held_attribute,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    visible = sum(1 for label in labels if label.visible)
    held = sum(1 for label in labels if label.held is True)
    unknown_held = sum(1 for label in labels if label.held is None)
    tracks = sorted({label.ball_id for label in labels})

    print(f"Converted {len(labels)} labels from {args.cvat_export}")
    print(f"Tracks: {len(tracks)} ({', '.join(str(track) for track in tracks)})")
    print(f"Visible labels: {visible}")
    print(f"Held labels: {held}")
    if unknown_held:
        print(f"Unknown held labels: {unknown_held}")
    print(f"Wrote {args.out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = Config()
    # CLI default: make the first run useful without forcing users to configure
    # a detector first. Library/default Config can remain stricter later if
    # desired; the command-line tool optimises for a smooth local workflow.
    cfg.detection.allow_coco = not args.require_model
    cfg.linking.backend = args.tracking
    cfg.detection.candidate_source = args.candidate_source
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
    parser.print_help()
    return 1
