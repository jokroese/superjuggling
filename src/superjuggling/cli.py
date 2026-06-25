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
from .pipeline import run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="superjuggling",
        description="Juggling consistency tracker — turn a clip into metrics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze", help="Analyze a juggling clip and write a report."
    )
    analyze.add_argument("video", help="Path to the input video clip.")
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
        help="Also render an annotated output video.",
    )
    analyze.add_argument(
        "--debug-overlays",
        action="store_true",
        help="Render diagnostic overlays in the annotated video. Implies "
        "--annotate.",
    )
    analyze.add_argument(
        "--allow-coco",
        action="store_true",
        help="Smoke test only: use COCO 'sports ball' weights when no "
        "fine-tuned prop model is set. Detections are unreliable (§4.2).",
    )
    return parser


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = Config()
    cfg.detection.allow_coco = args.allow_coco
    annotate = bool(args.annotate or args.debug_overlays)
    command = " ".join(shlex.quote(part) for part in sys.argv)
    try:
        result = run(
            args.video,
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
        if args.debug_overlays and not args.allow_coco:
            print(
                "hint: for a plumbing-only diagnostic run without fine-tuned "
                "weights, add --allow-coco.",
                file=sys.stderr,
            )
        return 1
    except FileNotFoundError:
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    print(
        f"Analyzed {result.meta.path} "
        f"({result.meta.duration_s:.1f}s, ~{result.count_estimate} props)"
    )
    print(f"Overall consistency: {result.metrics.overall_consistency:.0f}/100")
    print(f"Run written to {result.out_dir}/")
    if annotate:
        print(f"Annotated video written to {result.out_dir / 'annotated.mp4'}")
    if args.debug_overlays:
        print("Diagnostic overlays enabled")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return cmd_analyze(args)
    parser.print_help()
    return 1
