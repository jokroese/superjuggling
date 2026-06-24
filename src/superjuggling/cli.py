"""Command-line interface (tech spec §10).

superjuggling analyze run1.mp4 --out report/
"""

from __future__ import annotations

import argparse
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
        default=Path("report"),
        help="Output directory for metrics.json / summary.md (default: report/).",
    )
    analyze.add_argument(
        "--annotate",
        action="store_true",
        help="Also render an annotated output video.",
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
    try:
        result = run(args.video, args.out, cfg, annotate=args.annotate)
    except MissingCVDependency as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
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
    print(f"Report written to {args.out}/")
    if args.annotate:
        print(f"Annotated video written to {args.out / 'annotated.mp4'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return cmd_analyze(args)
    parser.print_help()
    return 1
