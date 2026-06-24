"""Tests for the report writer (tech spec §6)."""

from __future__ import annotations

import json
from pathlib import Path

from superjuggling.metrics import compute_metrics
from superjuggling.models import DropEvent, Hand, ThrowEvent, Timelines, VideoMeta
from superjuggling.report import build_report, write_report


def _fixture() -> tuple[VideoMeta, Timelines]:
    meta = VideoMeta("run1.mp4", fps=60.0, total_frames=600, width=1920, height=1080)
    throws = [
        ThrowEvent(
            t=i * 0.25,
            track_id=i % 3,
            hand=Hand.LEFT if i % 2 else Hand.RIGHT,
            apex_x=600.0 if i % 2 else 900.0,
            apex_y=300.0,
            height_px=240.0,
        )
        for i in range(20)
    ]
    tl = Timelines(throws=throws, drops=[DropEvent(t=4.0, track_id=1)])
    return meta, tl


def test_build_report_matches_schema() -> None:
    meta, tl = _fixture()
    metrics = compute_metrics(tl, meta.duration_s)
    report = build_report(meta, tl, metrics, n_tracks=3, count_estimate=3)

    assert report["video"]["path"] == "run1.mp4"  # type: ignore[index]
    assert report["props"]["tracks"] == 3  # type: ignore[index]
    assert len(report["events"]["throws"]) == 20  # type: ignore[index]
    assert "rhythm" in report["metrics"]  # type: ignore[operator]
    # Events serialise hand as the string enum value.
    assert report["events"]["throws"][0]["hand"] in {"L", "R", "?"}  # type: ignore[index]


def test_write_report_creates_files(tmp_path: Path) -> None:
    meta, tl = _fixture()
    metrics = compute_metrics(tl, meta.duration_s)
    paths = write_report(meta, tl, metrics, tmp_path, n_tracks=3, count_estimate=3)

    assert paths["json"].exists()
    assert paths["markdown"].exists()

    loaded = json.loads(paths["json"].read_text())
    assert loaded["metrics"]["overall_consistency"] >= 0.0
    assert "# Superjuggling" in paths["markdown"].read_text()
