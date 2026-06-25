"""Tests for the tracking report writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from superjuggling.models import CandidateFrame, TrackingSummary, Trajectory, VideoMeta
from superjuggling.report import build_report, write_report


def _fixture() -> tuple[
    VideoMeta, list[CandidateFrame], list[Trajectory], TrackingSummary
]:
    meta = VideoMeta("run1.mp4", fps=60.0, total_frames=600, width=1920, height=1080)
    frames = [
        CandidateFrame(frame_index=i, t=i / 60.0, candidates=[]) for i in range(10)
    ]
    trajectories = [
        Trajectory(
            track_id=1,
            t=np.asarray([0.0, 0.1, 0.2]),
            x=np.asarray([100.0, 110.0, 120.0]),
            y=np.asarray([200.0, 190.0, 210.0]),
            confidence=np.asarray([0.8, 0.9, 0.85]),
        )
    ]
    summary = TrackingSummary(
        frames=10,
        candidates=0,
        trajectories=1,
        trajectory_points=3,
        estimated_props=1,
        backend="bytetrack",
        mean_track_points=3.0,
        median_track_points=3.0,
    )
    return meta, frames, trajectories, summary


def test_build_tracking_report_matches_schema() -> None:
    meta, frames, trajectories, summary = _fixture()
    report = build_report(
        meta=meta,
        candidate_frames=frames,
        trajectories=trajectories,
        summary=summary,
    )

    assert report["video"]["path"] == "run1.mp4"  # type: ignore[index]
    assert report["tracking"]["trajectories"] == 1  # type: ignore[index]
    assert len(report["trajectories"]) == 1  # type: ignore[arg-type]


def test_write_report_creates_files(tmp_path: Path) -> None:
    meta, frames, trajectories, summary = _fixture()
    paths = write_report(meta, frames, trajectories, summary, tmp_path)

    assert paths["json"].exists()
    assert paths["markdown"].exists()

    loaded = json.loads(paths["json"].read_text())
    assert loaded["tracking"]["backend"] == "bytetrack"
    assert "# Superjuggling — Ball Tracking Report" in paths["markdown"].read_text()
