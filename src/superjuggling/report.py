"""Tracking report writer.

Writes a machine-readable ``tracking.json`` and a short ``summary.md`` focused
only on ball candidates and trajectories.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .models import CandidateFrame, TrackingSummary, Trajectory, VideoMeta


def build_tracking_summary(
    *,
    candidate_frames: list[CandidateFrame],
    trajectories: list[Trajectory],
    count_estimate: int,
    backend: str,
) -> TrackingSummary:
    lengths = np.asarray([len(tr.t) for tr in trajectories], dtype=np.float64)
    return TrackingSummary(
        frames=len(candidate_frames),
        candidates=sum(len(frame.candidates) for frame in candidate_frames),
        trajectories=len(trajectories),
        trajectory_points=int(sum(len(tr.t) for tr in trajectories)),
        estimated_props=count_estimate,
        backend=backend,
        mean_track_points=round(float(np.mean(lengths)), 2) if len(lengths) else 0.0,
        median_track_points=round(float(np.median(lengths)), 2)
        if len(lengths)
        else 0.0,
    )


def _trajectory_rows(trajectories: list[Trajectory]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for traj in trajectories:
        confidence = traj.confidence
        rows.append(
            {
                "track_id": traj.track_id,
                "points": len(traj.t),
                "t_start": float(traj.t[0]) if len(traj.t) else None,
                "t_end": float(traj.t[-1]) if len(traj.t) else None,
                "x_start": float(traj.x[0]) if len(traj.x) else None,
                "y_start": float(traj.y[0]) if len(traj.y) else None,
                "x_end": float(traj.x[-1]) if len(traj.x) else None,
                "y_end": float(traj.y[-1]) if len(traj.y) else None,
                "mean_confidence": (
                    None
                    if confidence is None or not len(confidence)
                    else round(float(np.mean(confidence)), 4)
                ),
            }
        )
    return rows


def build_report(
    *,
    meta: VideoMeta,
    candidate_frames: list[CandidateFrame],
    trajectories: list[Trajectory],
    summary: TrackingSummary,
) -> dict[str, object]:
    """Assemble the tracking report."""
    return {
        "video": {
            "path": meta.path,
            "fps": meta.fps,
            "duration_s": round(meta.duration_s, 2),
            "resolution": [meta.width, meta.height],
        },
        "tracking": asdict(summary),
        "candidates": {
            "frames": len(candidate_frames),
            "total": sum(len(frame.candidates) for frame in candidate_frames),
        },
        "trajectories": _trajectory_rows(trajectories),
    }


def write_json(report: dict[str, object], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tracking.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def write_markdown(report: dict[str, object], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.md"
    path.write_text(_render_markdown(report))
    return path


def _render_markdown(report: dict[str, object]) -> str:
    video = report["video"]
    tracking = report["tracking"]
    assert isinstance(video, dict)
    assert isinstance(tracking, dict)

    lines = [
        "# Superjuggling — Ball Tracking Report",
        "",
        f"**Clip:** `{video['path']}`  ",
        f"**Duration:** {video['duration_s']} s · "
        f"**fps:** {video['fps']} · "
        f"**resolution:** {video['resolution'][0]}×{video['resolution'][1]}  ",
        "",
        "## Tracking summary",
        "",
        f"- Backend: **{tracking['backend']}**",
        f"- Frames analysed: {tracking['frames']}",
        f"- Centre candidates: {tracking['candidates']}",
        f"- Trajectories: {tracking['trajectories']}",
        f"- Trajectory points: {tracking['trajectory_points']}",
        f"- Estimated props: {tracking['estimated_props']}",
        f"- Mean track length: {tracking['mean_track_points']} points",
        f"- Median track length: {tracking['median_track_points']} points",
        "",
        "_This pre-release report is intentionally limited to ball-tracking diagnostics._",
    ]
    return "\n".join(lines)


def write_report(
    meta: VideoMeta,
    candidate_frames: list[CandidateFrame],
    trajectories: list[Trajectory],
    summary: TrackingSummary,
    out_dir: Path,
) -> dict[str, Path]:
    """Write JSON + markdown tracking reports and return the paths."""
    report = build_report(
        meta=meta,
        candidate_frames=candidate_frames,
        trajectories=trajectories,
        summary=summary,
    )
    return {
        "json": write_json(report, out_dir),
        "markdown": write_markdown(report, out_dir),
    }
