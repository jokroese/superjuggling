"""Tests for ergonomic CLI path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from superjuggling.cli import _resolve_video_path, main
from superjuggling.labels import GroundTruthLabel, write_ground_truth_csv

from test_labels import CVAT_XML


def test_resolve_existing_explicit_video_path(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    assert _resolve_video_path(str(video)) == video


def test_resolve_filename_in_data_videos(tmp_path: Path) -> None:
    videos_dir = tmp_path / "data" / "videos"
    videos_dir.mkdir(parents=True)
    video = videos_dir / "clip.mp4"
    video.write_bytes(b"video")
    assert _resolve_video_path("clip.mp4", videos_dir=videos_dir) == video


def test_resolve_stem_in_data_videos(tmp_path: Path) -> None:
    videos_dir = tmp_path / "data" / "videos"
    videos_dir.mkdir(parents=True)
    video = videos_dir / "clip.mp4"
    video.write_bytes(b"video")
    assert _resolve_video_path("clip", videos_dir=videos_dir) == video


def test_unresolved_path_is_returned_for_later_error_handling(tmp_path: Path) -> None:
    videos_dir = tmp_path / "data" / "videos"
    videos_dir.mkdir(parents=True)
    assert _resolve_video_path("missing", videos_dir=videos_dir) == Path("missing")


def test_convert_cvat_cli_supports_frame_slice(
    tmp_path: Path,
    capsys: Any,
) -> None:
    cvat_xml = tmp_path / "annotations.xml"
    cvat_xml.write_text(CVAT_XML)
    out = tmp_path / "labels.csv"

    status = main(
        [
            "convert-cvat",
            str(cvat_xml),
            "--out",
            str(out),
            "--frame-start",
            "0",
            "--frame-end",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert out.exists()
    assert "Frames: 0..0 (1 labelled frames)" in captured.out


def test_evaluate_candidates_cli_writes_report(
    tmp_path: Path,
    capsys: Any,
) -> None:
    labels_path = write_ground_truth_csv(
        [
            GroundTruthLabel(
                frame_index=0,
                ball_id=1,
                x=10.0,
                y=10.0,
                w=10.0,
                h=10.0,
                visible=True,
                held=False,
                keyframe=True,
            )
        ],
        tmp_path / "labels.csv",
    )
    candidates_path = tmp_path / "debug_candidates.csv"
    candidates_path.write_text(
        "frame_index,t,x,y,score,radius_px,source,class_id\n"
        "0,0.0,12.0,10.0,0.9,10.0,heatmap,\n"
        "1,0.016,12.0,10.0,0.9,10.0,ignored,\n"
    )
    out_path = tmp_path / "candidate_evaluation.json"

    status = main(
        [
            "evaluate-candidates",
            str(candidates_path),
            "--labels",
            str(labels_path),
            "--radius-px",
            "5",
            "--out",
            str(out_path),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert out_path.exists()
    assert "Candidate evaluation @ 5px" in captured.out
    assert "Labelled frames: 1" in captured.out
    assert "Visible labels: 1" in captured.out
    assert "Predicted candidates on labelled frames: 1" in captured.out
    assert "Recall: 1.000" in captured.out
