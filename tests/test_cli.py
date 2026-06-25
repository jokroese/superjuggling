"""Tests for ergonomic CLI path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from superjuggling.cli import _resolve_video_path, main

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
