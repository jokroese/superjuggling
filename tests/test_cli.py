"""Tests for ergonomic CLI path resolution."""

from __future__ import annotations

from pathlib import Path

from superjuggling.cli import _resolve_video_path


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
