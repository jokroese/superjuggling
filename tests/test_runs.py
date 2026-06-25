"""Tests for per-run output directories and sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superjuggling.config import Config
from superjuggling.runs import (
    config_to_dict,
    prepare_run_dir,
    sha256_file,
    write_run_sidecars,
)


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"superjuggling")

    assert (
        sha256_file(path)
        == "15107ce8d32df2f9790d9b957819d42a78fbc7e64d179318fe75ef6a8028a885"
    )


def test_prepare_run_dir_auto_names_from_video_and_hash(tmp_path: Path) -> None:
    video = tmp_path / "juggling short.mp4"
    video.write_bytes(b"video")

    run_dir, run_id, digest = prepare_run_dir(
        path=video,
        out_dir=None,
        runs_dir=tmp_path / "runs",
        overwrite=False,
    )

    assert run_dir.exists()
    assert run_dir.name == run_id
    assert "juggling-short" in run_id
    assert digest[:7] in run_id


def test_prepare_run_dir_refuses_non_empty_explicit_out(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "existing"
    out.mkdir()
    (out / "metrics.json").write_text("{}")

    with pytest.raises(FileExistsError):
        prepare_run_dir(path=video, out_dir=out, runs_dir=tmp_path, overwrite=False)


def test_config_to_dict_contains_nested_sections() -> None:
    cfg = Config()
    data = config_to_dict(cfg)

    assert data["ingest"]["min_fps"] == 30.0
    assert "detection" in data
    assert "tracking" in data


def test_write_run_sidecars(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    run_dir = tmp_path / "runs" / "manual"
    run_dir.mkdir(parents=True)

    paths = write_run_sidecars(
        run_dir=run_dir,
        run_id="manual",
        input_path=video,
        input_sha256=sha256_file(video),
        cfg=Config(),
        annotate=True,
        debug_overlays=True,
        command="superjuggling analyze data/videos/clip.mp4",
        annotated_path=run_dir / "annotated.mp4",
    )

    assert paths["config"].exists()
    assert paths["command"].read_text().startswith("superjuggling analyze")

    metadata = json.loads(paths["run"].read_text())
    assert metadata["run_id"] == "manual"
    assert metadata["outputs"]["metrics"] == "metrics.json"
    assert metadata["outputs"]["annotated_video"] == "annotated.mp4"
