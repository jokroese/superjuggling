"""Run-directory and provenance helpers.

Generated reports, annotated videos and debug artefacts are analysis outputs,
not source files. Each CLI execution gets its own run directory with sidecars
that record the effective config and enough provenance to understand how the
artefacts were produced.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import Config


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of ``path`` without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    """Keep run IDs path-friendly without hiding the source video's name."""
    chars: list[str] = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        elif char in {" ", "."}:
            chars.append("-")
    slug = "".join(chars).strip("-_")
    return slug or "video"


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")


def _run_id(path: Path, input_sha256: str) -> str:
    return f"{_timestamp()}_{_slug(path.stem)}_{input_sha256[:7]}"


def prepare_run_dir(
    path: Path,
    out_dir: Path | None,
    runs_dir: Path,
    overwrite: bool,
) -> tuple[Path, str, str]:
    """Resolve and create the output directory for an analysis run.

    If ``out_dir`` is omitted, create a fresh directory under ``runs_dir`` using
    timestamp + video stem + short input hash. Explicit ``--out`` directories
    are protected against accidental overwrite unless ``--overwrite`` is set.
    """
    input_sha256 = sha256_file(path)

    if out_dir is not None:
        run_dir = out_dir
        run_id = run_dir.name
        if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
            msg = f"output directory already exists and is not empty: {run_dir}"
            raise FileExistsError(msg)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, run_id, input_sha256

    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(path, input_sha256)
    run_dir = runs_dir / run_id

    # Collision is very unlikely, but handle fast repeated invocations cleanly.
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{run_id}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir, run_dir.name, input_sha256


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Serialise the effective dataclass config into plain JSON values."""
    if not is_dataclass(cfg):
        msg = "cfg must be a dataclass instance"
        raise TypeError(msg)
    return asdict(cfg)


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_config_sidecar(cfg: Config, run_dir: Path) -> Path:
    """Write the effective config used for this run."""
    path = run_dir / "config.json"
    path.write_text(
        json.dumps(config_to_dict(cfg), indent=2, default=_json_default) + "\n"
    )
    return path


def _git_value(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = proc.stdout.strip()
    return value or None


def git_metadata() -> dict[str, object]:
    """Best-effort git metadata; safe outside git repos and in CI."""
    commit = _git_value("rev-parse", "HEAD")
    branch = _git_value("rev-parse", "--abbrev-ref", "HEAD")
    status = _git_value("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def package_version() -> str | None:
    try:
        return version("superjuggling")
    except PackageNotFoundError:
        return None


def write_command_sidecar(command: str | None, run_dir: Path) -> Path:
    """Write the exact command invocation when available."""
    path = run_dir / "command.txt"
    path.write_text((command or "").rstrip() + "\n")
    return path


def write_run_sidecar(
    *,
    run_dir: Path,
    run_id: str,
    input_path: Path,
    input_sha256: str,
    annotate: bool,
    debug_overlays: bool,
    annotated_path: Path | None,
) -> Path:
    """Write metadata/provenance for this run."""
    input_stat = input_path.stat()
    outputs: dict[str, str] = {
        "metrics": "metrics.json",
        "summary": "summary.md",
        "config": "config.json",
        "command": "command.txt",
    }
    if annotated_path is not None:
        outputs["annotated_video"] = annotated_path.name

    payload = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "size_bytes": input_stat.st_size,
        },
        "outputs": outputs,
        "options": {
            "annotate": annotate,
            "debug_overlays": debug_overlays,
        },
        "code": git_metadata(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "superjuggling": package_version(),
        },
    }

    path = run_dir / "run.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return path


def write_run_sidecars(
    *,
    run_dir: Path,
    run_id: str,
    input_path: Path,
    input_sha256: str,
    cfg: Config,
    annotate: bool,
    debug_overlays: bool,
    command: str | None,
    annotated_path: Path | None,
) -> dict[str, Path]:
    """Write all sidecars for an analysis run."""
    return {
        "config": write_config_sidecar(cfg, run_dir),
        "command": write_command_sidecar(command, run_dir),
        "run": write_run_sidecar(
            run_dir=run_dir,
            run_id=run_id,
            input_path=input_path,
            input_sha256=input_sha256,
            annotate=annotate,
            debug_overlays=debug_overlays,
            annotated_path=annotated_path,
        ),
    }
