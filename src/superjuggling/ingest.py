"""Stage 1 — Ingest (tech spec §4.1).

Reads clip metadata and yields frames lazily via supervision. Requires the
``cv`` extra.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from ._optional import require
from .config import IngestConfig
from .models import VideoMeta


class IngestError(ValueError):
    """Raised when a clip fails the ingest guardrails (§4.1)."""


def probe(path: str) -> VideoMeta:
    """Read resolution / fps / frame count via ``sv.VideoInfo``."""
    sv = require("supervision")
    info = sv.VideoInfo.from_video_path(path)
    return VideoMeta(
        path=path,
        fps=float(info.fps),
        total_frames=int(info.total_frames or 0),
        width=int(info.width),
        height=int(info.height),
    )


def validate(meta: VideoMeta, cfg: IngestConfig) -> list[str]:
    """Check ingest guardrails; return a list of human-readable warnings.

    fps is critical — every temporal metric is computed in seconds (§4.1).
    """
    warnings: list[str] = []
    if meta.height < cfg.min_height:
        warnings.append(
            f"resolution {meta.height}p is below the recommended {cfg.min_height}p"
        )
    if meta.fps < cfg.min_fps:
        raise IngestError(
            f"fps {meta.fps} is below the minimum {cfg.min_fps}; "
            "temporal metrics would be unreliable"
        )
    if meta.fps < cfg.recommended_fps:
        warnings.append(
            f"fps {meta.fps} is below the recommended "
            f"{cfg.recommended_fps} (apex aliasing risk, §8)"
        )
    if meta.duration_s > cfg.max_duration_s:
        raise IngestError(
            f"duration {meta.duration_s:.0f}s exceeds the "
            f"{cfg.max_duration_s:.0f}s guardrail"
        )
    return warnings


def frames(path: str) -> Iterator[NDArray[np.uint8]]:
    """Lazy frame generator (avoids loading the whole clip into memory)."""
    sv = require("supervision")
    yield from sv.get_video_frames_generator(path)
