"""Helper for the optional computer-vision stack (the ``cv`` extra).

Detection / tracking / annotation depend on supervision, ultralytics and
OpenCV, which are heavy and not installed in the default dev/CI environment.
These imports are deferred to call time and raise a clear, actionable error
when the extra is missing — keeping the tracking core and tests dependency-free.
"""

from __future__ import annotations

from types import ModuleType


class MissingCVDependency(RuntimeError):
    """Raised when a video-pipeline stage needs the optional ``cv`` extra."""


def require(module: str) -> ModuleType:
    """Import ``module`` or raise a friendly install hint."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        msg = (
            f"'{module}' is required for this stage of the video pipeline. "
            "Install the computer-vision extra with:\n"
            "    uv sync --extra cv"
        )
        raise MissingCVDependency(msg) from exc
