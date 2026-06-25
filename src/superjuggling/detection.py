"""Stage 2 — Detection (tech spec §4.2, §7).

Ball detector wrapper. Requires the ``cv`` extra.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ._optional import require
from .config import DetectionConfig


@runtime_checkable
class Detector(Protocol):
    """A per-frame prop detector. Returns an ``sv.Detections`` (typed ``Any``
    to keep this module importable without the ``cv`` extra)."""

    def __call__(self, frame: NDArray[np.uint8]) -> Any: ...


class YOLOPropDetector:
    """Ultralytics YOLO prop detector → ``sv.Detections`` (§4.2).

    Confidence is tuned low: props are small and motion-blurred at the apex,
    and we lean on tracking + smoothing to suppress spurious detections.
    """

    def __init__(self, cfg: DetectionConfig) -> None:
        ultralytics = require("ultralytics")
        self._conf = cfg.confidence
        self._ball_class_id: int | None = None
        if cfg.model_path:
            self._model = ultralytics.YOLO(cfg.model_path)
        elif cfg.allow_coco:
            # Smoke-test fallback: COCO weights filtered to "sports ball" (§4.2).
            import warnings

            warnings.warn(
                "Running with COCO 'sports ball' weights (--allow-coco): "
                "detections are unreliable for fast juggling props; use only "
                "to validate pipeline plumbing, not real metrics (§4.2).",
                stacklevel=2,
            )
            self._model = ultralytics.YOLO(cfg.coco_model_path)
            self._ball_class_id = cfg.coco_ball_class_id
        else:
            msg = (
                "DetectionConfig.model_path must point to fine-tuned prop "
                "weights (COCO 'sports ball' is unreliable for fast props; "
                "§4.2). Pass --allow-coco for a plumbing-only smoke test."
            )
            raise ValueError(msg)

    def __call__(self, frame: NDArray[np.uint8]) -> Any:
        sv = require("supervision")
        result = self._model(frame, conf=self._conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        if self._ball_class_id is not None:
            detections = detections[detections.class_id == self._ball_class_id]
        return detections
