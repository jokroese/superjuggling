"""Stage 2 — Detection (tech spec §4.2, §7).

Two detectors run per frame: a prop detector (fine-tuned YOLO) and a hand/pose
estimator (wrist keypoints). Models are pluggable behind the ``Detector``
protocol so we can swap Ultralytics for Roboflow Inference, RT-DETR, etc.
without touching the pipeline (§7).

Requires the ``cv`` extra.
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


@runtime_checkable
class PoseEstimator(Protocol):
    """A per-frame wrist/keypoint estimator. Returns ``sv.KeyPoints``."""

    def __call__(self, frame: NDArray[np.uint8]) -> Any: ...


class YOLOPropDetector:
    """Ultralytics YOLO prop detector → ``sv.Detections`` (§4.2).

    Confidence is tuned low: props are small and motion-blurred at the apex,
    and we lean on tracking + smoothing to suppress spurious detections.
    """

    def __init__(self, cfg: DetectionConfig) -> None:
        ultralytics = require("ultralytics")
        if not cfg.model_path:
            msg = (
                "DetectionConfig.model_path must point to fine-tuned prop "
                "weights (COCO 'sports ball' is unreliable for fast props; §4.2)"
            )
            raise ValueError(msg)
        self._model = ultralytics.YOLO(cfg.model_path)
        self._conf = cfg.confidence

    def __call__(self, frame: NDArray[np.uint8]) -> Any:
        sv = require("supervision")
        result = self._model(frame, conf=self._conf, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)


class YOLOPoseEstimator:
    """Ultralytics YOLO-Pose wrist estimator → ``sv.KeyPoints`` (§4.2)."""

    def __init__(self, cfg: DetectionConfig) -> None:
        ultralytics = require("ultralytics")
        model_path = cfg.pose_model_path or "yolov8n-pose.pt"
        self._model = ultralytics.YOLO(model_path)

    def __call__(self, frame: NDArray[np.uint8]) -> Any:
        sv = require("supervision")
        result = self._model(frame, verbose=False)[0]
        return sv.KeyPoints.from_ultralytics(result)
