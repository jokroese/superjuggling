"""Stage 6a — Annotated video (tech spec §4.6).

Draws motion trails, prop boxes, wrist keypoints and a live metrics overlay,
written through ``sv.VideoSink``. This is a draft scaffold: it wires the
supervision annotators the spec calls for and leaves the per-frame redraw of
historical detections as a TODO (it needs the raw per-frame detection buffer).

Requires the ``cv`` extra.
"""

from __future__ import annotations

from pathlib import Path

from ._optional import require
from .metrics import MetricsReport
from .models import VideoMeta


def annotate_video(
    meta: VideoMeta,
    metrics: MetricsReport,
    out_path: Path,
) -> Path:
    """Render the annotated output video.

    Draft: sets up the annotators and sink and stamps the headline metrics.
    Full per-frame trace rendering (§4.6) requires the per-frame detection
    buffer to be threaded through from the tracking stage — tracked as a
    follow-up.
    """
    sv = require("supervision")
    cv2 = require("cv2")

    trace = sv.TraceAnnotator()
    box = sv.BoxAnnotator()
    _ = (trace, box)  # configured; per-frame application is the follow-up

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_info = sv.VideoInfo.from_video_path(meta.path)

    with sv.VideoSink(str(out_path), video_info=video_info) as sink:
        for frame in sv.get_video_frames_generator(meta.path):
            cv2.putText(
                frame,
                f"consistency {metrics.overall_consistency:.0f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            sink.write_frame(frame)
    return out_path
