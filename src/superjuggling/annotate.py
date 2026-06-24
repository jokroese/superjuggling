"""Stage 6a — Annotated video (tech spec §4.6).

Replays the source clip and draws, per frame:

- motion trails per prop (``sv.TraceAnnotator``) — the signature juggling arcs,
- prop boxes + **track-ID tags** (``sv.BoxAnnotator`` / ``sv.LabelAnnotator``),
- wrist/arm keypoints (``sv.VertexAnnotator``) when pose data is present,
- the floor drop-zone outline,
- apex markers flashed at throw events,
- a live metrics HUD (consistency score, cadence, running throw/drop counts),

written through ``sv.VideoSink``. Requires the ``cv`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._optional import require

if TYPE_CHECKING:
    from .config import Config
    from .metrics import MetricsReport
    from .pipeline import AnalysisResult


_GREEN = (80, 220, 120)
_RED = (40, 40, 230)
_YELLOW = (40, 230, 230)
_WHITE = (245, 245, 245)


def _labels_for(detections: Any) -> list[str]:
    """Build a ``#<track-id>`` tag for each detection (tech spec §4.6)."""
    ids = detections.tracker_id
    if ids is None:
        return ["?"] * len(detections)
    return [f"#{int(tid)}" for tid in ids]


def _draw_hud(
    frame: Any,
    cv2: Any,
    metrics: MetricsReport,
    n_throws: int,
    n_drops: int,
) -> None:
    """Translucent top-left panel with the live metrics."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 132), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    lines = [
        (f"Consistency  {metrics.overall_consistency:.0f}/100", _GREEN),
        (f"Cadence  {metrics.rhythm.cadence_hz:.1f} thr/s", _WHITE),
        (f"Throws  {n_throws}", _WHITE),
        (f"Drops  {n_drops}", _RED if n_drops else _WHITE),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (14, 32 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _flash_apexes(
    frame: Any,
    cv2: Any,
    throws: list[Any],
    t: float,
    flash_s: float,
) -> None:
    """Flash a marker at each throw's apex for a short window after it fires."""
    for thr in throws:
        if 0.0 <= (t - thr.t) <= flash_s:
            center = (int(thr.apex_x), int(thr.apex_y))
            cv2.circle(frame, center, 14, _YELLOW, 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                thr.hand.value,
                (center[0] + 16, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _YELLOW,
                2,
                cv2.LINE_AA,
            )


def annotate_video(
    result: AnalysisResult,
    out_path: Path,
    cfg: Config,
) -> Path:
    """Render the annotated output video from collected per-frame overlay data.

    Requires the analysis to have been run with ``collect_overlay=True`` so the
    per-frame tracked detections are available.
    """
    sv = require("supervision")
    cv2 = require("cv2")

    meta = result.meta
    metrics = result.metrics
    frame_dets = result.frame_detections
    frame_kps = result.frame_keypoints
    fps = meta.fps or 30.0

    if not frame_dets:
        msg = (
            "No per-frame detections to annotate. Run the pipeline with "
            "annotate=True / collect_overlay=True first."
        )
        raise ValueError(msg)

    color_by_track = sv.ColorLookup.TRACK
    trace = sv.TraceAnnotator(
        trace_length=int(fps * 1.5), thickness=2, color_lookup=color_by_track
    )
    box = sv.BoxAnnotator(thickness=2, color_lookup=color_by_track)
    label = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, color_lookup=color_by_track
    )
    vertex = sv.VertexAnnotator(radius=6, color=sv.Color(*reversed(_GREEN)))

    floor_y = int(meta.height * cfg.events.floor_zone_fraction)
    throws = result.timelines.throws
    drop_times = sorted(d.t for d in result.timelines.drops)
    flash_s = max(cfg.events.min_inter_throw_s, 2.0 / fps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_info = sv.VideoInfo.from_video_path(meta.path)

    empty = sv.Detections.empty()
    with sv.VideoSink(str(out_path), video_info=video_info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(meta.path)):
            t = i / fps
            dets = frame_dets[i] if i < len(frame_dets) else empty

            frame = trace.annotate(frame, dets)
            frame = box.annotate(frame, dets)
            if len(dets):
                frame = label.annotate(frame, dets, labels=_labels_for(dets))

            if frame_kps and i < len(frame_kps) and frame_kps[i] is not None:
                try:
                    frame = vertex.annotate(frame, frame_kps[i])
                except (ValueError, IndexError):
                    pass  # no / empty keypoints on this frame — skip

            # Floor drop-zone outline (tech spec §4.4 / §4.6).
            cv2.line(frame, (0, floor_y), (meta.width, floor_y), _RED, 1, cv2.LINE_AA)

            _flash_apexes(frame, cv2, throws, t, flash_s)

            n_throws = sum(1 for thr in throws if thr.t <= t)
            n_drops = sum(1 for dt in drop_times if dt <= t)
            _draw_hud(frame, cv2, metrics, n_throws, n_drops)

            sink.write_frame(frame)
    return out_path
