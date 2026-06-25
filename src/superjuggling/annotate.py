"""Stage 6a — Annotated video (tech spec §4.6).

Replays the source clip and draws, per frame:

- motion trails per prop (``sv.TraceAnnotator``) — the signature juggling arcs,
- prop boxes + **track-ID tags** (``sv.BoxAnnotator`` / ``sv.LabelAnnotator``),
- optional diagnostic overlays:
  raw pre-tracking detections, low-confidence detections, recent trajectory
  samples, candidate centres, and per-frame debug counts,

written through ``sv.VideoSink``. Requires the ``cv`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._optional import require

if TYPE_CHECKING:
    from .config import Config
    from .pipeline import TrackingResult


_GREEN = (80, 220, 120)
_WHITE = (245, 245, 245)
_BLUE = (230, 160, 60)
_MAGENTA = (220, 80, 220)


def _labels_for(detections: Any) -> list[str]:
    """Build a ``#<track-id>`` tag for each detection (tech spec §4.6)."""
    ids = detections.tracker_id
    if ids is None:
        return ["?"] * len(detections)
    return [f"#{int(tid)}" for tid in ids]


def _draw_hud(
    frame: Any,
    cv2: Any,
    backend: str,
    candidates: int,
    trajectories: int,
) -> None:
    """Translucent top-left panel with tracking diagnostics."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 112), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    lines = [
        ("Ball tracking", _GREEN),
        (f"Backend  {backend}", _WHITE),
        (f"Candidates  {candidates}", _WHITE),
        (f"Trajectories  {trajectories}", _WHITE),
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


def _draw_debug_detections(
    frame: Any,
    cv2: Any,
    detections: Any,
    low_confidence_threshold: float,
) -> tuple[int, int]:
    """Draw raw detector output before tracking.

    This answers the first debugging question in most CV failures: did the
    detector see the ball at all, or did tracking lose it?
    """
    if detections is None or not len(detections):
        return 0, 0

    xyxy = detections.xyxy
    conf = detections.confidence
    if conf is None:
        conf = [1.0] * len(xyxy)

    n_low = 0
    for box, score in zip(xyxy, conf, strict=False):
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        score_f = float(score)
        is_low = score_f < low_confidence_threshold
        if is_low:
            n_low += 1

        color = _MAGENTA if is_low else _BLUE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{score_f:.2f}",
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return len(detections), n_low


def _draw_recent_trajectory_samples(
    frame: Any,
    cv2: Any,
    result: TrackingResult,
    t: float,
    window_s: float,
) -> int:
    """Draw recent accumulated trajectory samples as small points.

    The TraceAnnotator shows the tracker state frame-by-frame; this overlay
    shows what actually entered the downstream trajectory extractor.
    """
    n = 0
    t0 = max(0.0, t - window_s)
    for traj in result.trajectories:
        mask = (traj.t >= t0) & (traj.t <= t)
        xs = traj.x[mask]
        ys = traj.y[mask]
        for x, y in zip(xs, ys, strict=False):
            cv2.circle(frame, (int(x), int(y)), 2, _GREEN, -1, cv2.LINE_AA)
            n += 1
    return n


def _draw_debug_hud(
    frame: Any,
    cv2: Any,
    raw_count: int,
    low_count: int,
    tracked_count: int,
    sample_count: int,
) -> None:
    """Small bottom-left panel for diagnostic overlay counts."""
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 116), (360, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    lines = [
        f"raw detections: {raw_count} ({low_count} low-conf)",
        f"tracked detections: {tracked_count}",
        f"recent trajectory samples: {sample_count}",
        "debug: raw=blue low=magenta traj=green",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (14, h - 86 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            _WHITE,
            1,
            cv2.LINE_AA,
        )


def _draw_candidate_points(
    frame: Any,
    cv2: Any,
    candidates: list[Any],
) -> None:
    """Draw centre candidates, independent of any tracking backend."""
    for cand in candidates:
        radius = int(max(4.0, cand.radius_px or 4.0))
        cv2.circle(frame, (int(cand.x), int(cand.y)), radius, _BLUE, 1, cv2.LINE_AA)
        cv2.circle(frame, (int(cand.x), int(cand.y)), 2, _BLUE, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{cand.source}:{cand.score:.2f}",
            (int(cand.x) + 6, int(cand.y) + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _BLUE,
            1,
            cv2.LINE_AA,
        )


def annotate_video(
    result: TrackingResult,
    out_path: Path,
    cfg: Config,
    debug_overlays: bool = False,
) -> Path:
    """Render the annotated output video from collected per-frame overlay data.

    Requires the analysis to have been run with ``collect_overlay=True`` so the
    per-frame tracked detections are available.
    """
    sv = require("supervision")
    cv2 = require("cv2")

    meta = result.meta
    frame_dets = result.frame_detections
    fps = meta.fps or 30.0

    if not (
        frame_dets
        or result.candidate_frames
        or result.trajectories
        or result.frame_raw_detections
    ):
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_info = sv.VideoInfo.from_video_path(meta.path)

    empty = sv.Detections.empty()
    with sv.VideoSink(str(out_path), video_info=video_info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(meta.path)):
            t = i / fps
            dets = frame_dets[i] if i < len(frame_dets) else empty

            if frame_dets:
                frame = trace.annotate(frame, dets)
                frame = box.annotate(frame, dets)
                if len(dets):
                    frame = label.annotate(frame, dets, labels=_labels_for(dets))

            raw_count = 0
            low_count = 0
            sample_count = 0
            if debug_overlays:
                raw_dets = (
                    result.frame_raw_detections[i]
                    if i < len(result.frame_raw_detections)
                    else empty
                )
                raw_count, low_count = _draw_debug_detections(
                    frame, cv2, raw_dets, cfg.tracking.track_activation_threshold
                )
                sample_count = _draw_recent_trajectory_samples(
                    frame, cv2, result, t, window_s=1.0
                )
                if i < len(result.candidate_frames):
                    _draw_candidate_points(
                        frame, cv2, result.candidate_frames[i].candidates
                    )

            candidate_count = (
                len(result.candidate_frames[i].candidates)
                if i < len(result.candidate_frames)
                else 0
            )
            _draw_hud(
                frame,
                cv2,
                backend=result.summary.backend,
                candidates=candidate_count,
                trajectories=len(result.trajectories),
            )

            if debug_overlays:
                _draw_debug_hud(
                    frame,
                    cv2,
                    raw_count=raw_count,
                    low_count=low_count,
                    tracked_count=len(dets),
                    sample_count=sample_count,
                )

            sink.write_frame(frame)
    return out_path
