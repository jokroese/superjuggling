"""Stage 6a — Annotated video (tech spec §4.6).

Replays the source clip and draws, per frame:

- motion trails per prop (``sv.TraceAnnotator``) — the signature juggling arcs,
- prop boxes + **track-ID tags** (``sv.BoxAnnotator`` / ``sv.LabelAnnotator``),
- wrist/arm keypoints (``sv.VertexAnnotator``) when pose data is present,
- the floor drop-zone outline,
- apex markers flashed at throw events,
- a live metrics HUD (consistency score, cadence, running throw/drop counts),
- optional diagnostic overlays:
  raw pre-tracking detections, low-confidence detections, recent trajectory
  samples, drop flashes, and per-frame debug counts,

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


def _draw_debug_detections(
    frame: Any,
    cv2: Any,
    detections: Any,
    low_confidence_threshold: float,
) -> tuple[int, int]:
    """Draw raw detector output before tracking.

    This answers the first debugging question in most CV failures: did the
    detector see the ball at all, or did tracking/event extraction lose it?
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
    result: AnalysisResult,
    t: float,
    window_s: float,
) -> int:
    """Draw recent accumulated trajectory samples as small points.

    The TraceAnnotator shows the tracker state frame-by-frame; this overlay
    shows what actually entered the downstream trajectory/event extractor.
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


def _flash_drops(
    frame: Any,
    cv2: Any,
    result: AnalysisResult,
    t: float,
    flash_s: float,
) -> int:
    """Flash a visible marker when a drop event fires."""
    n = 0
    by_track = {traj.track_id: traj for traj in result.trajectories}
    for drop in result.timelines.drops:
        if not (0.0 <= (t - drop.t) <= flash_s):
            continue
        traj = by_track.get(drop.track_id)
        if traj is not None and len(traj.t):
            idx = int(abs(traj.t - drop.t).argmin())
            x = int(traj.x[idx])
            y = int(traj.y[idx])
        else:
            x = 32
            y = frame.shape[0] - 32
        cv2.line(frame, (x - 12, y - 12), (x + 12, y + 12), _RED, 2, cv2.LINE_AA)
        cv2.line(frame, (x - 12, y + 12), (x + 12, y - 12), _RED, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "DROP",
            (x + 16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _RED,
            2,
            cv2.LINE_AA,
        )
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


def annotate_video(
    result: AnalysisResult,
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

            if debug_overlays:
                _flash_drops(frame, cv2, result, t, flash_s)
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
