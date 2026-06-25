# Superjuggling — Juggling Consistency Tracker

**Tech Spec**
Author: sol@titles.xyz · Date: 2026-06-24 · Status: Draft v1

---

## 1. Overview

**Superjuggling** is a video-analysis tool that ingests a video of someone juggling
and outputs a battery of **consistency metrics** — quantitative measures of how
steady the pattern is over time. It is built on
[`roboflow/supervision`](https://github.com/roboflow/supervision), a
model-agnostic computer-vision toolkit, for detection, multi-object tracking,
spatial analysis, and annotated video output.

### Goals

- Detect and track every prop (ball/club/ring) and the juggler's hands
  throughout a clip.
- Reduce the video to clean, timestamped **throw/catch event timelines** per
  prop and per hand.
- Derive a battery of consistency metrics: rhythm, height, placement, pattern
  symmetry, drops, and clean-run endurance.
- Produce an **annotated output video** (motion trails, throw markers, live
  metrics) and a **structured report** (JSON + summary + plots).
- Run offline on recorded clips first; design the pipeline so it can later run
  near-real-time.

---

## 2. Background: what "consistency" means here

| Repeating event | Consistency = low variance of… | Reads as |
|---|---|---|
| throw | inter-throw interval | steady **rhythm** |
| throw | apex height | even, controlled **throws** |
| throw | apex x-position (per hand) | accurate **placement** |
| catch | catch-to-throw dwell time | clean **hand timing** |
| pattern | left vs. right balance | **symmetry** |

The core technical problem is therefore: **turn pixels into clean, labeled
event timelines for each prop and hand, then do time-series statistics on
them.**

---

## 3. System Architecture

```
                    ┌──────────────────────────────────────────────┐
   video.mp4 ─────▶ │  1. Ingest (sv.VideoInfo, frame generator)    │
                    └──────────────────────────────────────────────┘
                                       │ frames
                    ┌──────────────────▼───────────────────────────┐
                    │  2. Detection                                  │
                    │     • Prop detector  (YOLO via Ultralytics)    │
                    │     • Hand/pose keypoints (wrists)             │
                    │     → sv.Detections, sv.KeyPoints              │
                    └──────────────────┬───────────────────────────┘
                                       │ per-frame detections
                    ┌──────────────────▼───────────────────────────┐
                    │  3. Tracking & Smoothing                       │
                    │     • sv.ByteTrack  (stable prop IDs)          │
                    │     • sv.DetectionsSmoother (temporal smooth)  │
                    │     → per-prop (x,y) trajectories vs. time      │
                    └──────────────────┬───────────────────────────┘
                                       │ trajectories
                    ┌──────────────────▼───────────────────────────┐
                    │  4. Event Extraction                           │
                    │     • throw apex detection (local y-minima)    │
                    │     • catch detection (return to hand line)    │
                    │     • hand assignment (which hand threw it)    │
                    │     • drop detection (track loss / floor zone) │
                    └──────────────────┬───────────────────────────┘
                                       │ event timelines
                    ┌──────────────────▼───────────────────────────┐
                    │  5. Metrics Engine                             │
                    │     consistency stats over the timelines        │
                    └──────────────────┬───────────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                          ▼
        ┌──────────────────┐                    ┌─────────────────────┐
        │ 6a. Annotated     │                    │ 6b. Report          │
        │     video         │                    │  metrics.json + md  │
        │ (sv annotators,   │                    │  + plots (png)      │
        │  sv.VideoSink)    │                    └─────────────────────┘
        └──────────────────┘
```

The pipeline is a sequence of pure-ish stages so each can be tested in
isolation and cached. Stages 2–3 are per-frame and streamed; stages 4–6 operate
on the accumulated timelines.

---

## 4. Component Detail

### 4.1 Ingest

- `sv.VideoInfo.from_video_path(path)` → resolution, **fps**, total frames. fps
  is critical: every temporal metric is computed in seconds, not frames.
- `sv.get_video_frames_generator(path)` → lazy frame iterator (avoids loading
  the whole clip into memory).
- Validate: minimum resolution (≥720p recommended), minimum fps (≥30; 60 fps
  strongly preferred — see §8 on apex sampling), max-duration guardrail.

### 4.2 Detection

Two detectors run per frame:

**Prop detector.** A YOLO detection model (Ultralytics) fine-tuned on juggling
props. Output is converted with `sv.Detections.from_ultralytics(result)`.
- v1 ships with a **custom-trained model** because COCO's "sports ball" class is
  unreliable for small, fast, motion-blurred juggling balls and does not cover
  clubs or rings at all. Training data: labeled frames of common props (§7).
- Confidence threshold tuned low (props are small and blur badly at the apex of
  fast throws), with tracking + smoothing relied on to suppress spurious
  detections.

**Hand/pose estimator.** A keypoint model (e.g. YOLO-Pose) producing
`sv.KeyPoints` for the juggler's **wrists** (and elbows/shoulders for a body
reference frame). Wrist positions define the "hand line" used to detect catches
and to assign each throw to a hand.

### 4.3 Tracking & Smoothing

- `sv.ByteTrack` assigns a **persistent `tracker_id`** to each prop across
  frames — this is what lets us follow one ball through its whole flight and
  measure *its* throw heights over time.
  - ByteTrack tuned for fast small objects: lower
    `track_activation_threshold`, generous `lost_track_buffer` so a prop briefly
    occluded behind a hand isn't given a new ID on reappearance.
  - ID swaps between near-identical props are expected. Most metrics aggregate
    across props and tolerate this; per-prop metrics re-segment trajectories at
    apexes (§4.4) rather than trusting IDs end-to-end.
- `sv.DetectionsSmoother` smooths box position/size across a sliding window to
  produce clean trajectories for apex detection (raw detections jitter).
- Output of this stage: for each prop track, a time series
  `(t, x_center, y_center, w, h, confidence)`; plus per-frame wrist coordinates.

### 4.4 Event Extraction

Turns trajectories into discrete, timestamped events.

**Throw apex (the beat).** Image y increases downward, so a thrown prop's apex
is a **local minimum of y(t)**. For each prop trajectory:
1. Smooth y(t).
2. Find local minima with a minimum prominence (filters micro-jitter) and a
   minimum inter-apex spacing (refractory period from expected cadence).
3. Each apex = one throw event with: timestamp, apex height (peak y relative to
   the hand line), and apex x (lateral position).
4. Sub-frame refinement: fit a parabola to the 3 samples around the minimum to
   recover true apex time/height between frames (matters at 30 fps — see §8).

**Catch & dwell.** A catch is the trajectory returning to the hand line (local
maximum of y near a wrist). The interval a prop spends between catch and its
next throw is **hand dwell time** — a sensitive indicator of rushed/late hands.

**Hand assignment.** Each throw is attributed to left or right hand by the
nearest wrist at the moment of the preceding catch (or launch). This enables
per-hand rhythm, height, and symmetry metrics.

**Drop detection.** A drop is inferred when:
- a prop track terminates (lost beyond `lost_track_buffer`) **and** its last
  positions enter a floor `sv.PolygonZone` near the bottom of the frame, or
- the active prop count falls below the established baseline for > N frames.

The `sv.PolygonZone` along the frame floor gives robust drop confirmation and
filters tracks that merely left frame at the top.

### 4.5 Metrics Engine

Consumes event timelines, emits the metrics in §5. Pure Python/NumPy/SciPy; no
CV dependency, so it is unit-testable against synthetic timelines.

### 4.6 Outputs

**Run directory.** Each CLI invocation writes to a dedicated directory under
`runs/` (default name: `YYYY-MM-DD_HHMMSS_<video-stem>_<input-hash>/`; override
with `--out`). Input clips live in `data/videos/`; local model weights in
`models/`.

Per-run artefacts:

| File | Purpose |
|------|---------|
| `metrics.json` | Machine-readable metrics report (schema in §6) |
| `summary.md` | Human-readable summary |
| `annotated.mp4` | Annotated video, when `--annotate` or `--debug-overlays` is enabled |
| `config.json` | Effective configuration used for the run |
| `run.json` | Provenance: input hash, output names, options, git and environment metadata |
| `command.txt` | Exact command invocation |

**Annotated video** via supervision annotators written through `sv.VideoSink`:
- `sv.TraceAnnotator` — motion trails per prop (the signature juggling arcs).
- `sv.BoxAnnotator` / `sv.DotAnnotator` — prop boxes/centers with track IDs.
- `sv.VertexAnnotator` / `EdgeAnnotator` — wrist/arm keypoints.
- Custom overlay: live cadence, drop count, current consistency score, apex
  markers flashed at throw events, the floor drop-zone outline.

**Report** (see table above): future PNG plots (inter-throw-interval over time,
apex-height distribution, left/right balance, pattern envelope). Plotting is
matplotlib, decoupled from the pipeline; not yet implemented.

---

## 5. Metrics

All consistency metrics are reported as **coefficient of variation (CV =
std/mean)** so they're scale- and unit-independent and comparable across clips,
plus an inverted **0–100 consistency score** (`100 * (1 - clamp(CV))`) for
at-a-glance reading. Each metric also reports raw mean / std / n.

### 5.1 Rhythm (timing)

| Metric | Definition | Why it matters |
|---|---|---|
| **Throw cadence** | throws per second (mean) | the pattern's tempo |
| **Inter-throw-interval CV** | CV of time between consecutive throws | core rhythm steadiness |
| **Hand-dwell CV** | CV of catch-to-throw dwell time | rushed vs. late hands |
| **Left/right tempo balance** | ratio of mean inter-throw interval, L vs. R | even-handedness of timing |

### 5.2 Height & placement (spatial)

| Metric | Definition | Why it matters |
|---|---|---|
| **Throw-height CV** | CV of apex heights | even, controlled throws vs. erratic |
| **Apex-lateral CV** | CV of apex x-position, per hand | are throws landing in the same place |
| **Pattern width** | mean horizontal span of apexes | tightness/openness of the pattern |
| **Pattern-envelope stability** | drift of the bounding envelope of all apexes over time | does the pattern "walk" or stay planted |
| **Height-to-cadence coherence** | do height and tempo co-vary as physics predicts (higher throw ⇒ longer interval) | distinguishes intentional height changes from sloppiness |

### 5.3 Symmetry & balance

| Metric | Definition | Why it matters |
|---|---|---|
| **L/R height symmetry** | ratio of mean apex height, L vs. R | balanced cascade |
| **L/R placement symmetry** | mirror-distance of L vs. R apex x | symmetric pattern shape |
| **Throw-count balance** | fraction of throws per hand | both hands sharing the load |

### 5.4 Failure & endurance

| Metric | Definition | Why it matters |
|---|---|---|
| **Drop count / drop rate** | drops total and per minute | the hardest failure signal |
| **Longest clean run** | max time / max throws between drops | endurance of consistency |
| **Time-to-instability** | when CV first exceeds a threshold within the clip | when does fatigue/drift set in |
| **Recovery time** | mean time from drop to re-established stable pattern | resilience |

### 5.5 Composite

- **Overall consistency score** — weighted blend of rhythm, spatial, symmetry,
  and failure domains (weights configurable). One headline 0–100 number for
  longitudinal tracking across many clips/sessions.

---

## 6. Data Schemas

`metrics.json` (abridged):

```json
{
  "video": { "path": "run1.mp4", "fps": 60, "duration_s": 124.5, "resolution": [1920,1080] },
  "props": { "count_estimate": 3, "tracks": 9 },
  "events": {
    "throws": [{ "t": 1.02, "track_id": 4, "hand": "R", "apex_y": 312, "apex_x": 870, "height_px": 240 }],
    "catches":[{ "t": 1.41, "track_id": 4, "hand": "L" }],
    "drops":  [{ "t": 41.3, "track_id": 2 }]
  },
  "metrics": {
    "rhythm":   { "cadence_hz": 4.1, "iti_cv": 0.07, "dwell_cv": 0.11, "lr_tempo_balance": 0.98 },
    "spatial":  { "height_cv": 0.12, "apex_lateral_cv": 0.09, "pattern_width_px": 520, "envelope_drift": 0.03 },
    "symmetry": { "lr_height_symmetry": 0.96, "lr_placement_symmetry": 0.94, "throw_balance": 0.49 },
    "failure":  { "drops": 3, "drop_rate_per_min": 1.4, "longest_clean_run_s": 38.2, "recovery_s": 2.1 },
    "overall_consistency": 84
  },
  "scores": { "rhythm": 88, "spatial": 85, "symmetry": 90, "failure": 79 }
}
```

Internally, trajectories and event timelines are pandas DataFrames keyed by
time; this is the boundary between the CV pipeline (stages 2–4) and the
statistics engine (stage 5).

---

## 7. Models & Training Data

- **Prop detector**: fine-tune a small YOLO (e.g. YOLOv8/11-n or -s for speed)
  on labeled juggling frames. Bootstrap by auto-labeling with a generic
  detector + manual correction in Roboflow, then iterate.
  - Dataset target v1: ~2–3k labeled frames across prop types (balls, clubs,
    rings), lighting, and backgrounds. Heavy augmentation for motion blur and
    scale, since props are smallest and blurriest at the apex.
  - Export in YOLO format (supervision's dataset utilities handle
    load/split/merge).
- **Hands/pose**: off-the-shelf YOLO-Pose / equivalent for wrist keypoints; no
  training needed for v1.
- Models are pluggable behind a `Detector` interface so we can swap to Roboflow
  Inference, RT-DETR, or transformers models without touching the pipeline.

---

## 8. Technical Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Motion blur** on fast props | missed/low-confidence detections | high-fps capture (60+), low conf threshold, smoothing, motion-blur augmentation in training |
| **Apex aliasing** at low fps | throw heights/timing inaccurate | require ≥30 fps, recommend 60; sub-frame apex via parabola fit (§4.4) |
| **ID swaps** between identical props | corrupts per-prop metrics | aggregate metrics across props; re-segment trajectories at apexes instead of trusting ByteTrack IDs end-to-end |
| **Hand occlusion** at catch/throw | broken trajectories near the hand line | rely on the apex (top, unoccluded) as the beat; generous `lost_track_buffer`; infer catch from trajectory return |
| **Hand assignment errors** | wrong L/R attribution | use wrist keypoints + nearest-hand at catch; flag low-confidence assignments rather than guessing |
| **Camera motion** (handheld) | global drift contaminates spatial metrics | v1 assumes fixed camera; v2 measures apexes relative to shoulder midpoint (pose-anchored coords), not the frame |
| **Establishing prop count** | drop detection needs a baseline | estimate baseline as the mode of active-track count over a stable early window |
| **High prop counts (5/7-ball)** | denser, faster, more occlusion | metrics are count-agnostic; validate detector + tracker thresholds per count tier |

---

## 9. Phasing / Milestones

- **M0 — Skeleton (1 wk):** ingest → off-the-shelf detector → ByteTrack →
  annotated video out. Validates the supervision plumbing end to end. No metrics
  yet.
- **M1 — Rhythm & failure (2 wks):** custom prop model, apex/throw extraction,
  drop detection, §5.1 + §5.4 metrics + JSON report.
- **M2 — Spatial & symmetry (1.5 wks):** hand/pose integration, catch + hand
  assignment, §5.2 + §5.3, composite score, plots.
- **M3 — Hardening (ongoing):** camera-motion robustness, 5/7-ball validation,
  near-real-time mode, longitudinal dashboard across sessions.
- **v2 stretch:** siteswap-aware analysis (parse the pattern from throw
  heights/timing and validate it against a declared siteswap), trick
  classification, two-person passing.

---

## 10. Stack

- Python 3.11+
- `supervision`, `ultralytics` (YOLO), `numpy`, `scipy` (peak finding, signal
  processing), `pandas`, `matplotlib`
- `opencv-python` (transitively via supervision; used for custom overlays)
- CLI first: `superjuggling analyze data/videos/run1.mp4 --annotate` (writes to
  `runs/<timestamp>_<video-stem>_<input-hash>/`; use `--out` for a fixed path).
  Optional thin FastAPI wrapper later for upload-and-analyze.
- Tests: pytest, with synthetic trajectory/timeline fixtures for the metrics
  engine and a few short golden clips for the full pipeline.

---

## 11. Open Questions

1. Prop types in scope for v1 — balls only, or clubs/rings too? (affects
   detector training and apex modeling; clubs spin and have a non-point shape).
2. Prop-count range to support in v1 — just 3, or up to 5/7? (drives tracker
   tuning and occlusion handling).
3. Capture conditions we can mandate (fixed tripod? known fps? plain
   background?) — relaxing these is most of the M3/v2 work.
4. Is per-prop tracking ("this specific ball is consistently low") a v1
   requirement, or are aggregate metrics enough? Drives how hard we fight ID
   swaps.
5. Longitudinal store — do we need a DB/dashboard to track a juggler's
   consistency across sessions over time, or is per-clip JSON sufficient for v1?
