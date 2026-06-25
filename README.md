# Superjuggling

**Superjuggling** is a pre-release ball-tracking workbench for juggling videos.
Its current job is deliberately narrow: detect juggling balls, turn detections
into centre candidates, link those candidates into trajectories, and make the
tracking result easy to inspect.

## What's implemented (draft)

Current architecture:

| Stage | Module | Status |
|---|---|---|
| 1. Ingest | `ingest.py` | scaffold (needs `cv` extra) |
| 2. Detection | `detection.py`, `heatmap.py` | YOLO boxes + multi-frame motion heatmaps |
| 3. Candidates | `candidates.py` | YOLO boxes → centre candidates |
| 4. Linking | `tracking.py`, `linking.py` | ByteTrack + experimental centre/ballistic linkers |
| 5. Report | `report.py` | tracking JSON + markdown summary |
| 6. Annotated video | `annotate.py` | traces, IDs, candidates, debug overlays |

Consistency scoring, symmetry, hand assignment, catches and drops are not part
of the active product surface yet. Those should come after the tracking layer is
validated.

## Usage

```bash
# Analytical core + tests (no models needed):
uv sync --dev
uv run pytest

# Full video pipeline (downloads the CV stack: supervision, ultralytics, ...):
uv sync --extra cv
uv run superjuggling analyze run1
```

The CLI looks for clips in `data/videos/`, so these are equivalent when the
file exists:

```bash
uv run superjuggling analyze run1
uv run superjuggling analyze run1.mp4
uv run superjuggling analyze data/videos/run1.mp4
```

By default each analysis writes to a fresh run directory:

`runs/YYYY-MM-DD_HHMMSS_<video-stem>_<input-hash>/`

For example:

`runs/2026-06-25_153012_run1_8f3a2c1/`

Each run contains:

| File | Purpose |
|------|---------|
| `tracking.json` | Machine-readable tracking report |
| `summary.md` | Human-readable tracking summary |
| `annotated.mp4` | Annotated video (default; pass `--no-annotate` to skip) |
| `config.json` | Effective configuration used for the run |
| `run.json` | Input hash, output names, options, code and environment metadata |
| `command.txt` | Exact command invocation |

To choose a fixed output directory:

```bash
uv run superjuggling analyze data/videos/run1.mp4 --out runs/manual-test
```

Existing output directories are not overwritten unless you pass `--overwrite`.

The prop detector needs fine-tuned weights (`--out` aside, set
`Config.detection.model_path`); COCO's "sports ball" class is unreliable for fast
props (§4.2). Until custom weights are configured, the CLI falls back to COCO's
`sports ball` class so the tool is usable out of the box. Treat those numbers as
diagnostic, not final-quality tracking.

For strict runs that must use fine-tuned prop weights:

```bash
uv run superjuggling analyze run1 --require-model
```

`annotated.mp4` is rendered by default because it is the fastest way to inspect
tracking quality. For report-only runs:

```bash
uv run superjuggling analyze run1 --no-annotate
```

### Tracking and candidate backends

The default path is candidate-native and motion-aware:

```bash
uv run superjuggling analyze run1
```

which currently means:

```text
YOLO boxes + multi-frame motion heatmap → fused centre candidates → ballistic linker → ball trajectories
```

Candidate sources:

```bash
uv run superjuggling analyze run1 --candidate-source yolo
uv run superjuggling analyze run1 --candidate-source heatmap
uv run superjuggling analyze run1 --candidate-source hybrid
```

Tracking/linking backends:

```bash
uv run superjuggling analyze run1 --tracking bytetrack --candidate-source yolo
uv run superjuggling analyze run1 --tracking centre
uv run superjuggling analyze run1 --tracking ballistic
```

`bytetrack` is box-native, so it requires YOLO detections. `centre` and
`ballistic` consume centre candidates from any source.

### Diagnostic overlays

To debug detection/tracking failures, render the annotated video with raw
pre-tracking detections, low-confidence highlights, recent trajectory samples,
centre candidates, and per-frame debug counts:

```bash
uv run superjuggling analyze run1 --debug-overlays
```

This also writes debug CSV artefacts:

- `debug_candidates.csv`
- `debug_links.csv` when using `--tracking centre` or `--tracking ballistic`

---

## Quickstart

```bash
uv sync --dev
git add uv.lock
git commit -m "Add uv lockfile"
uv run pre-commit install
uv run pre-commit run -a
uv run pytest
```

CI uses `uv sync --locked`; it will fail until `uv.lock` is committed.

## Common commands

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyright
uv run pytest
```

## Updating dependencies

- Update lockfile (respecting constraints in pyproject.toml):

```bash
uv lock --upgrade
uv sync --dev
```

The uv-lock pre-commit hook also updates uv.lock when pyproject.toml changes.

---

## How you use it

From anywhere:

```bash
copier copy path/to/copier-uv-template my-new-repo
cd my-new-repo
uv sync --dev
uv run pytest
```
