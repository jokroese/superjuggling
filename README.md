# Superjuggling

**Superjuggling** ingests a video of someone juggling and outputs a battery of
**consistency metrics** — quantitative measures of how steady the pattern is
over time (rhythm, height, placement, symmetry, drops, endurance). See
[`docs/TECH_SPEC.md`](docs/TECH_SPEC.md) for the full design.

## What's implemented (draft)

This is a draft skeleton following the tech spec's stage architecture (§3):

| Stage | Module | Status |
|---|---|---|
| 1. Ingest | `ingest.py` | scaffold (needs `cv` extra) |
| 2. Detection | `detection.py` | pluggable `Detector` interface + YOLO impls |
| 3. Tracking & smoothing | `tracking.py` | ByteTrack + smoother wrapper |
| 4. Event extraction | `events.py` | **implemented** (apex/throw/catch/drop) |
| 5. Metrics engine | `metrics.py` | **implemented** (§5 metrics + scores) |
| 6a. Annotated video | `annotate.py` | scaffold |
| 6b. Report | `report.py` | **implemented** (JSON + markdown) |

The analytical core (stages 4–6b) is pure NumPy/SciPy and fully unit-tested
against synthetic fixtures — no models or video needed. The computer-vision
stages (1–3, 6a) sit behind the optional `cv` extra and are lazy-imported.

## Usage

```bash
# Analytical core + tests (no models needed):
uv sync --dev
uv run pytest

# Full video pipeline (downloads the CV stack: supervision, ultralytics, ...):
uv sync --extra cv
uv run superjuggling analyze run1.mp4 --out report/
```

Outputs `report/metrics.json` (schema in tech spec §6) and `report/summary.md`.

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

