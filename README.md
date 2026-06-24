# Superjuggling



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

