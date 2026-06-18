#!/usr/bin/env bash
# Local mirror of the dormant GitHub Actions gate (Actions minutes are exhausted).
# Run from repo root. Any failure aborts (set -e).
set -euo pipefail
RUN="env -u VIRTUAL_ENV uv run --no-sync"

echo "== lock-check =="
env -u VIRTUAL_ENV uv lock --check

echo "== lint: ruff check =="
$RUN ruff check src/ tests/
echo "== lint: ruff format --check =="
$RUN ruff format --check src/ tests/
echo "== lint: mypy =="
$RUN mypy src/jaxstro

echo "== test-matrix (current interpreter; CI does 3.11/3.12/3.13) =="
$RUN pytest -m "not slow" -q

echo "== ml-integration =="
env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest tests/integration -q

echo "== wheel-smoke =="
env -u VIRTUAL_ENV uv build --wheel -o dist/
rm -rf /tmp/jaxstro-clean
env -u VIRTUAL_ENV uv venv /tmp/jaxstro-clean
env -u VIRTUAL_ENV uv pip install --python /tmp/jaxstro-clean/bin/python dist/*.whl
/tmp/jaxstro-clean/bin/python -c "import jaxstro; print(jaxstro.__name__, 'imports clean')"

echo "ALL LOCAL GATES PASSED"
