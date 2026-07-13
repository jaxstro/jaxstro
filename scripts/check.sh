#!/usr/bin/env bash
# Full local release gate. PR CI is intentionally smaller; the scheduled/manual
# full-gate workflow mirrors these exhaustive checks across supported Python versions.
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

echo "== provenance registry freshness =="
$RUN python scripts/build_provenance_registry.py --check

echo "== scientific contract registry freshness =="
$RUN python scripts/build_contract_registry.py --check

echo "== documentation gate =="
bash scripts/check_docs.sh

echo "== test-matrix (current interpreter; CI does 3.11/3.12/3.13) =="
$RUN pytest -m "not slow" -q

echo "== ml-integration =="
env -u VIRTUAL_ENV uv sync --locked --extra dev --extra ml
env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest tests/integration -q

echo "== wheel-smoke =="
env -u VIRTUAL_ENV uv build --wheel -o dist/
rm -rf /tmp/jaxstro-clean
env -u VIRTUAL_ENV uv venv /tmp/jaxstro-clean
env -u VIRTUAL_ENV uv pip install --python /tmp/jaxstro-clean/bin/python dist/*.whl
/tmp/jaxstro-clean/bin/python -c "import jaxstro; print(jaxstro.__name__, 'imports clean')"

echo "ALL LOCAL GATES PASSED"
