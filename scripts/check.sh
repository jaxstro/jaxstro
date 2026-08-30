#!/usr/bin/env bash
# Full local release gate. PR CI is intentionally smaller; the scheduled/manual
# full-gate workflow mirrors these exhaustive checks across supported Python versions.
# Run from repo root. Any failure aborts (set -e).
set -euo pipefail
RUN="env -u VIRTUAL_ENV uv run --no-sync"

echo "== lock-check =="
env -u VIRTUAL_ENV uv lock --check

echo "== sync: development, benchmark, and reference gates =="
env -u VIRTUAL_ENV uv sync --locked --extra dev --group benchmark --group reference

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

echo "== scientific evidence index freshness =="
$RUN python scripts/build_evidence_index.py --check

echo "== research workflow registry freshness =="
$RUN python scripts/build_research_workflow_registry.py --check

echo "== documentation gate =="
npm ci --ignore-scripts
bash scripts/check_docs.sh

echo "== test-matrix (current interpreter; supported CI runtime is 3.13) =="
$RUN pytest -m "not slow" -q

echo "== ml-integration =="
env -u VIRTUAL_ENV uv sync --locked --extra dev --extra ml
env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest tests/integration -q

echo "== distribution artifacts =="
ARTIFACT_DIR="$(mktemp -d)"
WHEEL_VENV="$(mktemp -d)"
SDIST_VENV="$(mktemp -d)"
cleanup_distribution_artifacts() {
  rm -rf "$ARTIFACT_DIR" "$WHEEL_VENV" "$SDIST_VENV"
}
trap cleanup_distribution_artifacts EXIT

{
  env -u VIRTUAL_ENV uv --version
  $RUN python --version
  echo "hatchling==1.31.0"
} >"$ARTIFACT_DIR/build-provenance.txt"
env -u VIRTUAL_ENV uv build --python 3.13 -o "$ARTIFACT_DIR"

WHEEL_PATH=("$ARTIFACT_DIR"/*.whl)
SDIST_PATH=("$ARTIFACT_DIR"/*.tar.gz)
if [[ "${#WHEEL_PATH[@]}" -ne 1 || "${#SDIST_PATH[@]}" -ne 1 ]]; then
  echo "release gate failed: expected exactly one wheel and one sdist" >&2
  exit 1
fi

env -u VIRTUAL_ENV uv venv --python 3.13 "$WHEEL_VENV"
env -u VIRTUAL_ENV uv pip install --python "$WHEEL_VENV/bin/python" "${WHEEL_PATH[0]}"
env -u VIRTUAL_ENV uv run --no-sync python scripts/check_distribution.py \
  --wheel "${WHEEL_PATH[0]}" --sdist "${SDIST_PATH[0]}" \
  --python "$WHEEL_VENV/bin/python" --provenance "$ARTIFACT_DIR/build-provenance.txt"

env -u VIRTUAL_ENV uv venv --python 3.13 "$SDIST_VENV"
env -u VIRTUAL_ENV uv pip install --python "$SDIST_VENV/bin/python" "${SDIST_PATH[0]}"
env -u VIRTUAL_ENV uv run --no-sync python scripts/check_distribution.py \
  --wheel "${WHEEL_PATH[0]}" --sdist "${SDIST_PATH[0]}" \
  --python "$SDIST_VENV/bin/python" --provenance "$ARTIFACT_DIR/build-provenance.txt"

echo "ALL LOCAL GATES PASSED"
