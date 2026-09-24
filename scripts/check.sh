#!/usr/bin/env bash
# Full local release gate, split into named stages.
#   bash scripts/check.sh              run every stage in order
#   bash scripts/check.sh docs ml      run the named stages only
# The full-gate workflow runs each stage as its own parallel job, so local and
# CI run the same commands. PR and push CI (tests.yml) is a smaller fast gate.
# Run from repo root. Any failure aborts (set -e).
set -euo pipefail
RUN="env -u VIRTUAL_ENV uv run --no-sync"
STAGES=(static docs tests-unit tests-integration tests-validation ml distribution)

sync_dev() {
  env -u VIRTUAL_ENV uv sync --locked --extra dev --group benchmark --group reference
}

stage_static() {
  env -u VIRTUAL_ENV uv lock --check
  sync_dev
  $RUN ruff check src/ tests/ scripts/ laboratory/ examples/
  $RUN ruff format --check src/ tests/ scripts/ laboratory/ examples/
  $RUN mypy src/jaxstro
  $RUN python scripts/build_provenance_registry.py --check
}

# The contract, evidence-index, and workflow registry freshness checks run
# inside check_docs.sh, which Pages also runs on its own.
stage_docs() {
  sync_dev
  npm ci --ignore-scripts
  bash scripts/check_docs.sh
}

stage_tests() {
  sync_dev
  $RUN pytest -m "not slow" -q "tests/$1"
}

# Only tests that import an [ml] package gain coverage here; the rest of the
# integration tier runs in tests-integration.
stage_ml() {
  env -u VIRTUAL_ENV uv sync --locked --extra dev --extra ml
  env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest tests/integration/test_params_optax.py -q
}

stage_distribution() {
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
}

run_stage() {
  case "$1" in
    static) stage_static ;;
    docs) stage_docs ;;
    tests-unit) stage_tests unit ;;
    tests-integration) stage_tests integration ;;
    tests-validation) stage_tests validation ;;
    ml) stage_ml ;;
    distribution) stage_distribution ;;
  esac
}

if (($# == 0)); then
  set -- "${STAGES[@]}"
fi
for stage in "$@"; do
  if [[ " ${STAGES[*]} " != *" $stage "* ]]; then
    echo "unknown stage: $stage (stages: ${STAGES[*]})" >&2
    exit 2
  fi
done
for stage in "$@"; do
  echo "== $stage =="
  run_stage "$stage"
done
echo "GATES PASSED: $*"
