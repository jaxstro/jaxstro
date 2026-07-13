#!/usr/bin/env bash
# Reusable MyST content, route, link, and rendered-DOM gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PORT="${DOCS_APP_PORT:-4311}"
SERVER_PORT="${DOCS_SERVER_PORT:-4312}"
BASE_PATH="${BASE_URL:-}"
LOG_PATH="${TMPDIR:-/tmp}/jaxstro-myst-start-$$.log"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$LOG_PATH"
}
trap cleanup EXIT INT TERM

echo "== docs: scientific contract registry freshness =="
env -u VIRTUAL_ENV uv run --no-sync python \
  "$ROOT_DIR/scripts/build_contract_registry.py" --check

echo "== docs: scientific evidence index freshness =="
env -u VIRTUAL_ENV uv run --no-sync python \
  "$ROOT_DIR/scripts/build_evidence_index.py" --check

echo "== docs: strict static build =="
(
  cd "$ROOT_DIR/docs"
  myst build --html --ci --strict
)

if [[ ! -f "$ROOT_DIR/docs/_build/html/index.html" ]]; then
  echo "docs gate failed: docs/_build/html/index.html is missing" >&2
  exit 1
fi

echo "== docs: rendered DOM and route manifest =="
(
  cd "$ROOT_DIR/docs"
  exec myst start --port "$APP_PORT" --server-port "$SERVER_PORT"
) >"$LOG_PATH" 2>&1 &
SERVER_PID=$!

if ! env -u VIRTUAL_ENV uv run --no-sync python "$ROOT_DIR/scripts/check_docs_site.py" \
  --site "$ROOT_DIR/docs/_build/site" \
  --manifest "$ROOT_DIR/docs/route-manifest.json" \
  --base-url "http://localhost:$APP_PORT" \
  --base-path "$BASE_PATH"; then
  echo "== MyST server log ==" >&2
  tail -n 120 "$LOG_PATH" >&2
  exit 1
fi

echo "ALL DOCS GATES PASSED"
