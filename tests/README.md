# jaxstro Test Suite

Release-grade validation for the jaxstro foundation library — physical constants, unit
systems, coordinate transforms, spatial algorithms, AD-safe numerics, and the
`jaxstro.params` parameterization layer.

Counts drift as the suite grows — **see CI or `pytest --co` for the live number.**

## Quick Start

All commands run from the repo root. Use the canonical invocation that bypasses the active
virtualenv and skips re-syncing the locked environment:

```bash
# Full suite
env -u VIRTUAL_ENV uv run --no-sync pytest -q

# Include the optional ML extra (params/optax integration)
env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest -q

# Skip slow tests (the fast gate)
env -u VIRTUAL_ENV uv run --no-sync pytest -m "not slow" -q
```

## Test Architecture

The suite is organized into three tiers. Every test module lives under exactly one tier
directory, and `tests/conftest.py` auto-applies the matching marker from the test's path —
so the marker and the directory are always in sync (a structural guard,
`tests/validation/test_suite_structure.py`, enforces this).

| Tier | Purpose |
|------|---------|
| **unit** | Fast isolated functional correctness — shapes, bounds, round-trips, exact values. |
| **integration** | Cross-module / JAX-transform behavior — `jit`, `grad`, `vmap`, parity. |
| **validation** | Scientific validation — FD-vs-AD numerical truth, convergence, structural guards. |

See `pytest --co` for the live per-tier count.

## Selecting Tests

Two equivalent idioms — by marker or by directory:

```bash
# By marker (works regardless of where a file lives)
env -u VIRTUAL_ENV uv run --no-sync pytest -m unit -q
env -u VIRTUAL_ENV uv run --no-sync pytest -m integration -q
env -u VIRTUAL_ENV uv run --no-sync pytest -m validation -q

# By directory
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit -q
env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration -q
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation -q

# Count collected tests in a tier
env -u VIRTUAL_ENV uv run --no-sync pytest -m unit -q --co | grep -c "::"
```

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options]` and enforced
with `--strict-markers`, so an unregistered marker is an error rather than a silent typo.

## Adding a Test

1. Put the module under the correct tier directory (`tests/unit/`,
   `tests/integration/`, or `tests/validation/`).
2. Do **not** hand-apply a tier marker — the `conftest.py` path hook applies it for you.
3. Run the structural guard to confirm placement:
   ```bash
   env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_suite_structure.py -q
   ```

## References

- `pyproject.toml` — `[tool.pytest.ini_options]` markers + `--strict-markers`.
- `tests/conftest.py` — float64 setup + path→marker auto-marking.
- `tests/validation/test_suite_structure.py` — regression guard for the 3-tier layout.
