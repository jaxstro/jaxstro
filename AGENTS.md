# AGENTS.md (Codex) - jaxstro

Guidance for Codex when working in `jaxstro`.

## Read First
- `CLAUDE.md`
- `README.md`
- `pyproject.toml`

## Units Policy
- `DEFAULT_UNITS`: `CGS` — jaxstro is the domain-agnostic foundation, so its default is the
  physics-pure base all `UnitSystem`s are built from. Downstream *domain* packages set their own
  `DEFAULT_UNITS` (e.g. gravax/progenax use `STELLAR` = `ASTRO_DYNAMICAL`).
- Core APIs require explicit units or explicit `G`.
- Convenience wrappers may accept `units=None` and resolve to `DEFAULT_UNITS`.

## JAX Rules
- Use `jax.numpy`, `jax.lax.scan`, `jax.jit`, `jax.vmap`.
- Keep utilities domain-agnostic and differentiable.

## Testing
- `pytest`
- `ruff check src/ && ruff format src/`
- `mypy`

## Brain hub

This repo is a spoke of `~/brain`. Full handshake in `CLAUDE.md` (Brain hub section) and
`~/brain/AGENTS.md`: pull-only hub, never edit `~/brain` from here, send notes home via
`brain "…"` (and `brain "xref: …"` for cross-cutting insights).

<!-- brain-handshake -->

<!-- brain-status-convention -->
## Brain status updates
When you make notable progress, hit a blocker, or set the next action, update this repo's `STATUS.md` (`next:` / `blocker:` / `due:` lines) — the brain pulls it into the portfolio dashboard + standup via `federate.py` (see `~/brain/work/meta/status-convention.md`). Brain stays pull-only: never hand-edit `~/brain`; capture events with `brain "…"`.
