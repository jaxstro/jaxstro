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
