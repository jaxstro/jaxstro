# CLAUDE.md

Active engineering guidance for Jaxstro. Historical execution notes belong in
`STATUS.md`, development logs, plans, and git history—not in this file.

## Purpose and ownership

Jaxstro is the dependency-light foundation for a JAX-native differentiable
astrophysics ecosystem. It owns reusable constants, units, coordinates, numerical
primitives, spatial mechanics, spectral representation, parameter bridges,
provenance, and validation tooling.

It does not own domain simulations, scientific acceptance policy, filters or
photometry semantics, population models, stellar evolution, or many-body dynamics.
Downstream packages retain those responsibilities.

Correctness and honest derivative semantics take priority over speed. A finite
value, successful JAX transformation, or passing self-consistency check is not by
itself a scientific validation result.

## Read first

Before substantial changes, inspect:

- `AGENTS.md`
- `README.md`
- `STATUS.md`
- `pyproject.toml`
- the relevant source, tests, theory page, API reference, and validation anchors
- applicable ADRs under `docs/30-decisions/`

## Current package map

- `jaxstro.constants` — sourced CGS physical and photometric constants.
- `jaxstro.units` — current canonical ecosystem `UnitSystem` contract.
- `jaxstro.quantity` — implemented dimensional quantity evaluation; ecosystem
  adoption remains deferred pending downstream evidence.
- `jaxstro.astrometry`, `jaxstro.coords`, `jaxstro.geometry` — coordinate and
  geometric transformations with explicit singular boundaries.
- `jaxstro.numerics` — generic interpolation, integration, quadrature,
  rootfinding, distributions, linear algebra, ODE, optimization, operator,
  sampling, and special-function mechanics.
- `jaxstro.spatial` — discrete spatial indexing, approximate candidates, and
  exact fixed-radius pair mechanics.
- `jaxstro.spectra` — generic spectral coordinates, semantics, transformations,
  resampling, and prepared interpolation stencils; no photometric interpretation.
- `jaxstro.atmospheres` — catalog and artifact preparation plus evidence-gated
  atmosphere-spectrum evaluation.
- `jaxstro.params` — selective Equinox PyTree/vector parameter bridge; not an
  inference framework.
- `jaxstro.provenance` — deterministic runtime artifact manifests.
- `jaxstro.testing` — gradient audits, numerical ratchets, reports, and
  source-backed provenance cards; not scientific acceptance policy.
- `jaxstro.jaxconfig` — explicit float64/highest-matmul configuration.

## Commands

Use the repository `.venv` through uv. Remove an outer environment and never sync
implicitly during tests or tools:

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q
env -u VIRTUAL_ENV uv run --no-sync pytest -m unit -q
env -u VIRTUAL_ENV uv run --no-sync pytest -m integration -q
env -u VIRTUAL_ENV uv run --no-sync pytest -m validation -q
env -u VIRTUAL_ENV uv run --no-sync ruff check src tests scripts laboratory
env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests scripts laboratory
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
bash scripts/check_docs.sh
bash scripts/check.sh
```

Run focused tests first and expand proportionally. Never weaken a tolerance or
delete a contract test to obtain a pass.

## Units policy

- `DEFAULT_UNITS` is CGS because Jaxstro is the domain-agnostic foundation.
- Core APIs require explicit units or an explicit physical constant such as `G`.
- Convenience wrappers may accept `units=None` and resolve to `DEFAULT_UNITS`.
- Downstream domain packages define their own defaults.
- Do not add global unit context managers or hidden `G` lookup to core kernels.
- Physical constants cite their primary authority or derivation in the same
  change that introduces them.

## JAX and AD contracts

- Prefer `jax.numpy`, fixed-shape PyTrees, `jax.jit`, `jax.vmap`, and
  fixed-length `jax.lax.scan`.
- Do not use `jax.lax.while_loop` for differentiated iteration.
- Guard expensive scalar work with `jax.lax.cond`. Under `vmap`, a conditional
  can lower to select-style execution; use `lax.map` when physical per-lane
  skipping is part of the cost contract.
- Sanitize dangerous operations before selection. `jnp.where` evaluates both
  branch expressions, so an invalid dead branch can poison gradients.
- Treat clipping, floors, discrete indices, and sign decisions as explicit
  derivative boundaries rather than harmless safety operations.

Derivative targets must remain distinct:

- `newton` exposes sensitivity of its smooth finite executed iteration. This is
  not automatically the derivative of an ideal mathematical root.
- `safeguarded_bracketed_root` is value-first and branch-selected. It makes no
  implicit-root derivative claim.
- `implicit_bracketed_root` exposes a certified implicit derivative only when
  its uniqueness, smooth-branch, convergence, residual, bracket-width,
  finiteness, and conditioning gates pass. Independent finite differences
  validate this API; they are not performed inside each runtime solve.
- A JIT- or VMAP-compatible function is not necessarily differentiable in the
  scientifically intended sense.

New differentiable primitives require an independent analytic or central-FD
audit on their claimed smooth domain. Document nonsmooth and invalid domains.

## Load-bearing numerical invariants

Re-read the cited source before changing or repeating these claims.

### Cumulative trapezoid ordering

The uniform path in `src/jaxstro/numerics/integration.py` sums trapezoids before
multiplying by scalar `dx`. This dx-outside order is the canonical parity
contract. Nonuniform spacing keeps `diff(x)` inside the cumulative sum.

### Gauss-Hermite convention

`src/jaxstro/numerics/quadrature.py` creates probabilists' standard-normal nodes
and weights host-side from NumPy's physicists' rule using the documented square-
root-of-two rescaling. Nodes and weights are setup constants, not differentiated
inputs.

### Singular condition number

`condition_number` returns positive infinity for an exactly rank-deficient
matrix, never NaN. Its safe-denominator construction prevents dead-branch
poisoning. It is a diagnostic and is not smooth at repeated singular values.

### Root derivatives

Bisection and safeguarded sign-bracket decisions do not define a trustworthy
parameter derivative. Newton provides a finite executed-map derivative. Only the
separate certified implicit API claims the mathematical-root sensitivity, and it
fails closed when its assumptions or numerical gates fail.

### Finite power-law limit

The finite power-law normalization, log density, CDF, and PPF share smooth
removable-singularity kernels through `alpha = -1`. Do not restore an exact-value
branch whose forward value is correct but whose parameter derivative is wrong.

### Quantity adoption boundary

`jaxstro.units` remains canonical. `jaxstro.quantity` is an implemented evaluation
surface, not an approved ecosystem cutover. Migration requires downstream parity,
serialization, performance, ergonomics, and migration-cost evidence.

## Evidence and documentation

- Theory explains mathematics, assumptions, boundary behavior, and derivative
  meaning.
- API reference records signatures and ownership.
- Validation pages link claims to executable evidence.
- Source-backed provenance cards and runtime manifests are distinct evidence
  classes; neither substitutes for the other.
- Generated artifacts require deterministic emit/check or freshness gates.
- Documentation is designed around **predict → compute → audit → state the
  warranted claim**.
- Every measured result is reported in a table with metric identity, symbol,
  value, and units.

## Change discipline

- Work in the normal checkout unless the user explicitly requests a worktree.
- Preserve unrelated tracked and untracked changes.
- Use `apply_patch` for edits.
- Keep Jaxstro domain-agnostic and avoid duplicating established solver stacks.
- Add focused tests before implementation and commit coherent slices.
- Update `STATUS.md` when notable progress, a blocker, or the single next action
  changes.
- Before completion, run focused tests, Ruff, MyPy, evidence freshness, and the
  strict docs gate in proportion to the change.

## Brain hub

This repository is a spoke of `~/brain`, which remains pull-only from here.

- Never edit Brain directly from this session.
- Capture factual progress with `brain "what happened - short, factual"`.
- Capture cross-project insight with
  `brain "xref: <insight> - touches <other project / paper>"`.
- Pull focused context with `/brain-pack jaxstro`.
- See `~/brain/AGENTS.md` and `~/brain/guide/` for the full protocol.

<!-- brain-handshake: keep in sync with ~/brain/guide/how-to/set-up-a-project.md#spoke-stanza -->

<!-- brain-status-convention -->
When notable progress, a blocker, or the next action changes, update the
`next:` / `blocker:` / `due:` lines in `STATUS.md`. Brain pulls those fields into
the portfolio dashboard; it is not hand-edited from this repository.
