# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Overview

jaxstro is the shared **foundation / numerics** library for a differentiable
astrophysics ecosystem built on JAX. It provides physical constants, unit systems,
coordinate transforms, AD-safe numerical primitives (safe math, root-finding,
quadrature, integration, interpolation, linear algebra), spatial algorithms, and a
selective parameter-inference bridge (`jaxstro.params`) for higher-level packages
(gravax, progenax, fluxax, startrax, stellax, nebulax, ...).

**This is infrastructure, not physics.** There are no simulations here — just the
shared building blocks that every downstream package relies on. Because everything
flows through jaxstro, a silent numerical bug here (a NaN gradient, a wrong sentinel,
a stale constant) propagates into every package. **Correctness > speed.**

**Status:** v0.1.0 release-hardening (Phase C). 3-tier test suite (unit / integration /
validation); see `pytest --co` or CI for the live count. Apache-2.0, `py.typed`,
hatchling build, `uv.lock` pinned.

**Design principles:**

- **Infrastructure only** — no domain-specific physics simulations.
- **JAX-native** — every public function must work under `jit`, `vmap`, `grad`.
- **AD-safe** — primitives are written so gradients are finite and correct, not just
  the forward value (see "AD-safe numerics" below).
- **Minimal core deps** — `jax`, `jaxlib`, `equinox`, `jaxtyping` (equinox is a core
  dep per ADR-0002; optax/numpyro live behind the `[ml]` extra).
- **Provenance** — every physical constant cites its authority (CODATA / IAU / a paper).

**Docs:** a mystmd Diátaxis site lives in `docs/` (`cd docs && myst build`). The
AD-safe-numerics thesis and per-method theory pages are there; ADRs are ported to
`docs/30-decisions/`.

## Quick Commands

Use **uv** (the project `.venv`); do **not** use conda. `env -u VIRTUAL_ENV` avoids an
outer-venv clash and `--no-sync` runs against the installed env without re-locking.

```bash
# Install (core + dev). The ML extra (optax/numpyro) is only needed for the params
# inference-integration tests; the params bridge itself needs no new core dep.
env -u VIRTUAL_ENV uv pip install -e ".[dev]"
env -u VIRTUAL_ENV uv pip install -e ".[dev,ml]"   # adds optax/numpyro

# THE GATE (run from repo root /Users/anna/projects/jaxstro-dev/jaxstro):
env -u VIRTUAL_ENV uv run --no-sync pytest -q              # full suite
env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest -q   # incl. params/ml tier

# Tier selection — by marker (auto-applied from path) or by directory:
env -u VIRTUAL_ENV uv run --no-sync pytest -m unit -q          # ≡ tests/unit/
env -u VIRTUAL_ENV uv run --no-sync pytest -m integration -q   # ≡ tests/integration/
env -u VIRTUAL_ENV uv run --no-sync pytest -m validation -q    # ≡ tests/validation/
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit -q       # same, by path
env -u VIRTUAL_ENV uv run --no-sync pytest -m "not slow" -q    # fast inner loop

# Full local gate (mirrors the dormant CI: lock-check, lint, mypy, test matrix slice,
# ml-integration, wheel-smoke). This is the real Phase-C gate — run it before a commit:
bash scripts/check.sh

# Lint / format / type (any task touching src/ or tests/):
env -u VIRTUAL_ENV uv run --no-sync ruff check src/ tests/
env -u VIRTUAL_ENV uv run --no-sync ruff format --check src/ tests/
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro

# Docs:
cd docs && myst build      # expect 0 content warnings (broken xref == defect)
```

**Never weaken a test or tolerance to make it pass — fix the root cause.** A failing
FD-vs-AD grad-check or parity test is a real numerical defect, not a flaky test.

## Architecture

```text
src/jaxstro/
├── __init__.py          # Exports: constants, units, astrometry, numerics, coords, params
├── jaxconfig.py         # enable_high_precision() for float64
├── constants.py         # CGS physical constants (CODATA 2018, IAU 2015, Oke & Gunn 1983)
├── units.py             # UnitSystem dataclass + predefined systems + G property
├── astrometry.py        # Astrometric constants (K_PROPER_MOTION, MAS_PER_RAD, ...)
├── coords.py            # Coordinate transforms (sky_tangent, galactic, spherical, parallax)
├── numerics/
│   ├── types.py         # Type aliases (Array, ScalarFn)
│   ├── stats.py         # safe_log/exp/div, logsumexp, convergence
│   ├── interpolation.py # interp1d, TabulatedFunction1D (pytree)
│   ├── rootfinding.py   # bisect, newton, newton_with_grad, newton_ppf (lax.scan)
│   ├── integration.py   # trapz, cumulative_trapz, simpson
│   ├── quadrature.py    # Gauss-Legendre / Gauss-Hermite (probabilists') + Hermite-e
│   ├── sampling.py      # inverse_cdf_draw and reparameterized sampling helpers
│   ├── checks.py        # Validation: all_finite, is_monotonic, in_range
│   ├── compensated.py   # Neumaier compensated summation
│   ├── linear_algebra.py# norm2, project_onto, condition_number
│   └── rng.py           # PRNG key helpers
├── params/              # Selective parameter-inference bridge (PyTree ↔ flat vector)
│   ├── parameterization.py # Parameterization.from_where / from_filter
│   └── transforms.py    # Identity, Exp, Softplus, Sigmoid bijectors (+ log-det-Jacobian)
├── spatial/
│   ├── morton.py        # Morton (Z-order) encoding/decoding, wyhash32
│   ├── grid.py          # assign_particles_to_bins, fill_bins (reservoir)
│   └── neighbor.py      # approx_knn_candidates, stencil-based gathering
└── testing/             # grad-audit engine (FD-vs-AD truth) reused across the ecosystem
```

## Units Convention

**Always use CGS** (cm, g, s, erg) as the base. Available systems:

- `CGS` — base (g, cm, s)
- `ASTRO_STELLAR` / `STAR` — stellar evolution (Msun, Rsun, Myr)
- `ASTRO_DYNAMICAL` / `STELLAR` — star clusters (Msun, pc, Myr)
- `ASTRO_PLANETARY` / `BINARY` — solar system (Msun, AU, yr)

Each `UnitSystem` has a `.G` property for the gravitational constant in that system.

### Ecosystem Units Policy (Defaults)

Downstream packages must define a package-level `DEFAULT_UNITS` constant. Core APIs
require explicit units or explicit `G`, or accept objects that carry units. Convenience
wrappers may accept `units=None` and resolve to `DEFAULT_UNITS`. **Do not use global
context managers or `get_G()` in core code.**

## Key Patterns

```python
# Enable float64 BEFORE creating any JAX arrays (the ecosystem standard).
from jaxstro.jaxconfig import enable_high_precision
enable_high_precision()  # jax_enable_x64=True, matmul_precision="highest"

# Constants and units
from jaxstro import constants as C, units as U
us = U.ASTRO_DYNAMICAL
m, r, t = us.from_cgs(mass_g, length_cm, time_s)
G = us.G  # G in this unit system

# Coordinate transforms
from jaxstro.coords import sky_tangent, galactic_to_equatorial
ra_dec = sky_tangent(positions_pc, distance_pc=1000.0)

# Differentiable root-finding (fixed-iteration lax.scan; jit/vmap/grad-safe)
from jaxstro.numerics import rootfinding
root = rootfinding.newton(lambda x: x**2 - 2.0, x0=1.5)

# Spatial binning
from jaxstro.spatial import assign_particles_to_bins, fill_bins, approx_knn_candidates
bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=16)
```

## AD-safe numerics (the core discipline)

jaxstro's value is that its primitives are **differentiable correctly**, not just
evaluable. The forward value can be right while the gradient is silently NaN or zero.
These three traps cause almost every gradient bug in the ecosystem:

### 1. The `where` trap (NaN backprops through the *dead* branch)

`jnp.where(cond, safe, unsafe)` evaluates **both** branches before selecting. Under
`jax.grad`, the derivative of the *unselected* branch still contributes, so a branch
that produces `inf`/`NaN` (e.g. `1/0`, `sqrt` of a negative, `log(0)`) poisons the
gradient with `NaN` even though the forward value looks fine.

**Fix — double-`where` / safe-denominator:** sanitize the input to the dangerous op so
the dead branch is finite too. Pattern used in `linear_algebra.condition_number` and
`project_onto`:

```python
singular = s_min == 0.0
s_min_safe = jnp.where(singular, 1.0, s_min)            # make the dead branch finite
return jnp.where(singular, jnp.inf, s_max / s_min_safe) # then select the real result
```

Because the input to the division is sanitized, the gradient through the live branch is
clean and the dead branch carries no `NaN`. (`newton`/`newton_ppf` use the same
safe-denominator idea: `dfx_safe = where(dfx == 0, 1.0, dfx)`; `newton_ppf` adds an
additive `pdf_floor`.)

### 2. Fixed-iteration `lax.scan`, never `while_loop`

Convergence loops (`lax.while_loop`) are **not differentiable**. Every iterative solver
here (`bisect`, `newton`, `newton_with_grad`, `newton_ppf`) runs a **fixed** number of
`lax.scan` steps and does not terminate early — so `jax.grad` flows through every
iteration and the iteration count is a static compile-time constant. When adding a new
iterative primitive, use `lax.scan` with a fixed `length`, not a convergence test.

### 3. Saturation is a silent gradient killer

`jnp.clip`, `jnp.min`/`jnp.max`, and `jnp.floor` zero the gradient wherever they
saturate (and `floor` is zero a.e.). A `clip(x, lo, hi)` used as a "safety floor" looks
harmless in the forward pass but returns **exactly zero gradient** for any saturated
input — which can silently stall an optimizer. Saturation is legitimate (e.g.
`newton_ppf` clips iterates to `[lo, hi]` to stay in support), but it must be a
*deliberate* choice, never a reflexive "fix" for a NaN. Prefer a smooth guard
(safe-denominator, `softplus`) when you need the gradient to keep flowing.

## Critical Invariants

Each statement below is **verified against the cited source** — do not paraphrase from
memory, and re-verify against source before changing or asserting any of these. These
are jaxstro's "load-bearing facts": downstream parity and grad-checks depend on them.

- **`cumulative_trapz` uniform path is dx-OUTSIDE.** For `x is None`, it computes
  `cumsum(0.5 * (y_left + y_right))` first and multiplies by the scalar `dx` exactly
  once at the end — *not* `cumsum(0.5 * dx * (...))`. This is the single-source ordering
  shared with progenax's `cumulative_trapezoid` and is byte-for-byte identical on shared
  inputs (the two forms differ by ~1 ulp; dx-outside is the chosen reconciliation). The
  non-uniform (`x` given) path carries per-interval `diff(x)` *inside* the cumsum because
  there is no single scalar to factor out.
  *(src/jaxstro/numerics/integration.py)*

- **Gauss-Hermite is probabilists', built from physicists' + √2 rescale, host-side.**
  `gauss_hermite_nodes(n)` calls `numpy.polynomial.hermite.hermgauss` (physicists',
  weight `e^{-x²}`) once at call time, then applies `g = √2·x`, `w → w/√π` to obtain the
  probabilists' rule for expectations under `N(0,1)` (weights sum to 1). NumPy appears
  here **by design** (sanctioned, setup-only constant generation); the nodes/weights are
  frozen constants and `jax.grad` flows through the *integrand values*, never the nodes.
  This construction is chosen so the output is byte-identical to the progenax
  `_gauss_hermite` rule it consolidates.
  *(src/jaxstro/numerics/quadrature.py)*

- **`condition_number` singular sentinel is `+inf`.** A rank-deficient matrix
  (`sigma_min == 0`, exact float zero — including the zero matrix) returns `+inf`,
  matching `numpy.linalg.cond`, so a caller guarding `cond > threshold` correctly rejects
  it. The result is **never NaN** (double-`where` avoids the `0/0`). A merely
  near-singular matrix returns a finite, very large value. Not differentiable at
  coincident singular values — use as a diagnostic, not inside a differentiated objective.
  *(src/jaxstro/numerics/linear_algebra.py)*

- **`bisect` has structurally-zero gradient w.r.t. function parameters.** Bisection's
  bracket update uses `jnp.sign(...)` comparisons, which are piecewise-constant, so
  `d(root)/d(param)` is structurally zero for parameters captured inside `f`. For
  `d(root)/d(param)` use `newton` / `newton_with_grad` / `newton_ppf` (smooth Newton
  steps flow the parameter gradient through every iteration). Bisection is still fine for
  the forward root and for `d(root)/d(bracket)`.
  *(src/jaxstro/numerics/rootfinding.py)*

- **`params.from_vector` replaces leaves and does NOT re-run `__init__`.** It does
  `eqx.combine(forward(unravel(vec)), fixed)` — it swaps the free leaves' values in
  place; it never reconstructs the Equinox module, so any **derived/cached leaf computed
  in `__init__`** (e.g. a Plummer scale radius `a` cached from `r_h`) will go *stale*
  when you fit only its source leaf. **Fit the leaf the observable actually reads.** If
  the observable reads the cached derived leaf, either mark that leaf free, or restructure
  the model so the derived quantity is recomputed at use-time rather than cached.
  *(src/jaxstro/params/parameterization.py)*

## Provenance Discipline

**Every physical constant cites its authority — never assert a value from memory.**
Verify against `src/jaxstro/constants.py`, which carries the citation inline for each
constant. Authorities in use:

- **CODATA 2018** — Tiesinga et al. (2021), Rev. Mod. Phys. 93, 025010 — fundamental
  constants (`G_CGS`, `C_CGS`, `K_B`, `SIGMA_SB`, `ALPHA_FS`, `E_ESU`, `R_E`, `SIGMA_T`,
  `R_GAS`, particle masses, ...). Example invariant: `A_RAD = 7.565733250e-15`, derived
  exactly as `4·SIGMA_SB / C_CGS` from the CODATA-2018 Stefan–Boltzmann constant (the
  old `7.565767e-15` was a 4σ-discrepant stale value — see CHANGELOG).
- **IAU 2015 B3** — nominal solar parameters (`MSUN_G`, `RSUN_CM`, `LSUN_ERG_S`);
  **IAU 2012 B2** — astronomical unit. The year is the **Julian** year
  (365.25 d × 86400 s = 31 557 600 s exactly), *not* the tropical year — the provenance
  comment in `constants.py` is load-bearing; preserve it.
- **Oke & Gunn (1983)**, ApJ 266, 713 — AB photometric zero point
  (`f_AB = 3631 Jy`), used by `PhotometricUnits` / the Jy↔AB constants.

When adding a constant: add the citation comment **in the same commit**, and prefer
deriving from a cited base constant (as `A_RAD` derives from `SIGMA_SB`) over typing a
literal.

## Module Summary

| Module | Purpose |
|--------|---------|
| `constants` | Physical constants in CGS (CODATA 2018 / IAU 2015 / Oke & Gunn 1983) |
| `units` | `UnitSystem` dataclass with conversions and `.G` property |
| `astrometry` | Astrometric constants (`K_PROPER_MOTION`, `MAS_PER_RAD`) + `PhotometricUnits` (Jy/AB) |
| `coords` | Coordinate transforms (sky-tangent, galactic, spherical, parallax, proper motion) |
| `numerics` | AD-safe primitives: stats, rootfinding, quadrature, integration, interpolation, linalg, sampling, checks, compensated |
| `params` | Selective parameter-inference bridge: `Parameterization` + Identity/Exp/Softplus/Sigmoid bijectors |
| `spatial` | Morton encoding, grid binning, neighbor candidate gathering |
| `testing` | FD-vs-AD grad-audit engine reused for ecosystem-wide grad-checks |

## Public API

Exported from `jaxstro.__init__` and submodules:

- **constants** — `G_CGS`, `C_CGS`, `K_B`, `SIGMA_SB`, `A_RAD`, `ALPHA_FS`, `E_ESU`,
  `R_E`, `SIGMA_T`, `R_GAS`, `MSUN_G`, `RSUN_CM`, `LSUN_ERG_S`, `PC_CM`, `AU_CM`, ...
- **units** — `UnitSystem`; systems `CGS`, `ASTRO_STELLAR`/`STAR`,
  `ASTRO_DYNAMICAL`/`STELLAR`, `ASTRO_PLANETARY`/`BINARY`; each has `.G`, `.to_cgs`,
  `.from_cgs`, `.velocity_scale_km_s`.
- **coords** — `sky_tangent`, `galactic_to_equatorial`, `equatorial_to_galactic`,
  `cartesian_to_spherical`, `spherical_to_cartesian`, `compute_parallax`,
  `compute_proper_motions`.
- **numerics.stats** — `safe_log`, `safe_exp`, `safe_div`, `logsumexp`.
- **numerics.rootfinding** — `bisect`, `newton`, `newton_with_grad`, `newton_ppf`.
- **numerics.integration** — `trapz`, `cumulative_trapz`, `simpson`.
- **numerics.quadrature** — `gauss_legendre_nodes`, `gauss_hermite_nodes`,
  `hermite_e_basis`, `hermite_coefficients`.
- **numerics.linear_algebra** — `norm2`, `project_onto`, `condition_number`.
- **params** — `Parameterization` (`.from_where`, `.from_filter`, `.to_vector`,
  `.from_vector`, `.log_det_jacobian`); bijectors `Identity`, `Exp`, `Softplus`,
  `Sigmoid`.
- **spatial** — `morton_encode_3d`, `morton_decode_3d`, `assign_particles_to_bins`,
  `fill_bins`, `approx_knn_candidates`.

## Test Structure

3-tier architecture; markers are auto-applied from each test's path by
`tests/conftest.py` and declared `--strict-markers` in `pyproject.toml`.

```text
tests/
├── unit/          fast isolated correctness (shapes, bounds, round-trips):
│                  constants, units, astrometry, coords, numerics, quadrature,
│                  linear_algebra, checks, photometric, rng, sampling, spatial,
│                  jaxconfig + params (parameterization / transforms)
├── integration/   cross-module / JAX-transform tests (jit, grad, vmap, parity):
│                  grad_audit, integration_parity, params_optax
└── validation/    scientific/numerical truth: grad_checks (FD-vs-AD), suite-structure guard
```

See `tests/README.md` and `pytest --co` for the live counts (kept out of this file to
avoid drift).

## Common Issues

- **NaN gradient, finite forward value** → the `where` trap (§AD-safe #1). Sanitize the
  input to the dangerous op (double-`where`), don't just guard the output.
- **Gradient is exactly zero where you expected signal** → saturation (§#3, `clip`/`min`/
  `max`/`floor`) or differentiating `bisect` w.r.t. function params (use `newton_ppf`).
- **`condition_number` returned `+inf`** → the matrix is *exactly* singular; that's the
  intended sentinel, not a bug. A near-singular matrix returns a finite large value.
- **A fit converges but the observable doesn't move** → cached-derived-leaf staleness:
  `from_vector` didn't re-run `__init__`; fit the leaf the observable reads.
- **`cumulative_trapz` differs from progenax at ~1 ulp** → you (or a caller) used the
  dx-inside ordering. The dx-outside path is canonical; reconcile to it.
- **`mypy` / `ruff` failures after adding jaxtyping shapes** → `pyproject.toml` ignores
  `F722`/`F821` for the `"N 3"` annotation pattern; keep new annotations in that style.
- **No float64** → `enable_high_precision()` must run before the first JAX array is
  created (downstream packages call it at import; tests configure it in `conftest.py`).

## Debugging Checklist

- [ ] **Gradient NaN/zero?** Run a FD-vs-AD grad-check (`jaxstro.testing` engine) and
      audit every `where` for a poisoning dead branch and every `clip`/`floor` for
      saturation. Verify the iterative path is `lax.scan` (fixed length), not
      `while_loop`.
- [ ] **Wrong number?** Check the constant's provenance comment in `constants.py`; for
      quadrature/integration, confirm the dx-outside ordering and the √2 Gauss-Hermite
      rescale against the Critical Invariants above.
- [ ] **Parity drift with a downstream package?** Compare against the single-source
      invariant (cumulative_trapz dx-outside; probabilists' Gauss-Hermite) — jaxstro is
      the source of truth.
- [ ] **`params` round-trip fails?** Confirm the bijector support matches the leaf
      (`Exp`/`Softplus` for `>0`, `Sigmoid(lo,hi)` for bounded) and that you're fitting a
      *primary* leaf, not a cached derived one.
- [ ] **Recompilation / shape errors?** jaxtyping shape strings + static args (`axis`,
      `keepdims`, `n` in quadrature) must stay static.

## Adding New Code

- Keep functions small (~50 LOC preferred, ~100 cap) and modules cohesive (~300 LOC
  preferred, ~500 cap); split along a natural seam before exceeding.
- Domain-agnostic only — full ODE/optimization/linear solvers belong in
  diffrax / optimistix / lineax, not here.
- Ensure `jit`, `vmap`, `grad` compatibility; **add a FD-vs-AD grad-check** for any new
  differentiable primitive.
- Spatial algorithms → `spatial/`; coordinate transforms → `coords.py`; bijectors →
  `params/transforms.py`.
- Every physical constant carries its provenance citation in the same commit.

## Definition of Complete

A change to jaxstro is **not** complete until:

1. **Tests** — unit + (where relevant) integration/validation tests pass; new
   differentiable code has an FD-vs-AD grad-check; edge cases covered. No weakened
   tolerances.
2. **Gates green** — `bash scripts/check.sh` passes end-to-end (lock-check, `ruff check`,
   `ruff format --check`, `mypy src/jaxstro`, the test matrix slice, ml-integration,
   wheel-smoke).
3. **Docs** — `cd docs && myst build` is clean (0 content warnings); any new public
   primitive is reflected in the API/theory pages and the Public API section here.
4. **Provenance** — every new constant/coefficient cites its authority inline.
5. **No drift** — Critical Invariants above still hold (re-verify against source if you
   touched integration / quadrature / linear_algebra / rootfinding / params).

**This is research infrastructure: correctness, provenance, and gradient-safety over
speed.** Every numerical claim must be backed by a test or a cited source.

## Brain hub - this repo is a spoke of ~/brain (read-only from here)

- **Never edit `~/brain` from this session** - not hat homes, ADRs, configs,
  knowledge, or `_generated/`.
- **One write path home - the inbox, via capture:** use
  `brain "what happened - short, factual"`.
- **Cross-cutting insight:** use
  `brain "xref: <insight> - touches <other project / paper>"`.
  It becomes a brain concept and resurfaces there via `/brain-pack` (ADR-0019).
- **Full protocol + conventions:** read `~/brain/AGENTS.md` and `~/brain/guide/`
  before cross-session work (pull-only hub; spec -> session -> log handoffs,
  ADR-0018; modern mystmd if this is a MyST site).
- **Starting focused work here?** Pull context with `/brain-pack jaxstro`.
- **Need papers/equations?** Start with the pack's Relevant literature and
  Equation-critical sources. Read source notes in `~/brain/knowledge/sources/`;
  verify exact equations/tables against `~/brain/knowledge/library/<bibkey>.pdf`;
  use `~/brain/knowledge/derived/equation-digests/` only when rows are verified;
  treat `~/brain/knowledge/raw/` as search-only. Capture needed source-note
  expansions with `brain "source-note update: <bibkey> - <what this package needs>"`.

<!-- brain-handshake: keep in sync with ~/brain/guide/how-to/set-up-a-project.md#spoke-stanza -->

<!-- brain-status-convention -->
## Brain status updates
When you make notable progress, hit a blocker, or set the next action, update this repo's `STATUS.md` (`next:` / `blocker:` / `due:` lines) — the brain pulls it into the portfolio dashboard + standup via `federate.py` (see `~/brain/work/meta/status-convention.md`). Brain stays pull-only: never hand-edit `~/brain`; capture events with `brain "…"`.
