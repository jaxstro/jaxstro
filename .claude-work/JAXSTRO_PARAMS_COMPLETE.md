# `jaxstro.params` — Completion Document

**Branch:** `feature/jaxstro-params`
**Date:** 2026-06-17
**Plan:** `docs/plans/2026-06-17-jaxstro-params-implementation.md` (Tasks 0–4)
**Design / decision:** `docs/plans/2026-06-17-jaxstro-params-design.md`, ADR-0009

`jaxstro.params` is an **Equinox-only** utility for gradient-based inference over a
*subset* of an Equinox model's parameters: free/fixed marking + a `PyTree ↔ flat-vector`
bridge + a transform (bijector) registry for unconstrained-space inference. No new core
dependency; optax/numpyro are validation-only, behind the `[ml]` extra.

---

## 1. API Summary

### `Parameterization` (`jaxstro.params.parameterization`)

A static, pure `eqx.Module` (hashable, JIT-friendly). Fields are all
`eqx.field(static=True)`: a boolean `free_spec` PyTree, a leaf-aligned
`transform_spec`, and pre-aligned `free_meta` (`(shape, bijector)` per free leaf).

| Member | Signature | Purpose |
|--------|-----------|---------|
| `from_where` | `(model, where, transforms=None)` | Front door. `where=lambda m: (m.a, m.b)` selects whole array leaves; `transforms` align 1:1 with the **`where` selection**. |
| `from_filter` | `(model, free_spec, transforms=None)` | Escape hatch. Explicit boolean spec; `transforms` align with free leaves in PyTree order. |
| `to_vector` | `(model) -> Float[n]` | Free leaves → unconstrained ℝⁿ (per-leaf `bijector.inverse`, then ravel). |
| `from_vector` | `(model, vec) -> model` | Unconstrained ℝⁿ → physical model (unravel → `bijector.forward` → `eqx.combine`). Differentiable in `vec`. |
| `log_det_jacobian` | `(vec) -> scalar` | Σ over free scalar entries of `bijector.forward_log_det_jacobian` — change-of-variables term for unconstrained-space sampling. |

**Guarantees:** round-trip identity (`from_vector(m, to_vector(m)) == m` on array leaves),
fixed-/static-leaf preservation, and full `jit`/`vmap`/`grad` compatibility.

**Key design point (leaf-aligned transforms):** the flat vector is ordered by
**PyTree-leaf order**, not `where`-tuple order. Bijectors are lowered onto an
all-`Identity` template by riding the *same* `eqx.tree_at` lowering as `free_spec`, so
each bijector travels with its leaf regardless of selection-tuple ordering (regression
locked by `test_transform_follows_leaf_not_tuple_order`).

### Bijectors (`jaxstro.params.transforms`) — numpyro/TFP `forward: ℝ → physical`

| Bijector | `forward(u)` | Use | `forward_log_det_jacobian(u)` |
|----------|--------------|-----|-------------------------------|
| `Identity` | `u` | unconstrained | `0` |
| `Exp` | `eᵘ` | `x > 0` (`r_h`, `mass`) | `u` |
| `Softplus` | `log(1+eᵘ)` | gentle `x > 0` | `log σ(u)` |
| `Sigmoid(lo, hi)` | `lo + (hi−lo)·σ(u)` | `lo < x < hi` (e.g. `0<Q<1`) | `log(hi−lo) + log σ(u) + log σ(−u)` |

All log-Jacobians analytic and float64-stable (`jax.nn.softplus`, `jax.nn.log_sigmoid`),
each grad-checked against autodiff of `forward`.

### Usage sketch
```python
from jaxstro.params import Parameterization
from jaxstro.params.transforms import Exp, Sigmoid

param = Parameterization.from_where(
    model, where=lambda m: (m.r_h, m.Q), transforms=(Exp(), Sigmoid(0.0, 1.0)))
vec  = param.to_vector(model)                 # physical → unconstrained ℝⁿ
m2   = param.from_vector(model, vec)          # round-trips
g    = jax.grad(lambda v: loss(param.from_vector(model, v)))(vec)
ldj  = param.log_det_jacobian(vec)            # CoV term for numpyro
```

---

## 2. Zodiax-vs-ours rationale (ADR-0009)

Every ecosystem package doing gradient-based inference re-derives the same plumbing: mark
which leaves of an Equinox model are free vs fixed, flatten the free ones to a 1-D vector
for an optimizer/sampler, unflatten back. The dLux ecosystem solves this with the
**Zodiax** dependency. We **do not** vendor Zodiax (or abcdLux):

- **Equinox is enough.** Native `partition`/`filter`/`tree_at` + `jax.flatten_util.ravel_pytree`
  give the same selective-gradient capability with **zero new deps** — matching the
  thin-foundation posture (ADR-0001/0002).
- **Typed selectors > stringly-typed paths.** Our front door is a `where`-callable
  (`lambda m: (m.r_h, m.Q)`) that is IDE-checked and refactor-safe, versus Zodiax's
  string path leaves.
- **The genuinely-missing reusable piece is the bridge + transforms.** optax already
  operates on the free partition directly and numpyro only needs the flat vector, so the
  optax/numpyro **glue stays out of core** (YAGNI) — demonstrated only in the validation
  script under the `[ml]` extra.
- **Transforms earn v1 inclusion.** Bounded-positive params (`r_h>0`, `mass>0`, `0<Q<1`)
  are everywhere; the log-Jacobian is better done once, correctly, grad-checked, than
  re-derived per package.
- **Foundational for Informax** (OED / Fisher information over the free vector).

---

## 3. Per-package adoption guide

`jaxstro.params` sits **downstream-facing**: progenax/gravax/fluxax import it (one-way
arrow — `jaxstro.params` itself depends on nothing new).

- **progenax** — Fit profile/IMF parameters. Mark a free leaf the observable depends on
  (e.g. `PlummerProfile.a`, an IMF slope, total mass) with `Exp` (positive) or `Sigmoid`
  (bounded), build a loss on `density` / `enclosed_mass_fraction`, and drive optax/numpyro
  through `to_vector`/`from_vector` (+ `log_det_jacobian` for sampling). This is the
  validation target below. **Caveat:** if the parameter you want is a *source* of a cached
  derived leaf (e.g. `r_h` → cached `a`), free the derived leaf instead — `from_vector`
  does not re-run `__init__` (see §8).
- **gravax** — Inference over IC parameters (e.g. cluster `r_h`, virial ratio `Q`) feeding
  an integrator: `param.from_vector` inside the loss reconstructs the IC `eqx.Module`,
  then the differentiable integrator + likelihood close the loop. `Sigmoid(0,1)` for `Q`,
  `Exp` for radii/masses.
- **fluxax** — Tier-1 PSF-coefficient fitting: mark the PSF/background coefficient leaves
  free (often `Softplus`/`Sigmoid` to keep coefficients positive/bounded) and optimize the
  flat vector against a pixel likelihood.
- **Informax (future)** — Consumes the flat-vector Jacobian/Hessian bridge directly for
  optimal experimental design / Fisher information over the free vector.

Pattern for all: keep the bridge in the loss (`m = param.from_vector(model, vec)`), pass
`vec` to optax/numpyro, add `param.log_det_jacobian(vec)` as the CoV term when sampling.

---

## 4. Validation results

**Script:** `validation/validate_params.py` — runs whichever of optax / numpyro is
installed (degrades gracefully; aborts only if *both* absent). It best-effort-injects
`../progenax/src` onto `sys.path`, so it uses the **real** `progenax.PlummerProfile`
in any env where progenax's own deps (diffrax, …) are present, and falls back to a
self-contained toy `eqx.Module` otherwise (printing which path ran).

The validation was run **two ways** to cover both the real model and both backends:

### (A) Real `progenax.PlummerProfile` — run from progenax's env (diffrax + optax present)

```
======================================================================
VALIDATION TARGET: progenax.PlummerProfile (free: scale radius a > 0)
======================================================================

param        true        optax      numpyro     abs_err     rel_err
-------------------------------------------------------------------
a         1.22627     1.226273          n/a    0.00e+00    0.00e+00

FD-vs-AD gradient check (loss): max rel error = 1.246e-10

numpyro recovery : SKIP (numpyro not installed)
grad check       : PASS (max rel = 1.25e-10, tol 1e-5)
optax recovery   : PASS (|abs err| = 0.00e+00, tol 1e-3)

OVERALL: PASS
```

optax Adam over the unconstrained `Exp` vector recovers the true scale radius
`a = 1.22627` (the value implied by a Plummer profile with `r_h = 1.6`) to abs err
`0.00e+00`, with a machine-precision grad-check. numpyro is skipped because progenax's
env does not ship it (graceful degradation).

> **Why fit `a`, not `r_h`?** `PlummerProfile` stores **both** `r_h` and its derived
> scale radius `a` as separate array leaves (`a` computed from `r_h` in `__init__`).
> `enclosed_mass_fraction` reads `a`. The `params` bridge `from_vector` *replaces leaves*
> — it does **not** re-run `__init__` — so freeing only `r_h` leaves the cached `a` stale
> and the observable's gradient w.r.t. `r_h` is exactly zero (verified: optax stayed at
> init, grad = 0). Freeing `a` (the leaf the observable depends on) recovers correctly.
> See Lesson in §8 — this is the load-bearing adoption caveat.

### (B) Toy fallback — run from jaxstro's `[ml]` env (optax **and** numpyro)

```
param        true        optax      numpyro     abs_err     rel_err
-------------------------------------------------------------------
r_h       1.60000     1.600000     1.600040    0.00e+00    0.00e+00

FD-vs-AD gradient check (loss): max rel error = 4.450e-11

grad check       : PASS (max rel = 4.45e-11, tol 1e-5)
optax recovery   : PASS (|abs err| = 0.00e+00, tol 1e-3)
numpyro recovery : PASS (rel err = 2.50e-05, tol 5e-2)

OVERALL: PASS
```

The toy `_ToyPlummer` computes its scale radius **lazily** inside the observable, so
`r_h` *is* the leaf the observable depends on and is fit directly. Both backends recover
the truth: optax `r_h = 1.600000` (abs err `0.00e+00`); numpyro NUTS (400 warmup + 600
samples, `Normal(0,5)` prior + `log_det_jacobian` CoV term) posterior-mean
`r_h = 1.600040` (rel err `2.50e-05`). FD-vs-AD grad check at the non-converged start
point: max rel error `4.45e-11` (machine precision under x64).

---

## 5. Test results (per tier)

Full suite: **442 passed** (`env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest -q`),
up from the 441 baseline by the one new integration test.

| Tier | File(s) | Notes |
|------|---------|-------|
| unit | `tests/unit/test_params_parameterization.py` | round-trip, fixed-leaf, from_where==from_filter, vector length, empty set, grad, jit/vmap |
| unit | `tests/unit/test_params_transforms.py` | bijector round-trip + log-det vs autodiff + bounds |
| unit | `tests/unit/test_params_transformed.py` | transformed round-trip/bounds, log-det sum, leaf-not-tuple-order regression |
| integration | `tests/integration/test_params_optax.py` | **real optax loop** recovers a positive scalar to 1e-3 (skips cleanly via `pytest.importorskip("optax")`) |

Quality gates: `ruff check src/` → **All checks passed**;
`mypy src/jaxstro` → **Success: no issues found in 27 source files**.

---

## 6. Sibling smoke-tests (new `jaxstro.params` must not break siblings)

| Sibling | Result |
|---------|--------|
| progenax | **ok** (`import progenax`) |
| fluxax | **ok** (`import fluxax`) |
| gravax | **error — pre-existing env issue, NOT a regression**: gravax's own `.venv` has neither gravax nor jaxstro installed (`ModuleNotFoundError: No module named 'jaxstro'` on `from jaxstro.units import STELLAR`). `jaxstro.params` is purely additive and cannot have caused this — gravax cannot even resolve jaxstro. |

---

## 7. Files changed (Task 4)

- `tests/integration/test_params_optax.py` — new (optax integration test)
- `validation/validate_params.py` — new (CLI validation: optax + numpyro + grad-check)
- `README.md` — added `jaxstro.params` usage section + API-reference entry
- `.claude-work/JAXSTRO_PARAMS_COMPLETE.md` — this document

---

## 8. Lessons / notes

- **Grad-check at a non-converged point.** Evaluating FD-vs-AD at the optimizer's
  *minimum* gives a meaningless `0/0` relative error (both AD and FD ≈ 0). The check must
  run where the gradient is non-negligible (the start point) — fixed in the script.
- **`where` returning a single leaf + transforms.** When `transforms` are supplied,
  `where` must return a **tuple** (`lambda m: (m.r_h,)`), since the transform lowering
  rides `eqx.tree_at(where, …, replace=(bij,))` which expects a tuple to match.
- **⚠ Cached derived leaves go stale through the bridge (load-bearing caveat).**
  `from_vector` *replaces leaves*; it does **not** re-run the model's `__init__`. If a
  model caches a derived quantity as a *separate* leaf — as `PlummerProfile` does with its
  scale radius `a` (computed from `r_h` at construction) — then freeing only the **source**
  leaf (`r_h`) leaves the derived leaf (`a`) stale, and any observable that reads the
  derived leaf has **zero gradient** w.r.t. the source. This was caught by validating
  against the real progenax model (the toy, which derives `a` lazily, masked it). **Rule:**
  free the leaf the observable actually depends on (here `a`), or use models that derive
  such quantities lazily from stored leaves. A future `jaxstro.config` `from_config` layer
  (ADR-0010) — which rebuilds through the constructor — is the complementary path when you
  want to fit the *source* parameter of a cached derivation.
