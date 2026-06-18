# jaxstro.params — selective parameter-inference utility — design

**Status:** ratified in brainstorm with Anna (2026-06-17). Design approved; implementation next
(branch `feature/jaxstro-params` off main, after Phase B merged).
**Related ADRs:** `0009-jaxstro-params-selective-inference`, `0010-ecosystem-config-architecture`.
**Related:** `0001-thin-foundation-posture`, `0002-adopt-equinox-foundation`.

## Context

Every ecosystem package that does gradient-based inference re-derives the same plumbing: mark which
leaves of an Equinox model are *free* vs *fixed*, flatten the free ones into a 1-D vector for an
optimizer/sampler, and unflatten back. The dLux ecosystem solves this with **Zodiax**; we build our own
small, typed, **Equinox-only** version (no new core dependency — ADR-0001/0002). Concrete consumers:
progenax/gravax inference today, fluxax Tier-1 PSF-coefficient fitting, and the forthcoming **Informax**
(inference + optimal experimental design — OED needs exactly the flat-vector Jacobian/Hessian bridge).

## Decisions (ratified)

1. **Marking:** a `where`-callable selector (typed, IDE-checked — *not* Zodiax string paths) as the front
   door, with an explicit boolean filter-spec PyTree accepted as the low-level escape hatch. `where`
   selects **whole array leaves** in v1 (not sub-elements; YAGNI).
2. **Transforms:** included in v1 — a small bijector registry (Identity/Exp/Softplus/Sigmoid) with analytic
   log-Jacobians, for unconstrained-space inference of bounded params.
3. **optax/numpyro glue:** *deferred* from the core module (Equinox-only, zero optional deps); the
   validation script *demonstrates* a real optax loop and a tiny numpyro recovery under the `[ml]` extra.

## Module layout — `src/jaxstro/params/`

- `parameterization.py` — `Parameterization` (marking + PyTree↔vector bridge).
- `transforms.py` — bijector registry.
- `__init__.py` — exports `Parameterization`, the bijectors.

## Core API — `Parameterization`

Holds only a **static boolean filter-spec** (free/fixed mask) + a static **transform spec**, so it is a
tiny hashable jit-static object.

```python
class Parameterization(eqx.Module):
    free_spec: PyTree   = eqx.field(static=True)   # bool tree: True at free leaves
    transforms: tuple   = eqx.field(static=True)   # bijectors aligned with the where-selection

    @classmethod
    def from_where(cls, model, where, transforms=None): ...   # A: where=lambda m: (m.r_h, m.Q)
    @classmethod
    def from_filter(cls, model, free_spec, transforms=None): ...  # B: explicit bool tree

    def to_vector(self, model)        -> Float[Array, " p"]   # physical -> (inverse-transform) -> ravel
    def from_vector(self, model, vec) -> Model                # unravel -> (forward-transform) -> combine
    def log_det_jacobian(self, vec)   -> Float[Array, ""]     # sum of per-leaf fwd log-dets (numpyro CoV)
```

**Mechanics** (pure Equinox + `jax.flatten_util`):
- `from_where` lowers the `where`-lambda to a bool tree via `eqx.tree_at` over an all-False template — A is
  sugar over B.
- `to_vector` = filter free → apply each bijector's `inverse` (physical→ℝ) → `ravel_pytree(...)[0]`.
- `from_vector` = recompute the unravel from the free partition's structure → apply each bijector's
  `forward` (ℝ→physical) → `eqx.combine(free, fixed)`. No closures stored → stays a clean static PyTree.

**Invariants (→ first tests):**
- **Round-trip identity:** `from_vector(m, to_vector(m))` is `m` exactly (fixed leaves untouched).
- **Differentiable:** `jax.grad(λ vec: loss(from_vector(m, vec)))(vec)` correct (FD-checked).
- jit/vmap-safe: `vec` is the only traced array; structure is static.

## Transforms — `transforms.py`

numpyro/TFP convention: `forward(u)` maps **unconstrained ℝ → constrained physical**; `inverse` is the
reverse; `forward_log_det_jacobian` is **analytic** (not autodiff'd determinants). Each is a small
`eqx.Module`, float64-stable via `jax.nn.softplus`/`log_sigmoid`.

| Bijector | `forward(u)` | `fwd_log_det(u)` | covers |
|---|---|---|---|
| `Identity` | `u` | `0` | default |
| `Exp` | `exp(u)` | `u` | `mass>0`, `r_h>0` |
| `Softplus` | `softplus(u)` | `log_sigmoid(u)` | positivity (gentler) |
| `Sigmoid(lo,hi)` | `lo+(hi−lo)·σ(u)` | `log(hi−lo)+log_sigmoid(u)+log_sigmoid(−u)` | `0<Q<1`, bounded |

Transforms are passed **aligned with the `where` selection** (typed, tuple-ordered), default `Identity`.
`to_vector` applies `inverse` per free leaf before raveling; `from_vector` applies `forward` after
unraveling (so the reconstructed model always satisfies its bounds); `log_det_jacobian` sums the per-leaf
`forward_log_det_jacobian` for a correct numpyro posterior.

**Bijector invariants (→ tests):** round-trip `forward(inverse(x))==x`; analytic log-det matches
`jax.grad(forward)` (FD/AD); numerical stability at extreme `u`.

## Testing & validation (Definition of Complete)

**3-tier tests** (`tests/unit` + `tests/integration`):
- round-trip identity (exact, fixed-leaf preservation); `from_where ≡ from_filter`; edge cases (**empty
  free set**, **all-free**, **nested modules**); whole-array-leaf selection.
- gradient correctness FD-vs-AD through `loss(from_vector(m, vec))`; `jit` + `vmap` (batch of vectors).
- each bijector round-trip + analytic-log-det-vs-AD + stability; transformed `Parameterization`
  round-trips and the reconstructed model satisfies its bounds.
- integration: a real **optax** loop recovering a known value on a toy `eqx.Module`.

**Validation** `validation/validate_params.py` (test on **progenax**, not gravax): import a real progenax
Equinox model (IMF/profile), mark 1–2 free params, inject a known truth, recover it two ways — an optax
descent **and** a tiny **numpyro** chain (under `[ml]`, using `log_det_jacobian`) — printing
expected-vs-recovered + FD-vs-AD grad error. Clean fallback to a toy `eqx.Module` if progenax isn't
importable.

**Done =** 3-tier tests 100% green · validation quantitative recovery + grad-check · `.claude-work/`
completion doc (API, Zodiax-vs-ours rationale, per-package adoption guide) · docstrings + README usage ·
sibling import smoke-test · no new core dependency (optax/numpyro only behind `[ml]`).

## Build-vs-reuse: Zodiax

We do **not** vendor Zodiax (or abcdLux). Zodiax's value is selective-gradient plumbing over PyTrees, but
it uses **stringly-typed paths** (`"r_h"`) and is an extra dependency. Equinox's native
`partition`/`filter`/`tree_at` + `jax.flatten_util.ravel_pytree` give the same capability with **typed,
refactor-safe** selectors and zero new deps — matching the thin-foundation posture (ADR-0001).

## Out of scope (v1) / future

- Sub-element (per-index) freeing of a single array leaf — revisit on real need.
- A `jaxstro.config` `from_config`/`Configurable` convention (ADR-0010) — separate, optional `[config]`.
- optax/numpyro convenience helpers — add behind `[ml]` only when a pattern repeats across ≥2 packages.
