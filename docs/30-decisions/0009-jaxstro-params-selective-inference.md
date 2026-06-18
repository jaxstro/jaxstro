---
title: "ADR 0009 — jaxstro.params selective inference"
description: "Add an Equinox-only PyTree<->vector bridge plus a bijector registry, not a Zodiax dependency."
id: 0009
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 2
---

# 0009 — jaxstro.params: selective parameter-inference utility (Equinox-only, not Zodiax)

## Context

Every ecosystem package doing gradient-based inference re-derives the same plumbing: mark which leaves of an Equinox model are free vs fixed, flatten the free ones to a 1-D vector for an optimizer/sampler, unflatten back. dLux solves this with the Zodiax dependency. Consumers: progenax/gravax inference, fluxax Tier-1 PSF-coefficient fitting, and the forthcoming Informax (inference + OED, which needs the flat-vector Jacobian/Hessian bridge).

## Decision

Add **`jaxstro.params`** — a small, typed, **Equinox-only** utility (no new core dependency):

1. **Marking:** a `where`-callable selector (typed, IDE/refactor-safe) as the front door; an explicit boolean filter-spec PyTree as the low-level escape hatch. `where` selects **whole array leaves** in v1.
2. **PyTree↔flat-vector bridge** (`Parameterization.to_vector`/`from_vector`) built on `eqx.partition`/`tree_at` + `jax.flatten_util.ravel_pytree`. Round-trips exactly; jit/grad/vmap-safe.
3. **Transforms in v1:** a bijector registry (Identity/Exp/Softplus/Sigmoid) with analytic log-Jacobians (numpyro/TFP `forward: ℝ→physical` convention) for unconstrained-space inference of bounded params.
4. **optax/numpyro glue deferred** from the core module (Equinox-only); the validation script demonstrates optax + numpyro under an optional `[ml]` extra.

## Rationale

- **Do NOT vendor Zodiax/abcdLux:** Equinox's native `partition`/`filter`/`tree_at` + `ravel_pytree` give the same selective-gradient capability with **typed** selectors (vs Zodiax stringly-typed paths) and zero new deps — matching the thin-foundation posture (ADR-0001/0002).
- The genuinely-missing reusable piece is the **PyTree↔vector bridge + transforms**; optax already operates on the free partition directly, and numpyro only needs the flat vector — so glue stays out of core (YAGNI).
- Transforms earn v1 inclusion: bounded-positive params (`r_h>0`, `mass>0`, `0<Q<1`) are everywhere; the log-Jacobian is better done once, correctly, grad-checked, than re-derived per package.
- Foundational for **Informax** (OED/Fisher information over the free vector).

## Notes

Full design: `docs/plans/2026-06-17-jaxstro-params-design.md`. Implemented on `feature/jaxstro-params` off main (after Phase B merge). Validation tests against **progenax** (not gravax). A future `jaxstro.config` (ADR-0010) builds models that `params` then marks free — complementary bridges.

### Build-vs-reuse for the bijector registry (distrax / TFP / flowjax)

We **build** the four trivial elementwise bijectors (`Identity`/`Exp`/`Softplus`/`Sigmoid`) in core rather than depend on **distrax**. Verified 2026-06-17 (PyPI): distrax 0.1.9 hard-requires `tfp-nightly` (an *unpinned nightly* TensorFlow Probability) + `chex` + `numpy` + `absl-py`; release history shows a ~22-month dormancy (0.1.5 Nov-2023 -> 0.1.6 Sep-2025) but is **actively maintained again** (0.1.6-0.1.9 across Sep-2025 to Jun-2026; latest 0.1.9 on 2026-06-12). So the driver is the `tfp-nightly`-in-core dependency, not staleness. Reasons to keep ours:

- **Thin foundation (ADR-0001):** distrax would drag TFP-nightly into *every* `import jaxstro`, for what amounts to `exp`/`softplus`/`sigmoid`. An unpinned nightly is also a reproducibility liability for a research foundation.
- **Static-leaf invariant:** our bijectors are `eqx.Module`s storing bounds as `static=True` (zero array leaves), keeping `Parameterization` hashable/jit-static — the load-bearing property of the leaf-aligned `transform_spec`/`free_meta` design. distrax's chex-dataclass bijectors store params as *array* leaves and would break it.
- **Marginal value ~= 0** for four textbook transforms with analytic, grad-checked, extreme-`u`-stable log-dets.

**Reuse threshold (the day this flips):** when we need *non-trivial* bijectors — `Chain`/`Block` composition, normalizing flows (coupling/RealNVP/spline), autoregressive — re-deriving stable log-dets is real work and reuse wins. Adopt a maintained lib **behind the `AbstractBijector` adapter and gated to `[ml]`/`contrib`**, never in core. Provenance preference (DeepMind/Google-grade) is honored by routing to **TFP-on-JAX or distrax** there; but prefer **flowjax** (equinox-native, already in progenax's `experimental` extra) when it fits, since it preserves the static-leaf model that distrax's chex dataclasses do not.
