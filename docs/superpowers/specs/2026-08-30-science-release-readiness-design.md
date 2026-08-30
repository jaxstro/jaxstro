# Jaxstro science and release readiness design

**Status:** proposed from the live `main` audit at `c646df06c4cebbe77084131995f700639e4a99c4`; no runtime or release claim changes are authorized by this document.

## Decision

Treat release readiness and scientific readiness as separate claims with a shared evidence boundary.

- A **release-ready candidate** is a clean Python distribution whose declared support, artifacts, public modules, generated documentation, and required CI mirror have been exercised at one immutable commit. Runtime dependency resolution and the PEP 517 backend version are frozen; byte-identical artifacts across arbitrary builders are not claimed.
- A **scientifically-qualified core** is a deliberately small list of public contracts for which value, transform, failure, limitation, and independent validation evidence can be found without maintainer knowledge. It is not a claim that every importable callable is scientifically qualified.
- A **hard cutover** is allowed only after pinned consumer evidence establishes that no active consumer needs the old path, or after the consuming owners have merged and qualified their migrations. It is never justified merely by the existence of a better owner.

## Live evidence motivating the work

1. `scripts/check.sh` is the intended exhaustive local gate, but its artifact smoke builds only a wheel and imports only `jaxstro`; the release checklist requires both wheel and sdist inspection and every advertised top-level module.
2. `.github/workflows/full-gate.yml` has scheduled/manual triggers and separately approximates the local steps. It is not an exact `scripts/check.sh` mirror on each merge to `main`.
3. `scripts/check_docs.sh` depends on a globally installed `myst` executable. Pages installs `mystmd@1.10.1` globally, but the repository has no locked Node dependency graph and the local release gate does not provide that prerequisite. `.gitignore` also does not exclude the `node_modules/` tree that the repair will create.
4. `src/jaxstro/__init__.py` exports `contracts`, `evidence`, `quad`, and `spectra`, while `docs/70-project/direction/architecture.md` omits them from its directly importable-module inventory.
5. The generated contract inventory records 18 callable-level contracts and 235 explicitly unclassified public callables. Existing contracts already encode maturity, transform support, failure boundaries, evidence, and limitations; a new parallel validation framework is not needed.
6. `jaxstro.units` and `jaxstro.numerics.universal_kepler_step` have real downstream use. The latter is imported by Gravax. Progenax has four source imports and five direct-test import statements across two files for `jaxstro.numerics.integration` or `jaxstro.numerics.quadrature`. Jaxstro's own compatibility test establishes that their canonical `jaxstro.quad` replacements have exact callable identity or exact array parity, so a focused Progenax migration is warranted. Its checked-in editable Jaxstro lock record lacks Jaxstro's live `sympy` dependency, so the downstream locked environment must be refreshed before that migration can be qualified.
7. `jaxstro.quad` still imports the Hermite helpers from `jaxstro.numerics.quadrature`. Consumer migration is therefore not evidence that that legacy module is deletable.

## Scope and non-goals

This program hardens the existing foundation. It does not add `ml`, `uncertainty`, `signal`, fields, a general configuration system, new solver families, a quantity replacement cutover, a platform-performance benchmark program, or a PyPI upload.

The first supported release target is **CPython 3.13 on Ubuntu x86_64 CPU with `JAX_ENABLE_X64=1`**. Other JAX/JAXlib backends may run, but they are not a qualified support claim until a frozen CI matrix qualifies them. The package must not advertise GPU acceleration or operating-system independence as tested support before that evidence exists, including through PyPI classifiers.

The first scientifically-qualified profile contains only:

- module `jaxstro.units`;
- callable `jaxstro.numerics.safeguarded_bracketed_root`;
- callable `jaxstro.numerics.implicit_bracketed_root`; and
- callable `jaxstro.numerics.universal_kepler_step`.

`jaxstro.quad.fixed` and `jaxstro.quad.integrate` remain published as experimental methods with their current typed contracts and limitations. They are not silently promoted by the qualified-core profile.

## Architecture

Use the existing contract registry as the one scientific-claim owner. Add only a static `QUALIFIED_CORE_V1` selection and a generated/documented view of the selected existing contract records. The selection is data, not an abstraction layer: it avoids copying transform and failure semantics into a second schema.

Use a single canonical `PUBLIC_MODULES` tuple for the root namespace, documentation inventory, and installed-distribution smoke test. It repairs one observed duplicated truth without changing the owner of individual modules. The route manifest remains the canonical route owner and is changed with every navigation change.

Use a repository-local, lockfile-backed MyST CLI. This replaces global mutable tool installation; it does not add a runtime dependency to the wheel.

## Success criteria

1. The local release command builds wheel and sdist into a temporary directory with the pinned Hatchling backend, validates both artifacts' contents and metadata, and imports every canonical public top-level module from separate clean wheel and sdist environments.
2. The exact local release command runs on every `main` push in a CI job named `release-mirror`; a scheduled/manual run uses the same command. The existing slow `scientific-validation` lane remains separately scheduled/manual and is never implied to be part of the fast release mirror.
3. The MyST executable is resolved from `package-lock.json` locally and in CI; no workflow uses `npm install --global`.
4. README, architecture, release, API, route manifest, and package metadata state the same public surface, direct dependencies, support boundary, and non-claims.
5. The qualified-core profile resolves every named contract with validated maturity and public-surface evidence. Each selected callable has a limitation plus observable failure/value-first boundary and validation target. `jaxstro.units` is the explicit static-module exception: it records ownership/non-ownership, CGS dimensional policy, and scale/conversion/default unit-test evidence; it makes no transform or failure claim. Safeguarded root records new public-solver analytical and typed-failure validation before being selected. Its rendered page names the exact scientific boundary and excluded methods.
6. Progenax imports `jaxstro.quad.cumulative_trapezoid`, `jaxstro.quad.gauss_hermite_nodes`, and `jaxstro.quad.hermite_coefficients` directly; its public `progenax.numerics.cumulative_trapz` name remains stable; its refreshed lock passes `uv lock --check` before tests run.
7. A separately committed release attestation records candidate SHA X and CI evidence for X. It is not itself a qualified candidate and any future tag must target X.
8. A future legacy deletion requires an internal canonical-owner move with Jaxstro identity/value/AD evidence, then clean immutable revisions of at least two independent consumers. Progenax migration alone is insufficient.

## Delivery order

```text
release truth and package artifacts
  -> reproducible documentation toolchain
  -> exact CI release mirror
  -> retained slow scientific-validation lane
  -> qualified-core profile and rendered evidence
  -> Progenax canonical-quad migration and targeted qualification
  -> internal legacy-owner move plus pinned multi-consumer qualification
  -> only then a separately approved legacy or quantity cutover
```

The final two arrows require consumer-owner and scientific-owner judgment. A green Jaxstro suite alone cannot replace that evidence.
