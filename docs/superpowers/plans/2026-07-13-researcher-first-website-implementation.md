# Researcher-First Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Jaxstro MyST website around researcher onboarding, connected foundations, method families, scientific representations, auditable workflows, grouped API reference, validation evidence, and substantive background for every ratified future capability.

**Architecture:** Migrate the source tree incrementally so every commit retains a valid MyST TOC and route manifest. Preserve meaningful public routes while replacing generated `/index-N` routes, retire course-facing pages, rename the executable curriculum machinery as a research-workflow registry, and keep planned capabilities visibly separate from importable APIs. This plan changes documentation and documentation-support tooling only; the Python runtime API hard cut and new runtime modules remain separate projects.

**Tech Stack:** MyST Markdown and book theme, KaTeX math, Python 3.11+, JAX, pytest, JSON registries, existing contract/evidence generators, `scripts/check_docs.sh`, and GitHub Pages.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-13-researcher-first-docs-clean-api-design.md` exactly.
- The primary reader may never have used JAX or studied numerical methods.
- The MyST TOC is the canonical navigation; do not create a parallel navigation system.
- Number top-level source roots only: `00`, `10`, `20`, `30`, `40`, `50`, `60`, and `70`.
- Visible TOC depth stops at top-level section -> semantic subsection -> page.
- Use ASCII punctuation and LaTeX for mathematics; do not add Unicode mathematical symbols, arrows, or decorative icons.
- Keep the core derivation, assumptions, AD meaning, and claim boundary visible.
- Use dropdowns only for optional algebra, alternative derivations, or advanced telemetry.
- Planned capability pages are concept pages, not API reference and not implementation evidence.
- Keep meaningful routes stable through `docs/route-manifest.json`.
- Retire the instructor routes and all public course, instructor, rubric, and curriculum framing.
- Use canonical module-qualified imports in every new example.
- Do not change the runtime `jaxstro.numerics` export surface in this project.
- Do not add a new documentation framework or runtime dependency.
- Run `bash scripts/check_docs.sh` after every task that changes published documentation.
- Commit each task only after its focused tests and relevant gates pass.

## Shared Page Contracts

### Current method page

Every substantial current method page uses these visible sections in order:

```markdown
# Page title

Use this page when your research question requires the method described below.

## The question this method answers
## Before computation: what should be true?
## Define the mathematical objects
## Derive the method
## What the algorithm actually does
## What JAX differentiates
## Using it in Jaxstro
## How to audit the result
## Where the claim stops
## Connected ideas
```

### Planned capability or ecosystem guide

Every planned or delegated page uses this visible contract:

```markdown
# Page title

:::{important} Status
Planned Jaxstro capability, Ecosystem guide, or Deferred abstraction.
The runtime boundary is stated in one sentence.
:::

Use this page when you need the scientific idea before the runtime capability exists.

## The scientific question
## Prerequisites
## Mathematical objects
## Core derivation
## Assumptions and failure boundaries
## Worked conceptual example
## Ownership boundary
## Proposed interface
## Evidence required before implementation
## Connected ideas
```

For an ecosystem guide, replace `## Proposed interface` with
`## Canonical ecosystem interface`. Never imply that a delegated API is part of
Jaxstro.

### Section landing page

Each landing page contains:

1. One paragraph explaining the section's role in the scientific chain.
2. A MyST card grid for semantic subsections.
3. A short `note` connecting the section to the whole research workflow.
4. A table distinguishing current, planned, delegated, and deferred material.
5. Direct links to the next useful route.

## Target File Map

### Start here and foundations

```text
docs/00-start-here/index.md
docs/00-start-here/why-jax.md
docs/00-start-here/jax-from-first-principles.md
docs/00-start-here/choose-your-path.md
docs/00-start-here/first-research-calculation.md
docs/00-start-here/ways-to-use-these-docs.md
docs/10-foundations/index.md
docs/10-foundations/mathematical-objects/functions-units-scales.md
docs/10-foundations/mathematical-objects/linear-algebra-language-of-change.md
docs/10-foundations/mathematical-objects/what-is-a-derivative.md
docs/10-foundations/mathematical-objects/probability-and-distributions.md
docs/10-foundations/models-and-computation/what-is-a-model.md
docs/10-foundations/models-and-computation/models-inference-information.md
docs/10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md
docs/10-foundations/models-and-computation/from-relations-to-differentiable-programs.md
```

### API reference

```text
docs/50-api/index.md
docs/50-api/change-constraints/{autodiff,rootfinding,optimization,ode}.md
docs/50-api/approximation-integration/{interpolation,regular-grid,splines,integration,quadrature}.md
docs/50-api/linear-structure/{linear-algebra,compensated,operators,special}.md
docs/50-api/randomness/{distributions,rng,random,sampling,stats}.md
docs/50-api/discrete-space/{grids,meshes,spatial}.md
docs/50-api/physical-representations/{constants,units,quantity,coords,geometry,astrometry,params}.md
docs/50-api/scientific-data/{spectra,atmospheres}.md
docs/50-api/research-infrastructure/{checks,jaxconfig,contracts,evidence,provenance,testing}.md
```

---

### Task 1: Add the beginner JAX route and executable first example

**Files:**
- Create: `docs/00-start-here/why-jax.md`
- Create: `docs/00-start-here/jax-from-first-principles.md`
- Create: `docs/00-start-here/ways-to-use-these-docs.md`
- Create: `examples/onboarding/__init__.py`
- Create: `examples/onboarding/first_jax_map.py`
- Create: `tests/integration/test_jax_onboarding_docs.py`
- Create: `tests/unit/test_first_jax_map.py`
- Move: `docs/00-getting-started/index.md` -> `docs/00-start-here/index.md`
- Move: `docs/00-getting-started/how-to-learn.md` -> `docs/00-start-here/first-research-calculation.md`
- Move: `docs/05-foundations/choose-your-path.md` -> `docs/00-start-here/choose-your-path.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Modify: `docs/index.md`

**Interfaces:**
- Consumes: existing JAX dependency and the predict -> compute -> audit narrative.
- Produces: `scaled_luminosity(radius_ratio, temperature_ratio)`, executable onboarding routes, and the canonical `Start here` TOC group.

- [ ] **Step 1: Write the failing onboarding tests**

Create assertions equivalent to:

```python
PAGES = {
    "why-jax.md": ("# Why JAX?", "## When JAX is the wrong tool"),
    "jax-from-first-principles.md": (
        "# JAX from first principles",
        "## A Python function, a mathematical map, and a traced program",
    ),
    "ways-to-use-these-docs.md": (
        "# Ways to use these docs",
        "## Research-question first",
    ),
}

def test_beginner_pages_exist_and_name_their_boundaries():
    for name, phrases in PAGES.items():
        text = (ROOT / "docs/00-start-here" / name).read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase in text

def test_why_jax_does_not_promise_automatic_correctness_or_speed():
    text = (ROOT / "docs/00-start-here/why-jax.md").read_text(encoding="utf-8")
    for phrase in (
        "JAX does not make an algorithm correct",
        "JAX does not make every program faster",
        "JAX does not make every derivative scientifically meaningful",
    ):
        assert phrase in text
```

Test the scientific map with:

```python
def test_scaled_luminosity_value_and_gradient():
    value = scaled_luminosity(2.0, 0.5)
    d_radius, d_temperature = jax.grad(
        scaled_luminosity, argnums=(0, 1)
    )(2.0, 0.5)
    assert jnp.allclose(value, 0.25)
    assert jnp.allclose(d_radius, 0.25)
    assert jnp.allclose(d_temperature, 2.0)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
pytest tests/integration/test_jax_onboarding_docs.py tests/unit/test_first_jax_map.py -q
```

Expected: FAIL because the new pages and example module do not exist.

- [ ] **Step 3: Implement the executable scientific map**

Use this complete function:

```python
"""Small transformed JAX map used by the beginner documentation."""

import jax
import jax.numpy as jnp


def scaled_luminosity(radius_ratio: float, temperature_ratio: float):
    """Return L/L_ref = (R/R_ref)^2 (T/T_ref)^4."""
    radius = jnp.asarray(radius_ratio)
    temperature = jnp.asarray(temperature_ratio)
    return radius**2 * temperature**4


batched_scaled_luminosity = jax.vmap(scaled_luminosity)
compiled_scaled_luminosity = jax.jit(scaled_luminosity)
```

- [ ] **Step 4: Author the three beginner pages**

`why-jax.md` covers array programming, `jit`, `vmap`, `grad`, JVP, VJP,
PyTrees, keys, Jaxstro's added contracts, compilation latency, immutable arrays,
shape specialization, precision, and honest non-goals. Cite only official JAX
pages:

- `https://docs.jax.dev/en/latest/quickstart.html`
- `https://docs.jax.dev/en/latest/key-concepts.html`
- `https://docs.jax.dev/en/latest/benchmarking.html`

`jax-from-first-principles.md` reuses the scaled-luminosity map to show eager
evaluation, `vmap`, `jit`, `grad`, `make_jaxpr`, immutable updates, explicit
keys, PyTrees, `lax.cond`, 64-bit configuration, and common tracing errors. Link
to the official quickstart, JAX 101, tracing, JIT, and errors pages.

`ways-to-use-these-docs.md` includes the eight approved modes: complete
onboarding, research-question first, background recovery, API lookup, method
comparison, result audit, group or mentor reading, and contributor maintenance.

- [ ] **Step 5: Move the existing entry pages and update links**

Preserve `/choose-your-path`. Replace `/index-1` with `/start-here` and
`/how-to-learn` with `/first-research-calculation`. Add `/why-jax`,
`/jax-from-first-principles`, and `/ways-to-use-these-docs`. Update relative
links in the moved pages and homepage.

- [ ] **Step 6: Run focused and full documentation gates**

```bash
pytest tests/integration/test_jax_onboarding_docs.py tests/unit/test_first_jax_map.py -q
bash scripts/check_docs.sh
```

Expected: focused tests PASS and `ALL DOCS GATES PASSED`.

- [ ] **Step 7: Commit**

```bash
git add docs/00-start-here docs/myst.yml docs/route-manifest.json docs/index.md examples/onboarding tests/integration/test_jax_onboarding_docs.py tests/unit/test_first_jax_map.py
git commit -m "docs: add beginner JAX onboarding"
```

### Task 2: Rebuild Foundations as two connected semantic routes

**Files:**
- Create: `docs/10-foundations/index.md`
- Move: eight current foundation concept pages into `mathematical-objects/` and `models-and-computation/`
- Delete after content migration: `docs/05-foundations/why-this-documentation-works-this-way.md`
- Create: `tests/integration/test_researcher_foundations_structure.py`
- Modify: `docs/00-start-here/ways-to-use-these-docs.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Modify: foundation links throughout routed pages

**Interfaces:**
- Consumes: Start here routes from Task 1.
- Produces: the final `10-foundations` source root and stable concept routes.

- [ ] **Step 1: Write the failing structure test**

```python
FOUNDATION_PAGES = (
    "10-foundations/mathematical-objects/functions-units-scales.md",
    "10-foundations/mathematical-objects/linear-algebra-language-of-change.md",
    "10-foundations/mathematical-objects/what-is-a-derivative.md",
    "10-foundations/mathematical-objects/probability-and-distributions.md",
    "10-foundations/models-and-computation/what-is-a-model.md",
    "10-foundations/models-and-computation/models-inference-information.md",
    "10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md",
    "10-foundations/models-and-computation/from-relations-to-differentiable-programs.md",
)

def test_foundations_use_two_semantic_subsections():
    myst = (ROOT / "docs/myst.yml").read_text(encoding="utf-8")
    assert "title: Mathematical objects" in myst
    assert "title: Models and computation" in myst
    for path in FOUNDATION_PAGES:
        assert (ROOT / "docs" / path).is_file()
        assert myst.count(path) == 1
```

- [ ] **Step 2: Verify failure**

Run `pytest tests/integration/test_researcher_foundations_structure.py -q`.

Expected: FAIL because final paths do not exist.

- [ ] **Step 3: Move pages and rebuild the landing**

Use the exact target paths in this plan. Preserve meaningful routes such as
`/what-is-a-derivative`. Merge the unique philosophical material from
`why-this-documentation-works-this-way.md` into the new Foundations landing and
`ways-to-use-these-docs.md`, then retire its old route.

The landing explains:

```text
optional does not mean unimportant
concepts are connected, not prerequisites to pass
read linearly or enter from any method page
return to Foundations whenever an audit exposes a conceptual gap
```

- [ ] **Step 4: Normalize Foundations page openings**

Add a one-sentence `Use this page when` opening tailored to every page. Replace
course-facing language with research language. Preserve the existing predict,
compute, audit, warranted claim, and misconception material where it remains
scientifically useful.

- [ ] **Step 5: Run gates**

```bash
pytest tests/integration/test_researcher_foundations_structure.py tests/integration/test_foundations_docs.py tests/integration/test_foundations_concepts.py -q
bash scripts/check_docs.sh
```

Expected: all PASS.

- [ ] **Step 6: Pause for Start here and Foundations review**

Build locally with `myst start` and present these routes for user review:

```text
/start-here
/why-jax
/jax-from-first-principles
/ways-to-use-these-docs
/foundations
/what-is-a-derivative
/from-relations-to-differentiable-programs
```

Do not begin the methods migration until review comments are resolved.

- [ ] **Step 7: Commit**

```bash
git add docs/10-foundations docs/00-start-here docs/myst.yml docs/route-manifest.json tests/integration/test_researcher_foundations_structure.py
git add -u docs/05-foundations
git commit -m "docs: reorganize connected foundations"
```

### Task 3: Migrate current numerical methods into method families

**Files:**
- Create: `docs/20-methods/index.md`
- Move: current method pages from `docs/10-theory/` into five semantic subdirectories
- Create: `docs/20-methods/probability-sampling/sampling-and-resampling.md` by separating sampling material from the current random-method chapter
- Create: `tests/integration/test_methods_information_architecture.py`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Modify: routed cross-links to moved pages

**Interfaces:**
- Consumes: Foundations paths from Task 2.
- Produces: current method families and preserved method routes.

- [ ] **Step 1: Write the failing family test**

```python
FAMILIES = {
    "change-constraints-evolution": ("autodiff", "rootfinding", "optimization", "ode"),
    "approximation-integration": (
        "interpolation", "regular-grid", "bsplines", "cumulative-integration", "quadrature"
    ),
    "linear-structure": ("linear-algebra", "operators", "special-functions"),
    "probability-sampling": (
        "distributions", "random-computation", "sampling-and-resampling"
    ),
    "discrete-space": ("grids", "meshes", "spatial"),
}
```

For every page, assert one TOC occurrence, one route-manifest entry, and the
original meaningful public route.

- [ ] **Step 2: Verify failure**

Run `pytest tests/integration/test_methods_information_architecture.py -q`.

Expected: FAIL on missing `docs/20-methods` paths.

- [ ] **Step 3: Move the current pages mechanically**

Rename `cumulative-trapz.md` to `cumulative-integration.md` while preserving the
public route `/cumulative-trapz`. Rename `random.md` to
`random-computation.md` while preserving `/random`, and move its sampling and
resampling discussion into `sampling-and-resampling.md` at `/sampling`. Keep
`quantities.md`, `geometry.md`, and `science-patterns.md` for Tasks 5 and 7
rather than placing them under Methods.

- [ ] **Step 4: Create the Methods landing**

Use card titles:

```text
Change, constraints, and evolution
Approximation from finite information
Linear structure and reusable operators
Randomness as a computational object
Discrete worlds: grids, meshes, and neighborhoods
Signals as sampled evidence
```

- [ ] **Step 5: Update paths without rewriting full method narratives**

Change frontmatter, links, and route wiring only. Full layered method-page
rewrites occur in Tasks 10 and 11, keeping path defects separate from content
defects.

- [ ] **Step 6: Run gates and commit**

```bash
pytest tests/integration/test_methods_information_architecture.py -q
bash scripts/check_docs.sh
git add docs/20-methods docs/myst.yml docs/route-manifest.json tests/integration/test_methods_information_architecture.py
git add -u docs/10-theory
git commit -m "docs: group current numerical methods"
```

### Task 4: Add future method background and ecosystem guides

**Files:**
- Create: the nine future-method paths under `docs/20-methods/` listed in the approved design
- Create: `tests/integration/test_future_method_guides.py`
- Modify: `docs/20-methods/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`

**Interfaces:**
- Consumes: method families from Task 3.
- Produces: one QMC plan, four signal plans, and four delegated solver guides.

- [ ] **Step 1: Write the failing status and derivation test**

```python
GUIDES = {
    "change-constraints-evolution/nonlinear-systems.md": "Ecosystem guide",
    "change-constraints-evolution/adaptive-differential-equations.md": "Ecosystem guide",
    "approximation-integration/adaptive-quadrature.md": "Ecosystem guide",
    "linear-structure/iterative-linear-solvers.md": "Ecosystem guide",
    "probability-sampling/quasi-monte-carlo.md": "Planned Jaxstro capability",
    "signals/signal-axes.md": "Planned Jaxstro capability",
    "signals/windows-spectral-leakage.md": "Planned Jaxstro capability",
    "signals/spectral-estimation.md": "Planned Jaxstro capability",
    "signals/phase-and-delay.md": "Planned Jaxstro capability",
}

def test_future_method_guides_are_honest_and_substantive():
    for relative, status in GUIDES.items():
        text = (ROOT / "docs/20-methods" / relative).read_text(encoding="utf-8")
        assert status in text
        assert "## Core derivation" in text
        assert "## Evidence required before implementation" in text
```

- [ ] **Step 2: Verify failure**

Run `pytest tests/integration/test_future_method_guides.py -q`.

Expected: FAIL because the planned pages do not exist.

- [ ] **Step 3: Author delegated solver guides**

Use these mathematical spines and owners:

| Page | Visible derivation | Owner |
| --- | --- | --- |
| Nonlinear systems | `\mathbf{F}(\mathbf{x})=\mathbf{0}` and `\mathbf{J}\Delta\mathbf{x}=-\mathbf{F}` | Optimistix |
| Adaptive differential equations | local truncation error and accepted-step control | Diffrax |
| Adaptive quadrature | local error estimates and interval refinement | Quadax |
| Iterative linear solvers | `\mathbf{A}\mathbf{x}=\mathbf{b}`, residuals, and Krylov spaces | Lineax and JAX |

Each guide explains what the external package owns and what units, provenance,
telemetry, or evidence could justify a later Jaxstro adapter.

- [ ] **Step 4: Author QMC and signal pages**

The QMC page derives:

```{math}
\widehat{I}_{N}=\frac{1}{N}\sum_{n=1}^{N}f(\mathbf{u}_{n}).
```

It distinguishes deterministic low-discrepancy points from replicated random
scrambles used to estimate uncertainty.

The signal pages cover the discrete Fourier transform, sample cadence and
Nyquist frequency, window normalization and equivalent noise bandwidth,
one-sided and two-sided power conventions, cross spectra, phase wrapping, and:

```{math}
\tau(f)=-\frac{\phi(f)}{2\pi f}.
```

State that JAX owns FFT and convolution mechanics; Jaxstro's proposed value is
the scientific convention and evidence contract.

- [ ] **Step 5: Run gates and commit**

```bash
pytest tests/integration/test_future_method_guides.py -q
bash scripts/check_docs.sh
git add docs/20-methods docs/myst.yml docs/route-manifest.json tests/integration/test_future_method_guides.py
git commit -m "docs: add future method background"
```

### Task 5: Build Scientific representations from existing package concepts

**Files:**
- Create: `docs/30-representations/index.md`
- Move: `docs/10-theory/quantities.md` and `docs/10-theory/geometry.md`
- Move and refocus: quantity, spectra, and atmosphere architecture pages from `docs/20-architecture/`
- Create: current representation pages for constants, units, coordinates, astrometry, parameters, and scientific state
- Create: `tests/integration/test_representations_information_architecture.py`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`

**Interfaces:**
- Consumes: final Foundations links and current package modules.
- Produces: four current representation families with honest runtime boundaries.

- [ ] **Step 1: Write the failing representation test**

```python
OWNERS = {
    "units-quantities/constants-and-conventions.md": "jaxstro.constants",
    "units-quantities/unit-systems.md": "jaxstro.units",
    "units-quantities/quantities-and-dimensions.md": "jaxstro.quantity",
    "units-quantities/equivalencies.md": "jaxstro.quantity",
    "geometry-coordinates/coordinate-transformations.md": "jaxstro.coords",
    "geometry-coordinates/geometry.md": "jaxstro.geometry",
    "geometry-coordinates/astrometry.md": "jaxstro.astrometry",
    "spectra-atmospheres/spectral-representations.md": "jaxstro.spectra",
    "spectra-atmospheres/conservative-spectral-resampling.md": "jaxstro.spectra",
    "spectra-atmospheres/atmosphere-libraries.md": "jaxstro.atmospheres",
    "spectra-atmospheres/source-artifacts-and-adapters.md": "jaxstro.atmospheres",
    "parameters-state/parameters-and-transforms.md": "jaxstro.params",
    "parameters-state/pytrees-as-scientific-state.md": "jaxstro.params",
    "parameters-state/serialization-and-provenance.md": "jaxstro.provenance",
}
```

Import each owner and require exactly one TOC occurrence per page.

- [ ] **Step 2: Verify failure**

Run `pytest tests/integration/test_representations_information_architecture.py -q`.

- [ ] **Step 3: Move and refocus existing pages**

Preserve `/quantities`, `/geometry`, `/quantity-system`,
`/spectra-data-architecture`, and `/atmosphere-capabilities`. Replace
architecture-log openings with researcher-facing questions while retaining
source-backed ownership and data-boundary details.

- [ ] **Step 4: Author missing current pages and landing**

Every page states:

```text
mathematical object
physical convention
runtime owner
shape and unit policy
transform boundary
evidence link
downstream interpretation boundary
```

- [ ] **Step 5: Run gates and commit**

```bash
pytest tests/integration/test_representations_information_architecture.py -q
bash scripts/check_docs.sh
git add docs/30-representations docs/myst.yml docs/route-manifest.json tests/integration/test_representations_information_architecture.py
git add -u docs/10-theory docs/20-architecture
git commit -m "docs: organize scientific representations"
```

### Task 6: Add uncertainty and field background pages

**Files:**
- Create: `docs/30-representations/uncertainty/what-uncertainty-represents.md`
- Create: `docs/30-representations/uncertainty/linearized-propagation.md`
- Create: `docs/30-representations/uncertainty/sigma-point-propagation.md`
- Create: `docs/30-representations/uncertainty/ensemble-propagation.md`
- Create: `docs/30-representations/fields/fields-and-domains.md`
- Create: `docs/30-representations/fields/topology-and-discretization.md`
- Create: `docs/30-representations/fields/field-operators.md`
- Create: `tests/integration/test_future_representation_guides.py`
- Modify: `docs/30-representations/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`

**Interfaces:**
- Consumes: representation terminology from Task 5.
- Produces: the conceptual contract for future `jaxstro.uncertainty` and deferred fields.

- [ ] **Step 1: Write the failing guide test**

Require all seven pages, their status admonitions, `## Core derivation`, and
`## Evidence required before implementation`. Require `Planned Jaxstro
capability` on uncertainty pages and `Deferred abstraction` on field pages.

- [ ] **Step 2: Verify failure**

Run `pytest tests/integration/test_future_representation_guides.py -q`.

- [ ] **Step 3: Author uncertainty pages**

Use these visible derivations:

```{math}
\mathbf{C}_{y}\approx\mathbf{J}\mathbf{C}_{x}\mathbf{J}^{\mathsf{T}}.
```

```{math}
\widehat{\boldsymbol{\mu}}_{y}=\sum_i w_i f(\boldsymbol{\chi}_i),
\qquad
\widehat{\mathbf{C}}_y=\sum_i w_i
(f(\boldsymbol{\chi}_i)-\widehat{\boldsymbol{\mu}}_y)
(f(\boldsymbol{\chi}_i)-\widehat{\boldsymbol{\mu}}_y)^{\mathsf{T}}.
```

```{math}
\widehat{\boldsymbol{\mu}}_{y}=\frac{1}{N}\sum_{n=1}^{N}f(\mathbf{x}_n).
```

Explain local-linearization failure, covariance rank, sigma-point weight
conventions, ensemble Monte Carlo error, key ownership, and the boundary with
NumPyro, BlackJAX, and Informax.

- [ ] **Step 4: Author field pages**

Define a field as a map from a domain to values, distinguish coordinates from
topology, derive discrete gradient/divergence examples, and explain why a
runtime `jaxstro.fields` package is deferred until two consumers establish
shared abstractions.

- [ ] **Step 5: Run gates and commit**

```bash
pytest tests/integration/test_future_representation_guides.py -q
bash scripts/check_docs.sh
git add docs/30-representations docs/myst.yml docs/route-manifest.json tests/integration/test_future_representation_guides.py
git commit -m "docs: add uncertainty and field foundations"
```

### Task 7: Rebuild Research workflows and rename the executable registry

**Files:**
- Create: `docs/40-workflows/index.md`
- Move: data-processing how-to pages into `docs/40-workflows/data-pipelines/`
- Move: investigations into `docs/40-workflows/investigations/`
- Move/refocus: `docs/10-theory/science-patterns.md` and provenance material into workflow families
- Create: `docs/40-workflows/scientific-ml/preprocessing.md`
- Create: `docs/40-workflows/scientific-ml/data-plans.md`
- Create: `docs/40-workflows/scientific-ml/auditable-training.md`
- Create: `docs/40-workflows/scientific-ml/ecosystem-boundaries.md`
- Create: `docs/40-workflows/differentiable-research/what-jax-differentiates.md`
- Create: `docs/40-workflows/differentiable-research/auditing-derivatives.md`
- Create: `docs/40-workflows/differentiable-research/branches-limits-implicit-sensitivities.md`
- Create: `docs/40-workflows/reproducible-research/random-state-ownership.md`
- Create: `docs/40-workflows/reproducible-research/contracts-and-provenance.md`
- Create: `docs/40-workflows/reproducible-research/evidence-and-claim-boundaries.md`
- Move: `docs/curriculum/units.json` -> `docs/40-workflows/investigations/registry.json`
- Move: `scripts/build_curriculum_registry.py` -> `scripts/build_research_workflow_registry.py`
- Move: `tests/unit/test_curriculum_registry.py` -> `tests/unit/test_research_workflow_registry.py`
- Move: `tests/integration/test_curriculum_conventions.py` -> `tests/integration/test_research_workflow_conventions.py`
- Create: `tests/integration/test_research_workflows_information_architecture.py`
- Modify: `scripts/check.sh`
- Modify: `scripts/check_docs.sh`
- Modify: investigation integration-test docstrings
- Move: `docs/validation/curriculum-coverage.json` -> `docs/validation/research-workflow-coverage.json`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`

**Interfaces:**
- Consumes: current contracts, evidence IDs, examples, and canonical method paths.
- Produces: schema version 2 research-workflow registry and five workflow families.

- [ ] **Step 1: Write failing registry and navigation tests**

Define the final manifest fields exactly:

```python
WORKFLOW_FIELDS = {
    "id",
    "title",
    "page",
    "example",
    "prerequisites",
    "public_apis",
    "contract_ids",
    "evidence_ids",
    "validation_targets",
    "limitations",
}
```

Require schema version `2`, no `instructor_route`, generated index at
`docs/40-workflows/investigations/index.md`, and coverage at
`docs/validation/research-workflow-coverage.json`.

- [ ] **Step 2: Verify failure**

```bash
pytest tests/unit/test_research_workflow_registry.py tests/integration/test_research_workflows_information_architecture.py -q
```

Expected: FAIL until files are renamed and schema migrated.

- [ ] **Step 3: Rename and simplify the registry**

Rename the public functions and signatures exactly:

- `load_and_validate_workflows(root: Path) -> list[dict[str, Any]]`
- `validate_unique_references(identity: str, workflow: dict[str, Any], fields: Sequence[str]) -> None`
- `render_outputs(root: Path = ROOT) -> dict[Path, str]`

Import `Sequence` from `collections.abc` and keep the existing deterministic
sorting, duplicate rejection, path resolution, and rendering logic.

Remove `validate_instructor_route`. Change emitted prose and error messages from
curriculum/unit to research workflow/investigation. Keep fail-closed contract,
evidence, API-path, example, prerequisite, and validation-target checks.

- [ ] **Step 4: Move existing workflows and update registry paths**

Preserve meaningful routes for investigations and data-pipeline pages. Update
manifest prerequisites to final Foundations paths and investigation pages to
final Workflow paths.

- [ ] **Step 5: Author scientific-ML background pages**

| Page | Mathematical spine | Boundary |
| --- | --- | --- |
| Preprocessing | `z=(x-\mu)/s` and whitening by covariance factorization | host fit, JAX apply |
| Data plans | disjoint index sets and fixed-shape batch plans | no hidden split state |
| Auditable training | `\theta_{k+1}=U(\theta_k,g_k,s_k)` for exactly `K` steps | optimizer agnostic |
| Ecosystem boundaries | model, optimizer, provenance, and inference ownership table | Equinox, Optax, Jaxstro, Informax |

Each page is marked `Planned Jaxstro capability` and names the proposed module
without placing symbols in API reference.

- [ ] **Step 6: Run registry, workflow, and docs gates**

```bash
python scripts/build_research_workflow_registry.py --emit
pytest tests/unit/test_research_workflow_registry.py tests/integration/test_research_workflow_conventions.py tests/integration/test_research_workflows_information_architecture.py -q
bash scripts/check_docs.sh
```

Expected: generated artifacts fresh and all tests PASS.

- [ ] **Step 7: Pause for Methods, representations, and workflows review**

Present the rendered landing pages plus QMC, signal, uncertainty, fields,
scientific ML, and one ecosystem guide. Resolve content and visual feedback
before API work.

- [ ] **Step 8: Commit**

```bash
git add docs/40-workflows docs/validation/research-workflow-coverage.json scripts/build_research_workflow_registry.py scripts/check.sh scripts/check_docs.sh tests
git add -u docs/50-howto docs/70-investigations docs/curriculum docs/validation/curriculum-coverage.json scripts/build_curriculum_registry.py
git commit -m "docs: organize auditable research workflows"
```

### Task 8: Split the omnibus API reference by method owner

**Files:**
- Create: all paths under `docs/50-api/` listed in Target File Map
- Create: `tests/integration/test_grouped_api_reference.py`
- Move/refocus: `docs/40-api/contracts.md` and provenance API pages
- Delete after migration: `docs/40-api/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Modify: API links throughout routed pages

**Interfaces:**
- Consumes: current importable modules and contract/evidence pages.
- Produces: grouped, owner-qualified API pages without changing runtime exports.

- [ ] **Step 1: Write the failing API ownership test**

Use a mapping whose keys are page paths and values are actual owner modules:

```python
API_OWNERS = {
    "change-constraints/autodiff.md": "jaxstro.numerics.autodiff",
    "change-constraints/rootfinding.md": "jaxstro.numerics.rootfinding",
    "change-constraints/optimization.md": "jaxstro.numerics.optimization",
    "change-constraints/ode.md": "jaxstro.numerics.ode",
    "approximation-integration/interpolation.md": "jaxstro.numerics.interpolation",
    "approximation-integration/regular-grid.md": "jaxstro.numerics.regular_grid",
    "approximation-integration/splines.md": "jaxstro.numerics.splines",
    "approximation-integration/integration.md": "jaxstro.numerics.integration",
    "approximation-integration/quadrature.md": "jaxstro.numerics.quadrature",
    "linear-structure/linear-algebra.md": "jaxstro.numerics.linear_algebra",
    "linear-structure/compensated.md": "jaxstro.numerics.compensated",
    "linear-structure/operators.md": "jaxstro.numerics.operators",
    "linear-structure/special.md": "jaxstro.numerics.special",
    "randomness/distributions.md": "jaxstro.numerics.distributions",
    "randomness/rng.md": "jaxstro.numerics.rng",
    "randomness/random.md": "jaxstro.numerics.random",
    "randomness/sampling.md": "jaxstro.numerics.sampling",
    "randomness/stats.md": "jaxstro.numerics.stats",
    "discrete-space/grids.md": "jaxstro.numerics.grids",
    "discrete-space/meshes.md": "jaxstro.numerics.meshes",
    "discrete-space/spatial.md": "jaxstro.spatial",
    "physical-representations/constants.md": "jaxstro.constants",
    "physical-representations/units.md": "jaxstro.units",
    "physical-representations/quantity.md": "jaxstro.quantity",
    "physical-representations/coords.md": "jaxstro.coords",
    "physical-representations/geometry.md": "jaxstro.geometry",
    "physical-representations/astrometry.md": "jaxstro.astrometry",
    "physical-representations/params.md": "jaxstro.params",
    "scientific-data/spectra.md": "jaxstro.spectra",
    "scientific-data/atmospheres.md": "jaxstro.atmospheres",
    "research-infrastructure/checks.md": "jaxstro.numerics.checks",
    "research-infrastructure/jaxconfig.md": "jaxstro.jaxconfig",
    "research-infrastructure/contracts.md": "jaxstro.contracts",
    "research-infrastructure/evidence.md": "jaxstro.evidence",
    "research-infrastructure/provenance.md": "jaxstro.provenance",
    "research-infrastructure/testing.md": "jaxstro.testing",
}

def test_every_api_page_names_an_importable_owner():
    for relative, owner in API_OWNERS.items():
        importlib.import_module(owner)
        text = (ROOT / "docs/50-api" / relative).read_text(encoding="utf-8")
        assert f"`{owner}`" in text
```

- [ ] **Step 2: Verify failure**

Run `pytest tests/integration/test_grouped_api_reference.py -q`.

- [ ] **Step 3: Create the API landing and owner pages**

The landing states the canonical policy:

```python
from jaxstro.numerics import rootfinding
from jaxstro.numerics.rootfinding import safeguarded_bracketed_root
```

It states that flat callable re-exports from `jaxstro.numerics` are legacy
inventory awaiting Project 2 and are not the canonical documentation path.

Each owner page contains:

```text
Owner import path
Purpose
Public records and callables
Shape and dtype expectations
JAX transforms and AD classification
Failure behavior
Contract and evidence links
One canonical import example
```

- [ ] **Step 4: Migrate the omnibus content without duplicating descriptions**

Move each signature or symbol family from the current omnibus page to exactly
one owner page. Keep the existing safeguard that `pchip_slopes` and
`monotone_cubic_interp` each receive one description.

- [ ] **Step 5: Run API and docs gates**

```bash
pytest tests/integration/test_grouped_api_reference.py tests/integration/test_api_reference.py tests/integration/test_contract_docs.py -q
bash scripts/check_docs.sh
```

Expected: PASS. Rewrite legacy API tests to check canonical owner imports while
leaving runtime compatibility untouched.

- [ ] **Step 6: Commit**

```bash
git add docs/50-api docs/myst.yml docs/route-manifest.json tests/integration/test_grouped_api_reference.py tests/integration/test_api_reference.py
git add -u docs/40-api
git commit -m "docs: group API reference by method owner"
```

### Task 9: Consolidate Validation and Project, then retire course framing

**Files:**
- Reorganize: `docs/60-validation/` into landing, numerical, data, and methods
- Create: `docs/70-project/index.md`
- Move: architecture overview and science-general vision into `docs/70-project/direction/`
- Move: development-log pages into `docs/70-project/development/`
- Move: ADRs into `docs/70-project/decisions/`
- Move: release pages into `docs/70-project/release/`
- Move: bibliography into `docs/70-project/bibliography/`
- Delete: `docs/80-instructor/`
- Create: `tests/integration/test_public_research_software_language.py`
- Modify: `tests/integration/test_assessment_scorecard.py`
- Modify: `tests/integration/test_sota_assessment.py`
- Modify: development, scorecard, and SOTA prose
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`

**Interfaces:**
- Consumes: research-workflow coverage artifact and grouped API routes.
- Produces: final Validation and Project roots and zero routed course framing.

- [ ] **Step 1: Write the failing public-language test**

Read only Markdown files named in the route manifest:

```python
FORBIDDEN_TERMS = re.compile(
    r"\b(course|courses|curriculum|instructor|instructors|"
    r"teaching assistant|teaching assistants|assessment rubric)\b",
    re.IGNORECASE,
)
FORBIDDEN_UNICODE = {
    "\u2192", "\u2190", "\u2194", "\u2264", "\u2265", "\u2260",
    "\u2248", "\u2208", "\u2209", "\u221e", "\u2211", "\u220f",
    "\u221a", "\u2202", "\u2207", "\u00d7", "\u00b7", "\u2212",
    "\u2013", "\u2014",
}

def test_routed_pages_use_research_software_language_and_latex_math():
    manifest = json.loads((DOCS / "route-manifest.json").read_text())
    for relative in manifest:
        if not relative.endswith(".md"):
            continue
        text = (DOCS / relative).read_text(encoding="utf-8")
        assert FORBIDDEN_TERMS.search(text) is None, relative
        assert not (set(text) & FORBIDDEN_UNICODE), relative
```

Preserve accented proper names and bibliographic spelling.

- [ ] **Step 2: Verify failure**

Run `pytest tests/integration/test_public_research_software_language.py -q`.

Expected: FAIL on current instructor pages and routed course phrasing.

- [ ] **Step 3: Migrate Validation and Project**

Preserve `/evidence-index`, `/sota-assessment`, and
`/future-capabilities-roadmap`. Replace `/index-8` with `/validation` and
`/index-9` with `/project`. Keep individual ADR routes stable but list only the
ADR index in the primary TOC.

- [ ] **Step 4: Rewrite the roadmap to the approved SOTA ownership map**

Use this priority order:

```text
1. jaxstro.ml
2. jaxstro.numerics.qmc
3. jaxstro.uncertainty
4. jaxstro.signal
5. consumer-driven ecosystem adapters
6. fields deferred until two consumers
```

Move iterative linear solvers, nonlinear systems, adaptive quadrature, and
adaptive differential equations into delegated ecosystem guidance. Preserve
the checklist form and evidence-based definition of done.

- [ ] **Step 5: Retire instructor pages and reframe scorecards**

Delete the three instructor pages and remove their routes. Replace executable
curriculum and instructor-route metrics with executable research-workflow
coverage, contract coverage, evidence linkage, and limitation coverage. Remove
tests that grade course-use materials; preserve tests for investigation
reasoning and warranted claims.

- [ ] **Step 6: Replace Unicode source symbols**

For routed Markdown and `docs/myst.yml`:

- Replace prose arrows with `->`.
- Replace mathematical relations with LaTeX such as `$x \le y$`,
  `$\nabla f$`, and `$a \times b$`.
- Replace en and em dashes with commas, colons, parentheses, or ASCII hyphens.

- [ ] **Step 7: Run gates and commit**

```bash
pytest tests/integration/test_public_research_software_language.py tests/integration/test_assessment_scorecard.py tests/integration/test_sota_assessment.py -q
bash scripts/check_docs.sh
git add docs/60-validation docs/70-project docs/myst.yml docs/route-manifest.json tests/integration
git add -u docs/20-architecture docs/30-decisions docs/80-instructor docs/90-development-log docs/95-release docs/99-bibliography
git commit -m "docs: consolidate validation and project guidance"
```

### Task 10: Apply the layered narrative to change and approximation methods

**Files:**
- Modify: all current pages under `docs/20-methods/change-constraints-evolution/`
- Modify: all current pages under `docs/20-methods/approximation-integration/`
- Create: `tests/integration/test_method_page_contract.py`
- Modify: related Foundations, API, and Validation cross-links

**Interfaces:**
- Consumes: Shared Current method page contract.
- Produces: derivation-complete change and approximation chapters.

- [ ] **Step 1: Write the failing page-contract test**

```python
REQUIRED_HEADINGS = (
    "## The question this method answers",
    "## Before computation: what should be true?",
    "## Define the mathematical objects",
    "## Derive the method",
    "## What the algorithm actually does",
    "## What JAX differentiates",
    "## Using it in Jaxstro",
    "## How to audit the result",
    "## Where the claim stops",
    "## Connected ideas",
)
```

Require every current method page in these two families to include all headings
in order.

- [ ] **Step 2: Verify failure**

Run `pytest tests/integration/test_method_page_contract.py -q`.

- [ ] **Step 3: Rewrite one family at a time**

Preserve correct current material and add these missing derivations:

```text
autodiff: JVP and VJP as linear maps
rootfinding: bracket invariants, safeguarded proposals, and implicit derivative
optimization: gradient descent, line search, and convergence diagnostics
ODE: Euler, midpoint, RK4, and local/global truncation distinction
interpolation: linear, monotone cubic, regular-grid, and boundary policies
splines: basis construction, knot support, derivatives, and regularization
integration: trapezoid accumulation and Gaussian quadrature exactness
```

Use `important` for assumptions, `warning` for AD/failure boundaries, `tip` for
method choice, and `seealso` for connected pages. Do not hide core algebra.

- [ ] **Step 4: Run focused docs tests**

```bash
pytest tests/integration/test_method_page_contract.py tests/integration/test_rootfinding_docs.py tests/integration/test_interpolation_docs.py tests/integration/test_bspline_docs.py tests/integration/test_regular_grid_docs.py -q
bash scripts/check_docs.sh
```

- [ ] **Step 5: Commit**

```bash
git add docs/20-methods tests/integration/test_method_page_contract.py
git commit -m "docs: derive change and approximation methods"
```

### Task 11: Apply the layered narrative to linear, random, and discrete methods

**Files:**
- Modify: current pages under `docs/20-methods/linear-structure/`
- Modify: current pages under `docs/20-methods/probability-sampling/`
- Modify: current pages under `docs/20-methods/discrete-space/`
- Modify: `tests/integration/test_method_page_contract.py`
- Modify: related Foundations, API, and Validation cross-links

**Interfaces:**
- Consumes: method-page test and grammar from Task 10.
- Produces: derivation-complete remaining current method families.

- [ ] **Step 1: Extend the test page list and verify failure**

Add all current pages in these three families to the exact same
`REQUIRED_HEADINGS` contract. Run the test and confirm the newly added pages
fail.

- [ ] **Step 2: Rewrite the linear and operator pages**

Cover covariance, least squares, conditioning, operator composition, basis
functions, PyTree linearity, and what differentiating through the operations
means.

- [ ] **Step 3: Rewrite the probability and random pages**

Cover support, normalization, inverse CDF sampling, PRNG key ownership,
resampling, limiting distributions, pathwise versus discrete operations, and
the exact boundary of differentiability.

- [ ] **Step 4: Rewrite grids, meshes, and spatial pages**

Cover cell/face geometry, conservative transfer, neighborhoods, Morton order,
pair construction, discontinuous topology changes, and validation by
conservation and brute-force comparison.

- [ ] **Step 5: Run gates and commit**

```bash
pytest tests/integration/test_method_page_contract.py tests/integration/test_linear_algebra_docs.py tests/integration/test_random_docs.py tests/integration/test_spatial_docs.py -q
bash scripts/check_docs.sh
git add docs/20-methods tests/integration/test_method_page_contract.py
git commit -m "docs: derive linear random and discrete methods"
```

### Task 12: Final homepage, MyST grammar, accessibility, and release verification

**Files:**
- Rewrite: `docs/index.md`
- Modify: every section landing as required by Shared Page Contracts
- Create: `tests/integration/test_myst_semantic_grammar.py`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: all final routes, pages, contracts, and evidence.
- Produces: the complete eight-route website ready for user review and publication.

- [ ] **Step 1: Write the failing final architecture test**

Require these top-level titles exactly once and in order:

```python
TOP_LEVEL = (
    "Start here",
    "Foundations",
    "Numerical methods",
    "Scientific representations",
    "Research workflows",
    "API reference",
    "Validation and evidence",
    "Project",
)
```

Require `/start-here`, `/methods`, `/representations`, `/workflows`, `/api`,
`/validation`, and `/project`; reject `/index-1` through `/index-11` and the
three retired instructor routes.

Update `project.exclude` so `audits/**`, `plans/**`, `superpowers/**`, and
`_build/**` remain internal and outside the published site.

- [ ] **Step 2: Write semantic MyST checks**

Require card grids only on landing pages, stable labels on equations and
figures, alt text and captions on figures, status admonitions on every
planned/delegated/deferred page, and no tabs except an explicit allowlist.

The dropdown check enforces that `Core derivation`, `Assumptions`, `What JAX
differentiates`, and `Where the claim stops` occur outside any `dropdown`
directive range.

- [ ] **Step 3: Rewrite the homepage**

Introduce the scientific chain, link `Why JAX?`, provide
research-question-first cards, and offer direct routes to Foundations, Methods,
Workflows, API, and Validation. Say astro-first and science-general without
listing the entire TOC.

- [ ] **Step 4: Perform the semantic MyST pass**

Verify KaTeX equations, labels, references, definitions, algorithms, glossary
terms, figures, tables, and admonitions in rendered output. Remove decorative
boxes and repeated callouts that interrupt the narrative.

- [ ] **Step 5: Run the complete local gate**

```bash
pytest tests/integration/test_jax_onboarding_docs.py tests/integration/test_researcher_foundations_structure.py tests/integration/test_methods_information_architecture.py tests/integration/test_future_method_guides.py tests/integration/test_representations_information_architecture.py tests/integration/test_future_representation_guides.py tests/integration/test_research_workflows_information_architecture.py tests/integration/test_grouped_api_reference.py tests/integration/test_public_research_software_language.py tests/integration/test_method_page_contract.py tests/integration/test_myst_semantic_grammar.py -q
bash scripts/check.sh
bash scripts/check_docs.sh
```

Expected: all tests PASS, the full code gate passes, and `ALL DOCS GATES
PASSED`.

- [ ] **Step 6: Run accessibility and layout review**

Inspect desktop and narrow-width rendering for TOC scanning, heading hierarchy,
equation and code overflow, table readability, keyboard navigation, focus
visibility, figure alt text, and admonition density.

- [ ] **Step 7: Pause for final user review**

Provide the local rendered site and a checklist of all eight top-level routes.
Do not publish until review comments are resolved.

- [ ] **Step 8: Verify the published site**

After explicit publication approval, run the Pages workflow and check over live
HTTP:

```text
/
/start-here/
/why-jax/
/jax-from-first-principles/
/foundations/
/methods/
/representations/
/workflows/
/api/
/validation/
/project/
/future-capabilities-roadmap/
```

Also verify one CSS asset, one equation page, one API page, and one planned
capability page.

- [ ] **Step 9: Update status and commit**

Record the website milestone and next API-hard-cut project in `STATUS.md`.

```bash
git add docs tests scripts examples STATUS.md
git commit -m "docs: complete researcher-first website"
git status --short
```

Expected final status: no scoped tracked or untracked website changes. Preserve
unrelated user-owned files.

## Specification Coverage

| Approved requirement | Implemented by |
| --- | --- |
| Beginner JAX motivation, mental model, and documentation-use modes | Tasks 1-2 |
| Eight-route MyST information architecture | Tasks 1-9 and 12 |
| Connected optional Foundations | Task 2 |
| Methods grouped by durable families | Tasks 3-4 |
| Scientific representations and future uncertainty/fields | Tasks 5-6 |
| Research workflows and scientific ML | Task 7 |
| API grouped by method type and true Python owner | Task 8 |
| Validation, Project, route migration, and course-language retirement | Task 9 |
| Visible derivations, JAX meaning, audits, and claim boundaries | Tasks 10-11 |
| MyST semantic grammar, KaTeX, accessibility, visual review, and live HTTP checks | Task 12 |
| No runtime API hard cut or future-module implementation in Project 1 | Project 1 Exit Gate |

## Project 1 Exit Gate

Project 1 is complete only when all Task 12 checks pass and the rendered site is
approved. The next project begins with a generated per-symbol API inventory and
multi-repository consumer migration. Do not hard-cut `jaxstro.numerics` exports
or implement `jaxstro.ml`, QMC, uncertainty, or signal code under this plan.
