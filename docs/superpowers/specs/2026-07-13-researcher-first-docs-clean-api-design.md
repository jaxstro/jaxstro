---
title: Researcher-first documentation and clean public API design
date: 2026-07-13
status: approved architecture; written specification awaiting final review
---

# Researcher-first documentation and clean public API design

## Purpose

Jaxstro will present one coherent research-software website for scientists who
may be new to numerical methods, JAX, automatic differentiation, or the
scientific ideas behind the package. The site will not assume that readers know
which prerequisites they are missing. It will provide optional background,
derivations, method chapters, research workflows, API reference, and evidence
without making the main route feel like a course.

This design also fixes the public API boundary before new runtime modules are
implemented. The documentation and the Python namespace must teach the same
ownership model: coherent modules are public; incidental convenience re-exports
are not.

The program is intentionally decomposed into three projects:

1. Harden and reorganize the MyST website, including substantive planned-
   capability pages.
2. Inventory, migrate, and hard-cut the public Python API.
3. Design and implement each future runtime module in its own approved slice.

The first implementation plan covers Project 1. Project 2 receives its own
consumer-aware migration plan. Project 3 does not begin until the website and
public API are hardened.

## Decisions this specification supersedes

This specification supersedes the course-oriented navigation and curriculum
framing in these earlier design documents:

- `2026-07-12-science-first-package-docs-design.md`
- `2026-07-12-scientific-contracts-evidence-curriculum-design.md`

It preserves their evidence-first reasoning cycle, conceptual foundations,
scientific contracts, generated evidence, claim calibration, and executable
research investigations. It replaces their instructor section, rubric framing,
and course-facing language with researcher onboarding, self-checks, audit
guidance, and optional conceptual depth.

## Audience and voice

### Primary reader

The primary reader is a new research student beginning a computational project.
The reader may know astronomy but not numerical analysis, know mathematics but
not JAX, or know programming but not scientific inference. The site must not
require the reader to diagnose that gap before finding useful material.

### Additional readers

- A working scientist looking for one method or API contract.
- A package maintainer checking ownership and compatibility.
- A reviewer auditing a numerical or derivative claim.
- A research mentor linking a student to optional background.

### Voice contract

- Research software first, with optional pedagogical depth.
- Concept-first and supportive without classroom management language.
- Explicit about assumptions, approximations, failure modes, and evidence.
- Astronomy may motivate an idea, but domain policy remains downstream.
- No public sections named course, instructor, lesson, or assessment rubric.
- The words student and researcher are allowed when they describe the audience.
- Authored prose uses ASCII punctuation. Mathematical notation uses LaTeX
  rendered by KaTeX rather than Unicode mathematical symbols.

## The site's organizing idea

The website combines a connected foundations spine with a methods atlas. The
homepage and foundations explain a single worldview:

```text
scientific question
  -> mathematical representation
  -> numerical method
  -> executed JAX program
  -> independent audit
  -> warranted scientific claim
```

The methods, representations, workflows, API, and validation sections are
different ways to enter that same chain. A concept map may visualize the
connections, but the MyST table of contents is the canonical navigation.

## Canonical MyST table of contents

Only top-level source directories are numbered. A top-level section may contain
semantic subdirectories, but the visible TOC must not exceed this depth:

```text
top-level section
  -> semantic subsection
     -> individual page
```

Headings within an individual page belong in the page's secondary navigation.
A semantic subsection is created only when it contains at least three durable,
related pages. Small sections remain flat.

### Home

`docs/index.md` introduces Jaxstro as astro-first, science-general,
evidence-first JAX infrastructure. It presents a compact set of routes rather
than reproducing the whole TOC.

### Start here: begin with a research question

Source root: `docs/00-start-here/`

- `index.md`: what Jaxstro is, what it owns, and where to begin.
- `choose-your-path.md`: routes for computation-first, astronomy-first,
  statistics-first, complete-foundations, and returning researchers.
- `first-research-calculation.md`: a bounded predict -> compute -> audit example.
- `how-to-read-the-site.md`: how concept pages, method pages, API pages, and
  evidence pages relate.

### Foundations: the ideas we will not assume

Source root: `docs/10-foundations/`

`mathematical-objects/`:

- Functions, units, and scales.
- Linear algebra as the language of change.
- What is a derivative?
- Probability and distributions.

`models-and-computation/`:

- What is a model?
- Models, inference, and information.
- Sensitivity, conditioning, and identifiability.
- From mathematical relations to differentiable programs.

The section landing page explains why optional background is rigorous rather
than remedial. The current readiness self-check moves to Start here and is
described as routing, not assessment.

### Numerical methods: ways of turning questions into computations

Source root: `docs/20-methods/`

`change-constraints-evolution/`, titled **Change, constraints, and evolution**:

- Automatic differentiation.
- Scalar root finding.
- Nonlinear systems and fixed points.
- Optimization.
- Differential equations.

`approximation-integration/`, titled **Approximation from finite information**:

- One-dimensional interpolation.
- Regular-grid interpolation.
- B-splines.
- Cumulative integration.
- Quadrature.
- Adaptive quadrature in the JAX ecosystem.

`linear-structure/`, titled **Linear structure and reusable operators**:

- Linear algebra methods.
- Linear operators.
- Iterative linear solvers in the JAX ecosystem.
- Stable special functions and bases.

`probability-sampling/`, titled **Randomness as a computational object**:

- Probability distributions.
- PRNG keys and reproducible random computation.
- Sampling and resampling.
- Quasi-Monte Carlo.

`discrete-space/`, titled **Discrete worlds: grids, meshes, and neighborhoods**:

- Grid construction and conservative binning.
- Structured meshes.
- Spatial search and neighborhood methods.

`signals/`, titled **Signals as sampled evidence**:

- Signal axes, cadence, and units.
- Windows and spectral leakage.
- Power and cross-spectral estimation.
- Phase and delay.

Pages describing Optimistix, Lineax, Quadax, or another external owner are
ecosystem guides, not promises that Jaxstro will duplicate the algorithm.

### Scientific representations: what the computation means

Source root: `docs/30-representations/`

`units-quantities/`:

- Constants and conventions.
- Unit systems.
- Quantities and dimensions.
- Equivalencies and representation changes.

`geometry-coordinates/`:

- Geometry.
- Coordinate transformations.
- Astrometry.

`spectra-atmospheres/`:

- Spectral coordinates and densities.
- Conservative spectral resampling.
- Atmosphere libraries and coverage.
- Source artifacts, adapters, and scientific boundaries.

`parameters-state/`:

- Parameters, constraints, and transformations.
- PyTrees as scientific state.
- Serialization and provenance boundaries.

`uncertainty/`:

- Uncertainty as a representation of incomplete knowledge.
- Linearized covariance propagation.
- Sigma-point propagation.
- Ensemble propagation and diagnostics.

`fields/`:

- Fields as values attached to a domain.
- Coordinates, topology, and discretization.
- Differential and conservative field operators.

The fields pages are conceptual background for a deferred capability. They do
not establish a `jaxstro.fields` API before two real consumers define it.

### Research workflows: from a model to an auditable result

Source root: `docs/40-workflows/`

`scientific-ml/`:

- Preprocessing without data leakage.
- Deterministic data splitting and batching.
- Fixed-step, auditable training.
- What Optax and Equinox own.

`data-pipelines/`:

- Querying atmosphere spectra.
- NEWERA data processing.
- BOSZ data processing.
- Sonora data processing.
- TLUSTY data processing.

`differentiable-research/`:

- What JAX differentiates.
- Auditing derivatives.
- Branches, limits, and implicit sensitivities.

`reproducible-research/`:

- Explicit random-state ownership.
- Scientific contracts and provenance.
- Evidence artifacts and claim boundaries.

`investigations/`:

- Root values and sensitivities.
- Removable limits in finite power laws.
- Interpolation boundary policies.

The existing executable investigations remain research workflows. Their useful
predict -> compute -> audit -> state the warranted claim structure remains, but
instructor notes and grading language are removed.

### API reference: the actual importable surface

Source root: `docs/50-api/`

- `index.md`: public API policy and compact module index.
- `change-constraints/`: autodiff, rootfinding, optimization, and ODE.
- `approximation-integration/`: interpolation, regular-grid interpolation,
  splines, integration, and quadrature.
- `linear-structure/`: linear algebra, operators, and special functions.
- `randomness/`: distributions, random streams, sampling, and statistics.
- `discrete-space/`: grids, meshes, and spatial algorithms.
- `physical-representations/`: constants, units, quantity, coordinates,
  geometry, astrometry, and parameters.
- `scientific-data/`: spectra and atmospheres.
- `research-infrastructure/`: contracts, evidence, provenance, and testing.

The API reference documents only importable, supported surfaces. Planned
modules do not receive fake API pages. Their concept pages may show a clearly
labeled proposed interface, but no proposed symbol appears in the API index.
Each API page names its true Python owner even though the TOC groups pages by
method type. These documentation groups do not create matching Python
namespaces.

### Validation and evidence: why a result should be trusted

Source root: `docs/60-validation/`

- `index.md`: evidence classes and how to read a claim.
- `evidence-index.md`: generated evidence registry.
- `numerical/`: rootfinding performance and implicit-root gradients.
- `data/`: spectra performance and later data-library evidence.
- `methods/`: reusable guides for convergence, limiting cases, AD versus finite
  differences, conservation, and provenance checks.

Validation pages report what evidence demonstrates and what it does not prove.
They do not repeat the conceptual derivation or API signature.

### Project: direction, architecture, and decisions

Source root: `docs/70-project/`

- `index.md`: project identity and ownership boundaries.
- `direction/`: science-general vision, architecture overview, and future
  capabilities.
- `development/`: numerical roadmap, SOTA assessment, package scorecard, and
  development log.
- `decisions/`: ADR index. Individual ADRs remain reachable from the index but
  do not all occupy the primary TOC.
- `release/`: release policy and checklist.
- `bibliography/`: references.

Internal `docs/audits/`, `docs/plans/`, and `docs/superpowers/` remain outside
the published numbering and primary navigation.

## Planned-capability pages

The website will include substantive background pages for every ratified future
capability before runtime implementation. A one-line status announcement is not
sufficient.

Every planned-capability page must contain:

1. The scientific question the capability answers.
2. Prerequisite concepts with links into Foundations.
3. The mathematical objects and core derivation.
4. Assumptions, approximations, and failure boundaries.
5. A small worked conceptual example.
6. Jaxstro's proposed ownership boundary.
7. The external ecosystem owner where Jaxstro delegates machinery.
8. A clearly labeled, non-executable API sketch when useful.
9. The validation evidence required before implementation can be called ready.
10. Connections to related methods, representations, workflows, and evidence.

Each page begins with an `important` admonition stating one of these statuses:

- **Planned Jaxstro capability:** the runtime API does not exist yet.
- **Ecosystem guide:** Jaxstro does not plan to duplicate the runtime method.
- **Deferred abstraction:** the concept is documented, but the module boundary
  awaits concrete consumers.

The initial planned set is:

| Capability | Documentation home | Runtime ownership |
| --- | --- | --- |
| Scientific ML | Research workflows | Future `jaxstro.ml` |
| Quasi-Monte Carlo | Numerical methods | Future `jaxstro.numerics.qmc` |
| Uncertainty propagation | Scientific representations | Future `jaxstro.uncertainty` |
| Signal methods | Numerical methods | Future `jaxstro.signal` |
| Iterative linear solvers | Numerical methods | Lineax and JAX; guide only |
| Nonlinear systems | Numerical methods | Optimistix; guide only |
| Adaptive quadrature | Numerical methods | Quadax; guide only |
| Adaptive differential equations | Numerical methods | Diffrax; guide only |
| Multidimensional fields | Scientific representations | Deferred until two consumers exist |

The initial source map is fixed as follows:

```text
docs/20-methods/change-constraints-evolution/nonlinear-systems.md
docs/20-methods/change-constraints-evolution/adaptive-differential-equations.md
docs/20-methods/approximation-integration/adaptive-quadrature.md
docs/20-methods/linear-structure/iterative-linear-solvers.md
docs/20-methods/probability-sampling/quasi-monte-carlo.md
docs/20-methods/signals/signal-axes.md
docs/20-methods/signals/windows-spectral-leakage.md
docs/20-methods/signals/spectral-estimation.md
docs/20-methods/signals/phase-and-delay.md
docs/30-representations/uncertainty/what-uncertainty-represents.md
docs/30-representations/uncertainty/linearized-propagation.md
docs/30-representations/uncertainty/sigma-point-propagation.md
docs/30-representations/uncertainty/ensemble-propagation.md
docs/30-representations/fields/fields-and-domains.md
docs/30-representations/fields/topology-and-discretization.md
docs/30-representations/fields/field-operators.md
docs/40-workflows/scientific-ml/preprocessing.md
docs/40-workflows/scientific-ml/data-plans.md
docs/40-workflows/scientific-ml/auditable-training.md
docs/40-workflows/scientific-ml/ecosystem-boundaries.md
```

## Method-page contract

Every substantial method page uses the following layered research narrative:

1. **The question this method answers.** Begin with the problem, not the API.
2. **Before computation: what should be true?** State units, signs, scales,
   limits, invariants, conditioning, and expected failures.
3. **Define the mathematical objects.** Name domains, codomains, shapes,
   parameters, and conventions.
4. **Derive the method.** Keep the core conceptual derivation visible.
5. **What the algorithm actually does.** Explain iteration, branch selection,
   stopping, telemetry, and finite-precision behavior.
6. **What JAX differentiates.** Distinguish the mathematical relation, executed
   program, custom rule, and unsupported boundaries.
7. **Using it in Jaxstro.** Show canonical module-qualified imports and a
   bounded example.
8. **How to audit the result.** Link analytic checks, limits, convergence,
   independent methods, and evidence artifacts.
9. **Where the claim stops.** State non-ownership and unsupported conclusions.
10. **Connected ideas.** Link foundations, adjacent methods,
    representations, workflows, API, and validation.

The main derivation, its assumptions, the meaning of the derivative, and the
boundary of the scientific claim must never be hidden in a dropdown. Optional
alternative derivations, repetitive algebra, and advanced telemetry may use a
clearly labeled dropdown.

## MyST visual and semantic grammar

MyST elements carry consistent meaning throughout the site:

- Cards and grids appear on section landing pages, not throughout narrative
  pages.
- LaTeX equations use KaTeX rendering, equation labels, and cross-references.
- Definitions and algorithms identify formal objects and procedures.
- `tip` explains how to choose among methods.
- `important` states assumptions, contracts, or planned status.
- `warning` marks derivative, numerical, data, or failure boundaries.
- `note` provides the big-picture or philosophical connection.
- `seealso` connects related concepts and pages.
- Figures require a caption, stable label, and descriptive alt text.
- Tables compare methods, contracts, assumptions, or evidence classes.
- Dropdowns contain optional depth, never essential reasoning.
- Tabs are rare and reserved for genuinely equivalent views.
- Glossary terms stabilize vocabulary across beginner and reference pages.
- Code examples are captioned when their scientific purpose is not obvious.

Pages must not become a wall of boxes. A page normally has one dominant
narrative and only the semantic callouts that materially improve comprehension.

## Public route migration

Source files may be physically renumbered and regrouped. Meaningful public
routes remain stable through `docs/route-manifest.json`, including routes such
as `/rootfinding`, `/interpolation`, `/foundations`, and evidence pages.

Meaningless generated landing routes are replaced deliberately:

```text
/index-1  -> /start-here
/index-2  -> /methods
/index-3  -> /representations
/index-5  -> /api
/index-7  -> /workflows
/index-8  -> /validation
/index-9  -> /project
```

The obsolete instructor-facing routes are intentionally retired after any
research-relevant content is migrated:

```text
/instructor-resources
/teaching-with-jaxstro
/assessment-rubric
```

The route manifest is the source of truth. No route is inferred merely from a
source directory name.

## Clean public Python API

### Canonical import policy

The package root exposes coherent public modules and a very small number of
package-level constants. It does not re-export every public callable.

Canonical usage is module-qualified:

```python
from jaxstro.numerics import rootfinding

result = rootfinding.safeguarded_bracketed_root(...)
```

or symbol-qualified from the owner:

```python
from jaxstro.numerics.rootfinding import safeguarded_bracketed_root
```

This is not canonical and will be removed:

```python
from jaxstro.numerics import safeguarded_bracketed_root
```

### Namespace rules

- `jaxstro.__all__` lists coherent public submodules plus explicitly approved
  package-level constants such as `DEFAULT_UNITS`.
- `jaxstro.numerics.__all__` lists coherent numerical submodules only.
- A symbol is publicly owned by exactly one module.
- Types, statuses, constants, and result records remain with the method that
  defines their semantics.
- Documentation examples use only canonical owner paths.
- `jaxstro.methods` and `jaxstro.representations` are documentation concepts,
  not new Python namespaces.
- No permanent compatibility aliases remain after migration.

### Migration sequence

1. Generate a complete public-symbol inventory from the current package.
2. Classify every symbol as public, internal, or accidental export.
3. Canonicalize Jaxstro documentation, examples, and tests.
4. Audit sibling packages for flat imports and prepare explicit migrations.
5. Migrate and verify sibling consumers in their own repositories.
6. Reduce `jaxstro.numerics.__init__` to coherent submodule exposure.
7. Hard-cut retired flat exports without a permanent deprecation layer.
8. Regenerate API reference, contract coverage, and downstream evidence.

Because sibling packages are separate repositories, the hard cut is a distinct
project with an explicit multi-repository checkpoint. The website redesign may
teach the canonical imports before the runtime hard cut, but it must label the
inventory state honestly until the cut is complete.

## Future runtime ownership

### Future `jaxstro.ml`

The initial package is deliberately thin:

```text
jaxstro.ml
|-- preprocessing
|-- data
`-- training
```

It owns host-fit/JAX-apply preprocessing, deterministic split and batch plans,
fixed-shape fixed-step training state, and auditable training traces. It uses
Optax-compatible optimizer protocols and Equinox-compatible PyTrees. It does
not own optimizers, a model zoo, general losses, posterior inference, or a
second provenance system.

### Future `jaxstro.numerics.qmc`

It owns JAX-native Sobol and Latin-hypercube construction, replicated scrambles,
discrepancy diagnostics, explicit key ownership, and fixed-shape behavior. It
must be verified against established reference sequences and convergence tests.

### Future `jaxstro.uncertainty`

It owns linearized covariance pushforward, sigma-point propagation, keyed
ensemble propagation, and conditioning diagnostics. It propagates uncertainty
through scientific maps; it does not construct posteriors or duplicate NumPyro,
BlackJAX, or Informax.

For a differentiable map

```{math}
:label: eq-future-covariance-pushforward

\mathbf{y} = f(\mathbf{x}),
```

the local first-order covariance contract begins from

```{math}
:label: eq-future-covariance-linearized

\mathbf{C}_{y}
\approx
\mathbf{J}\,\mathbf{C}_{x}\,\mathbf{J}^{\mathsf{T}},
\qquad
\mathbf{J}
=
\frac{\partial f}{\partial \mathbf{x}}.
```

The documentation must derive the approximation and state where local
linearization fails.

### Future `jaxstro.signal`

It owns scientific conventions around sample axes, cadence, windows, one-sided
and two-sided spectra, amplitude and power normalization, equivalent noise
bandwidth, phase, and delay. JAX owns FFT and convolution mechanics. Jaxstro
must not wrap those functions without adding a scientific contract.

### Delegated solver capabilities

- Lineax and JAX own general iterative linear solvers.
- Optimistix owns general nonlinear root, fixed-point, least-squares, and
  minimization solvers.
- Quadax owns adaptive quadrature.
- Diffrax owns adaptive ODE, SDE, and CDE solving.
- Optax owns optimizer transformations.
- Equinox owns neural-network and callable-PyTree construction.

Jaxstro may later add a small adapter only when a concrete consumer requires
one and the adapter adds units, shape, provenance, telemetry, or evidence that
the external owner does not provide. An adapter is not permission to fork the
underlying algorithm.

## Documentation data flow

```text
runtime source and __all__
  -> public-symbol inventory
  -> contract registry
  -> generated API summaries
  -> hand-authored concept and method pages
  -> evidence and validation links
  -> MyST build
  -> route and rendered-site checks
```

Generated inventories answer what exists. Hand-authored pages answer why it
exists, how it works, and what can be concluded. Evidence pages answer what has
been checked. These roles must not be collapsed into one omnibus page.

## Failure behavior

The documentation gate fails when:

- a TOC entry or route-manifest entry points to a missing page;
- a meaningful retained route disappears;
- a generated API or contract page is stale;
- a public API page names a symbol that is not importable from its documented
  owner;
- a planned capability is presented as implemented;
- a method page hides essential assumptions or derivative meaning;
- a new figure lacks alt text, a caption, or a stable label;
- a new mathematical symbol is written as decorative Unicode rather than
  LaTeX;
- retired course or instructor framing returns to the published navigation;
- strict MyST produces unresolved references or content errors.

Historical internal plans and audits may preserve their original wording. The
public site and new source material follow the new research-software contract.

## Verification strategy

### Website project

- Run the strict `scripts/check_docs.sh` gate.
- Verify every moved page and retained route through
  `docs/route-manifest.json`.
- Add targeted integration tests for the new TOC groups, planned-capability
  status boxes, retired instructor navigation, and meaningful landing routes.
- Search published sources for obsolete course and instructor framing.
- Check new pages for Unicode mathematical symbols and decorative icons.
- Verify equation, figure, table, glossary, and cross-reference rendering in
  the built site.
- Inspect the built navigation at desktop and narrow widths.
- Run a keyboard and alt-text accessibility pass.
- Confirm the deployed GitHub Pages routes over live HTTP after release.

### API project

- Test the generated symbol inventory against actual imports.
- Test that canonical owner paths import under supported Python versions.
- Test that removed flat imports fail after the hard cut.
- Run the full Jaxstro quality gate.
- Run focused gates in every migrated sibling package.
- Regenerate contract and evidence indexes after the namespace change.

### Future-module projects

Each module receives a separate scientific design, tests written around its
contract, reference comparisons, JAX transform checks, failure-state tests,
documentation derivations, and evidence artifacts. A planned-capability page
does not count as implementation evidence.

## Delivery sequence and approval gates

```text
1. Researcher-first website migration and planned-capability background
   -> full local docs verification
   -> user visual and content review
   -> deployed route verification

2. Clean API inventory and multi-repository migration
   -> user review of the per-symbol classification
   -> downstream verification
   -> hard cut

3. Runtime capability slices
   -> jaxstro.ml
   -> jaxstro.numerics.qmc
   -> jaxstro.uncertainty
   -> jaxstro.signal
```

The module order is the current recommendation, not permission to implement all
modules in one change. Each runtime capability remains its own approval-gated
design and implementation unit.

## Completion criteria for Project 1

The website hardening project is complete when:

- the MyST TOC uses the approved eight-route architecture;
- source folders use the approved numbering and semantic subsections;
- foundations remain complete, optional, and connected;
- course and instructor framing is absent from the published website;
- every current method has a clearly grouped conceptual home;
- every ratified future capability has a substantive background page with an
  honest status and ownership boundary;
- the API section is grouped by actual module owners and uses canonical imports;
- MyST elements follow the semantic grammar in this specification;
- equations use LaTeX and KaTeX throughout;
- meaningful public routes remain stable and new landing routes are readable;
- all local documentation gates pass;
- the rendered website receives user review before publication;
- the published site is verified over live HTTP.

## Non-goals for Project 1

- No new runtime module implementation.
- No fake API reference for planned symbols.
- No general solver implementation already owned by the JAX ecosystem.
- No redesign of scientific contract or evidence schemas unless the source
  migration exposes a concrete defect.
- No separate textbook, instructor portal, or course export.
- No navigation system parallel to the MyST TOC.
