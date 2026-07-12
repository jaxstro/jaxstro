---
title: Science-first package and curriculum documentation design
date: 2026-07-12
status: approved
---

# Science-first package and curriculum documentation design

## Objective

Make the Jaxstro website a state-of-the-art, evidence-first guide to
differentiable computational science while preserving its durable module-level
theory, API, architecture, validation, and how-to sections.

The redesign must serve three readers without duplicating the site:

1. a scientist deciding whether Jaxstro can support a real calculation;
2. a developer auditing an API, transform boundary, or numerical claim;
3. a computational-science student learning how equations become trustworthy
   software.

## Controlling principles

- Preserve the existing Diátaxis structure and all module sections.
- Add a science-first conceptual spine; do not replace reference material with
  marketing prose.
- Connect every strong claim to executable evidence.
- Separate generic numerical machinery from downstream scientific ownership.
- State AD contracts explicitly: smooth iterative map, branch-selected value
  map, certified implicit derivative, validation-only, or fail-closed.
- Use astronomy as the proving ground while naming science-general patterns.
- Do not invent downstream results, performance claims, or citations.

## Entry-gate corrections

The redesign begins with two code-review fixes:

1. `map_safeguarded_bracketed_root` must reject endpoint arrays that are not
   one-dimensional batch vectors. A mapped scalar solve must never receive an
   accidental trailing endpoint dimension.
2. Implicit-root gradient evidence must be generated and checked by a
   reproducible script. Its test must recompute deterministic scientific
   metrics rather than only checking schema and loose bounds. Environment and
   revision metadata must be refreshed on emission.

ADR 0008 must also clarify that it rejects the heavy Information Field Theory
framework from core. It does not prohibit the dependency-free implicit function
theorem rule used by the separately certified scalar-root API.

## Information architecture

### Homepage

The homepage will retain the three entry doors but lead with scientific
capabilities:

- express quantities and conventions explicitly;
- construct JAX-native numerical maps with named AD behavior;
- locate events, equilibria, and inverse mappings robustly;
- connect tabulated scientific data to differentiable models;
- audit claims through independent validation and provenance.

A compact “science enabled” section will link equations and scientific tasks to
the relevant module chapters and evidence anchors.

### How to learn with Jaxstro: predict, compute, audit

Add a first-class Getting Started page titled **How to learn with Jaxstro:
predict, compute, audit**. This is the clearest framing for the primary audience:
research students and students in computational-science courses. It is more
actionable than a generic “Why this matters” page and less inward-looking than
“Why the documentation is designed this way.”

The page explains the site's recurring reasoning cycle:

1. **Predict** the mathematical behavior before execution: units, signs,
   limiting cases, invariants, conditioning, likely failure modes, and whether a
   derivative should exist.
2. **Compute** with an explicit method whose tolerances, branches, shapes, and
   telemetry are visible rather than hidden behind a plausible scalar answer.
3. **Audit** the result against analytic identities, independent numerical
   checks, convergence behavior, provenance, and the exact boundary of the
   method's scientific claim.

The page must explain why the order matters. Prediction prevents post-hoc
storytelling; computation connects equations to an executed algorithm; audit
distinguishes a reproducible result from a merely finite output. It must also
show that the cycle is iterative: a failed audit changes the model, method,
tolerance, or claim and begins a new prediction.

Use one compact rootfinding example and one smooth-distribution limiting example
to demonstrate the cycle. Link each stage to theory, API, and validation pages.
The page becomes the curriculum bridge from installation to the module chapters
and supplies a reusable callout pattern for later lessons.

### Science-enabled patterns page

Add one cross-cutting page organized by scientific pattern rather than module:

- locate an event or equilibrium;
- differentiate a certified equilibrium;
- integrate a conserved or accumulated quantity;
- interpolate a tabulated physical model;
- draw from a finite distribution smoothly across a limiting parameter;
- transform coordinates, units, and spectral densities;
- find local spatial interactions;
- connect model PyTrees to inference parameters;
- preserve provenance from source artifact to numerical claim.

Each example follows this template:

1. scientific question;
2. governing equation or invariant;
3. generic Jaxstro primitive;
4. supported JAX transforms;
5. failure or rejection state;
6. validation evidence;
7. downstream ownership boundary.

### Module chapters

All existing module chapters remain. They become the stable instructional and
technical units beneath the cross-cutting science page. Where appropriate, a
module chapter gains:

- learning objectives;
- prerequisites;
- a motivating observable or scientific question;
- a worked example;
- an AD and boundary-contract box;
- a validation anchor;
- a short “what this does not prove” section;
- links back to related science patterns.

The API reference remains concise and signature-oriented. Validation remains a
claim-to-evidence index rather than becoming a second theory section.

## Curriculum layer

The site may be used directly in computational-science courses without creating
a separate textbook tree. Curriculum affordances will be embedded selectively:

- **Learning objectives** at the start of substantial theory chapters.
- **Predict → compute → audit** activities that ask students to anticipate a
  failure mode, run or inspect a calculation, and judge the evidence.
- **Concept checks** on units, conditioning, discretization, branch selection,
  and gradient interpretation.
- **Worked examples** that distinguish mathematical truth from the executed
  numerical map.
- **Extension prompts** that connect generic methods to astronomy, geoscience,
  mechanics, instrumentation, and inverse problems.
- **Instructor notes** only where they add real value; these should be visually
  distinct and not interrupt the reference path.

The first curriculum-quality reference chapter is scalar root finding. It will
teach robust forward solves, telemetry, cost masking, exact-root precedence,
and the difference between differentiating an algorithm and applying the
implicit function theorem under a certificate.

Every substantial module chapter should use the three labels consistently when
an activity is present. The labels describe epistemic work, not interface steps:
students must state an expectation, inspect the executed numerical map, and
decide what the evidence warrants claiming.

## Rootfinding science examples

The rootfinding chapter and science-pattern page will use bounded examples:

- a symmetric endpoint consistency equation;
- an optical-depth or photosphere surface equation;
- a Kepler or event-surface equation;
- equilibrium or boundary matching;
- an implicit sensitivity such as `f(x, theta) = x**2 - theta`;
- a nonsmooth or flat-root counterexample that must fail the derivative gate.

Examples describe scientific patterns, not downstream package runtime policy.

## Visual system

Figures use consistent semantics:

- blue: verified mathematical or bracket evidence;
- amber: proposed or branch-selected numerical state;
- green: certified accepted result;
- red: rejected, invalid, or unsupported claim;
- gray: masked or externally owned state.

Every numerical figure must have a source-controlled generator or a clearly
documented deterministic construction. Every figure needs descriptive alt text,
legible labels, and a caption stating what is demonstrated and what is not.

Priority figures:

1. **Safeguarded root trace** — residual curve, contracting sign bracket,
   IQI/secant proposals, midpoint fallbacks, and terminal status.
2. **Value map versus implicit derivative** — branch-selected scan on one side;
   certified IFT sensitivity and its assumptions on the other.
3. **Derivative certificate dashboard** — uniqueness assertion, smoothness
   assertion, convergence, finiteness, residual, width, and slope gates.
4. **Power-law removable singularity** — naive branch and smooth kernels through
   `alpha=-1`, including derivative continuity.
5. **AD contract map** — smooth, piecewise-smooth, discrete, value-first,
   certified implicit, validation-only, and fail-closed APIs.
6. **Validation triangle** — analytic result, AD, and independent finite
   difference, with provenance and tolerance attached.
7. **Foundation ownership flow** — generic Jaxstro primitives feeding multiple
   downstream scientific packages without absorbing their runtime policy.
8. **Units/data/physics boundaries** — quantities, spectra, atmospheres, and
   domain interpretation as distinct layers.

The first implementation tranche should produce the root trace and value-versus-
IFT figures. The remaining concepts are prioritized in the package-wide roadmap
and should not be rendered as decorative filler.

## Package-wide SOTA assessment

Assess the complete package and website across:

- scientific breadth and ownership discipline;
- numerical robustness and conditioning;
- AD honesty and transform coverage;
- dimensional safety;
- API cohesion and serialization;
- performance and compilation evidence;
- evidence freshness and provenance;
- curriculum quality and discoverability;
- accessibility and visual communication;
- downstream adoption and cross-project reuse.

For each category record:

1. delivered strengths with file and test evidence;
2. high-confidence gaps;
3. one or two highest-impact next investments;
4. the evidence required before claiming completion.

The assessment should rank work into three horizons:

- **Now:** correctness, stale evidence, misleading contracts, and navigation.
- **Next:** reference-quality science stories and figures across major modules.
- **Later:** larger capabilities that require separate scientific validation or
  would materially expand the foundation boundary.

## Verification

The implementation must run:

- focused rootfinding, implicit-gradient, evidence, API, and documentation tests;
- affected module tests for every edited science example;
- Ruff format/check and full-source MyPy;
- evidence emit/check commands;
- MyST build with the existing page-count, route, and warning gates;
- an accessibility/content inspection of new figures and callouts;
- a final code and scientific-claims review.

Measured scientific values must remain in metric tables with identity, symbol,
value, and units.

## Non-goals

- Do not replace module documentation with a narrative landing page.
- Do not turn Jaxstro into a domain simulator or general solver stack.
- Do not claim every API is differentiable.
- Do not add a heavy documentation framework or plotting dependency without a
  separate decision.
- Do not fabricate downstream validation to make examples appear complete.
