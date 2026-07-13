# Executable Foundations Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Use targeted subagents only at the B1 and B4 review checkpoints.

**Goal:** Add an optional, self-contained computational-science foundations spine and three executable research investigations for research students and courses, while preserving Jaxstro's module-by-module theory and API documentation.

**Architecture:** Add three new documentation layers around the existing module reference: `05-foundations` supplies readiness recovery and first-principles concepts, `70-investigations` applies public Jaxstro APIs through predict → compute → audit, and `80-instructor` supplies teaching guidance and assessment. Repository-owned investigation modules are executable source of truth; MyST pages explain them. A curriculum manifest links units to scientific contracts and indexed evidence without moving astronomy-specific runtime policy into Jaxstro.

**Tech Stack:** MyST Markdown, Python 3.11+, JAX, public `jaxstro` APIs, JSON, pytest, Ruff, MyPy. No new dependency.

## Global constraints

- Work inline in the normal checkout; preserve unrelated and untracked changes.
- Use `env -u VIRTUAL_ENV uv run --no-sync` for every Python, pytest, Ruff, and MyPy command.
- TDD every executable, navigation, generated-registry, and content contract.
- Keep the existing `10-theory` module chapters and `40-api` reference intact.
- Primary audience: research students and computational-science or astronomy courses with uneven preparation.
- Use astronomy as teaching context, never as Jaxstro runtime ownership.
- Every substantial unit follows `predict → compute → audit → state the warranted claim`.
- Every measured numerical result appears in a four-column metric table with identity, symbol, value, and units.
- Examples import public APIs; copied code blocks and notebooks cannot become a second source of truth.
- Commit each checkpoint-sized slice.

---

## B1 — Foundations framework and readiness routing

### Task 1: Add the documentation-design rationale

**Files:**
- Create: `docs/05-foundations/why-this-documentation-works-this-way.md`
- Modify: `docs/00-getting-started/how-to-learn.md`
- Modify: `docs/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Create: `tests/integration/test_foundations_docs.py`

**Content contract:** Title the page **Why this documentation works this way**. Explain that prerequisite completion is not the same as current preparedness; optional recovery is rigorous rather than remedial; prediction prevents post-hoc storytelling; computation exposes the executed numerical map; audit limits the claim; module reference and curriculum routes answer different questions.

- [ ] Write failing tests for the title, audience, predict-compute-audit-claim cycle, module-section preservation, navigation, and route.
- [ ] Run the RED test.
- [ ] Author and link the page; retain all existing module entries.
- [ ] Run the focused GREEN test and docs route test.
- [ ] Commit: `docs: explain the evidence-first learning design`.

### Task 2: Add an ungraded choose-your-path readiness route

**Files:**
- Create: `docs/05-foundations/foundations.md`
- Create: `docs/05-foundations/choose-your-path.md`
- Modify: `docs/myst.yml`, `docs/route-manifest.json`
- Modify: `tests/integration/test_foundations_docs.py`

**Route contract:** Short, ungraded self-checks route learners to foundations by concept and research task. Do not assign readiness labels or scores. Include paths for computation-first learners, astronomy students, statistics/inference students, and returning researchers.

- [ ] Add failing content/navigation tests.
- [ ] Implement the index and readiness router.
- [ ] Verify and commit: `docs: add optional foundations readiness routes`.

### Checkpoint B1

Use one targeted pedagogy/navigation reviewer. Resolve all Critical and Important findings before B2.

---

## B2 — Complete conceptual foundations spine

### Task 3: Functions, units, scales, models, parameters, and dimensionality

**Files:**
- Create: `docs/05-foundations/functions-units-scales.md`
- Create: `docs/05-foundations/what-is-a-model.md`
- Create: `tests/integration/test_foundations_concepts.py`
- Modify: navigation and route manifest

**Required science examples:** Newtonian gravity for units/signs/scales; Stefan–Boltzmann luminosity for a model map; parallax for inversion; spectra as high-dimensional observations with lower-dimensional structure.

**Model page contract:** Define conceptual, mathematical, computational, statistical, generative, and surrogate models. Explain assumptions, parameters, state, latent variables, observables, hyperparameters, nuisance parameters, information compression, and spatial versus parameter/data/state/intrinsic/effective/model dimensionality.

- [ ] Write and run failing concept and navigation tests.
- [ ] Author both pages with Predict, Compute, Audit, misconception, and warranted-claim blocks.
- [ ] Verify and commit: `docs: teach models dimensions units and scales`.

### Task 4: Linear algebra as the language of change

**Files:**
- Create: `docs/05-foundations/linear-algebra-language-of-change.md`
- Modify: `docs/10-theory/linear-algebra.md`
- Modify: tests/navigation

**Contract:** Build from vectors as perturbations and linear maps before matrices. Cover basis, dot products, norms, projection, Jacobians, covariance geometry, eigenvectors, singular vectors, null spaces, condition numbers, quadratic forms, and Hessians. The existing module chapter remains the numerical-method reference and links back.

- [ ] TDD content and reciprocal-link contracts.
- [ ] Author and verify.
- [ ] Commit: `docs: teach linear algebra as local change`.

### Task 5: What a derivative is

**Files:**
- Create: `docs/05-foundations/what-is-a-derivative.md`
- Modify: `docs/10-theory/autodiff.md`, `docs/10-theory/rootfinding.md`
- Modify: tests/navigation

**Contract:** Connect three views: derivative as local change/rate, derivative as the best local linear map, and derivative as scientific sensitivity/evidence. Develop scalar derivatives, directional derivatives, gradients, Jacobians, JVPs, VJPs, tangent/cotangent language, likelihood scores, Hessians, Fisher information, implicit sensitivities, and the derivative of the executed program. Use Stefan–Boltzmann, gravity, and implicit roots. State branch and nonsmooth boundaries explicitly.

- [ ] TDD required concepts and reciprocal links.
- [ ] Author layered core plus optional deeper sections.
- [ ] Verify and commit: `docs: explain derivatives from first principles`.

### Task 6: Probability, inference, information, sensitivity, and differentiable programs

**Files:**
- Create: `docs/05-foundations/probability-and-distributions.md`
- Create: `docs/05-foundations/models-inference-information.md`
- Create: `docs/05-foundations/sensitivity-conditioning-identifiability.md`
- Create: `docs/05-foundations/from-relations-to-differentiable-programs.md`
- Modify: tests/navigation

**Probability contract:** mass versus density, support, normalization, expectation, covariance, conditioning, transformations, sampling, aleatoric versus epistemic uncertainty.

**Inference contract:** physical model, observable, measurement model, likelihood, prior, posterior, predictive distribution, nuisance parameters, model checking. Distinguish retained data, Shannon information, and parameter information. State that discarded information cannot be recovered by optimization and precision does not imply adequacy.

**Sensitivity contract:** conditioning, identifiability, degeneracy, null directions, local versus global sensitivity, finite differences versus AD, and evidence required for derivative claims.

**Programming contract:** relation versus executed program, control flow, fixed scans, PyTrees, JIT/VMAP, value-first versus implicit derivatives, and scientific ownership boundaries.

- [ ] TDD all content/navigation contracts.
- [ ] Author the four linked pages.
- [ ] Run focused content and docs tests.
- [ ] Commit: `docs: complete the computational science foundations spine`.

---

## B3 — Executable research investigations

### Task 7: Create the investigation result and manifest contracts

**Files:**
- Create: `examples/investigations/__init__.py`
- Create: `examples/investigations/_common.py`
- Create: `docs/curriculum/units.json`
- Create: `scripts/build_curriculum_registry.py`
- Create: `tests/unit/test_curriculum_registry.py`
- Create: `docs/70-investigations/index.md` (generated)
- Create: `docs/validation/curriculum-coverage.json` (generated)
- Modify: checks/navigation/routes

**Interfaces:** A frozen `InvestigationResult` holds a unit ID, prediction, metric rows, audit checks, and warranted claim. The curriculum manifest names prerequisites, public APIs, contract IDs, evidence IDs, limitations, and instructor route. The builder validates references against `contracts.json` and `evidence-index.json`, then emits deterministic index and coverage artifacts.

- [ ] Write failing schema, missing-reference, freshness, and navigation tests.
- [ ] Implement minimal records and registry builder.
- [ ] Verify and commit: `feat: define executable curriculum contracts`.

### Task 8: Root values and sensitivities investigation

**Files:**
- Create: `examples/investigations/root_values_and_sensitivities.py`
- Create: `docs/70-investigations/root-values-and-sensitivities.md`
- Create: `tests/integration/test_root_investigation.py`
- Modify: curriculum manifest/generated outputs

Use a dimensionally explained analytic residual and public value-first/certified APIs. Require root, residual, bracket width, AD/analytic error, and certificate metrics; show why the executed branch-selected solver is not the IFT derivative.

- [ ] TDD deterministic metrics, JIT behavior, audit checks, and claim text.
- [ ] Implement example and page.
- [ ] Regenerate, verify, commit: `feat: add executable root sensitivity investigation`.

### Task 9: Finite power-law removable-singularity investigation

**Files:**
- Create: `examples/investigations/powerlaw_removable_limit.py`
- Create: `docs/70-investigations/powerlaw-removable-limit.md`
- Create: `tests/integration/test_powerlaw_investigation.py`
- Modify: curriculum manifest/generated outputs

Use public normalization/logpdf/CDF/PPF APIs. Require normalization, CDF/PPF round trip, analytic limiting derivative, central-FD/AD error, and support-boundary checks through `alpha=-1`.

- [ ] TDD, implement, regenerate, verify.
- [ ] Commit: `feat: add executable removable-limit investigation`.

### Task 10: Interpolation policy investigation

**Files:**
- Create: `examples/investigations/interpolation_boundary_policies.py`
- Create: `docs/70-investigations/interpolation-boundary-policies.md`
- Create: `tests/integration/test_interpolation_investigation.py`
- Modify: curriculum manifest/generated outputs

Use public 1D and regular-grid interpolation APIs. Predict clamp/fill/reject behavior, compute branch-stable interior derivatives, and audit values against an affine analytic function. Do not generalize interior derivative evidence to knots or policy boundaries.

- [ ] TDD, implement, regenerate, verify.
- [ ] Commit: `feat: add executable interpolation policy investigation`.

---

## B4 — Instructor guidance, assessment, and closeout

### Task 11: Add instructor notes and a claim-calibrated rubric

**Files:**
- Create: `docs/80-instructor/instructor-resources.md`
- Create: `docs/80-instructor/teaching-with-jaxstro.md`
- Create: `docs/80-instructor/assessment-rubric.md`
- Create: `tests/integration/test_instructor_docs.py`
- Modify: navigation/routes/curriculum manifest

Rubric dimensions: prediction quality, model/units clarity, method inspection, independent audit, derivative interpretation, provenance, failure analysis, and warranted claim. Include common misconceptions, facilitation notes, accessibility, optional prerequisite recovery, and extension prompts for astronomy and computational-science courses.

- [ ] TDD content and navigation.
- [ ] Author, verify, and commit: `docs: add instructor guidance and evidence rubric`.

### Task 12: Close Phase B

**Files:**
- Modify: `docs/90-development-log/package-assessment-scorecard.md`
- Modify: `docs/90-development-log/sota-assessment.md`
- Modify: `STATUS.md`
- Modify: assessment tests

- [ ] Derive curriculum counts from `curriculum-coverage.json` in tests.
- [ ] Record delivered versus remaining visual/instructor/export limitations without automatic grade inflation.
- [ ] Run focused investigation/content tests, generated checks, Ruff, MyPy, and the complete docs gate.
- [ ] Use two targeted final reviewers: pedagogy/accessibility and scientific/API honesty. Resolve all Critical and Important findings.
- [ ] Commit: `docs: close executable foundations curriculum phase`.

## Completion criteria

- Existing module theory and API sections remain navigable and unchanged in ownership.
- Foundations provide a broad optional on-ramp without readiness labels.
- The derivative, model, dimensionality, probability, inference, sensitivity, and linear-algebra pages are connected from first principles.
- Three deterministic investigations execute public APIs and produce auditable metric tables.
- Curriculum references resolve to scientific contracts and indexed evidence.
- Instructor notes and claim-calibrated assessment exist.
- No downstream import, new dependency, or astronomy-specific runtime logic is added.
