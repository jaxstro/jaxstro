# Science-First Curriculum Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Jaxstro's website into an evidence-first computational-science learning environment for research students and courses while preserving its module reference structure and fixing the two code-review defects that would undermine its claims.

**Architecture:** Work in three ratcheted tranches. First repair mapped-scalar input validation and make implicit-gradient evidence reproducible. Then add the predict-compute-audit learning spine, science-pattern navigation, and two public-API-derived figures, using the existing MyST and JaxtroViz infrastructure. Finally audit every package module and documentation family, publish a ranked SOTA roadmap, and verify the full site without claiming unvalidated science.

**Tech Stack:** Python 3.11+, JAX, jaxtyping, pytest, Ruff, MyPy, MyST Markdown, Matplotlib/Seaborn/Pillow through the existing `viz` extra, JaxtroViz figure registry.

## Global Constraints

- Work inline in `/Users/anna/projects/jaxstro-dev/jaxstro`; do not create a worktree.
- Preserve all existing and untracked changes; never reset, clean, or overwrite unrelated work.
- Use `env -u VIRTUAL_ENV uv run --no-sync` for every Python, pytest, Ruff, and MyPy command.
- Use TDD for code, evidence, navigation, and figure contracts.
- Preserve the existing Diátaxis and module-level documentation structure.
- The primary audience is research students and computational-science courses.
- Explain and consistently apply **predict → compute → audit** as epistemic work, not UI steps.
- Keep domain interpretation and downstream runtime policy out of Jaxstro.
- Every numerical figure must be derived from public APIs or explicit analytic identities.
- Every measured numerical result must be reported in a table with metric identity, symbol, value, and units.
- Use source-controlled JaxtroViz builders, accessible alt text, deterministic exports, and existing visual semantics.
- Do not add dependencies; Matplotlib, Seaborn, and Pillow remain optional under the existing `viz` extra.
- Commit each independently reviewable task.

---

## File responsibility map

- `src/jaxstro/numerics/rootfinding.py`: scalar, mapped, and certified root APIs; only endpoint-rank validation changes here.
- `scripts/benchmark_implicit_root.py`: canonical emitter/checker for implicit-root analytic, AD, and central-FD evidence.
- `docs/validation/implicit-root-gradients.json`: generated evidence artifact; never hand-maintained after Task 2.
- `docs/00-getting-started/how-to-learn.md`: predict-compute-audit philosophy and first curriculum activity.
- `docs/10-theory/science-patterns.md`: scientific-question-to-module routing layer.
- `docs/10-theory/rootfinding.md`: reference-quality rootfinding lesson and worked scientific patterns.
- `laboratory/jaxtroviz/rootfinding.py`: public-API-derived root trace and value-versus-IFT figures.
- `laboratory/jaxtroviz/registry.py`: figure metadata and deterministic site paths.
- `docs/10-theory/figures/`: rendered site assets only; builders remain in `laboratory/jaxtroviz`.
- `docs/20-architecture/science-general-vision.md`: package-wide scientific ownership and admission rules.
- `docs/60-validation/index.md`: concise claim-to-evidence routing, not duplicated theory.
- `docs/90-development-log/sota-assessment.md`: evidence-backed package and website assessment with Now/Next/Later priorities.
- `docs/index.md`, `docs/myst.yml`, `docs/route-manifest.json`: homepage and navigation integration.
- `tests/unit/` and `tests/integration/`: executable contracts for every code, evidence, page, route, and figure claim.

---

### Task 1: Reject non-vector mapped root endpoints

**Files:**
- Modify: `src/jaxstro/numerics/rootfinding.py:546-570`
- Modify: `tests/unit/test_bracketed_root.py`

**Interfaces:**
- Consumes: `map_safeguarded_bracketed_root(f, args, lo, hi, *, ...)`.
- Produces: an eager `ValueError` unless `lo.ndim == hi.ndim == 1`, before `lax.map` tracing.

- [ ] **Step 1: Add the failing endpoint-rank tests**

```python
@pytest.mark.parametrize(
    ("lo", "hi"),
    [
        (jnp.asarray(0.0), jnp.asarray(4.0)),
        (jnp.zeros((2, 1)), jnp.ones((2, 1))),
        (jnp.zeros(2), jnp.ones((2, 1))),
    ],
)
def test_lax_map_rejects_nonvector_endpoint_arrays(lo, hi) -> None:
    with pytest.raises(ValueError, match="one-dimensional batch vectors"):
        rootfinding.map_safeguarded_bracketed_root(
            lambda x, target: x * x - target,
            jnp.asarray([1.0, 2.0]),
            lo,
            hi,
            max_steps=8,
        )
```

- [ ] **Step 2: Run the RED gate**

Run:

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_bracketed_root.py -k nonvector
```

Expected: scalar inputs reach the old “leading dimension” error and trailing-rank inputs do not fail with the required contract.

- [ ] **Step 3: Implement exact rank validation**

Replace the current `< 1` condition with:

```python
if lo.ndim != 1 or hi.ndim != 1:
    raise ValueError("lo and hi must be one-dimensional batch vectors")
```

Keep the existing leading-length and `args`-leaf checks unchanged.

- [ ] **Step 4: Run focused GREEN and transform regression tests**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_bracketed_root.py \
  tests/validation/test_bracketed_root_algorithms.py
```

Expected: all pass, including scalar/map parity and fixed trace shapes.

- [ ] **Step 5: Run Ruff/MyPy and commit**

```bash
env -u VIRTUAL_ENV uv run --no-sync ruff check \
  src/jaxstro/numerics/rootfinding.py tests/unit/test_bracketed_root.py
env -u VIRTUAL_ENV uv run --no-sync mypy \
  src/jaxstro/numerics/rootfinding.py
git add src/jaxstro/numerics/rootfinding.py tests/unit/test_bracketed_root.py
git commit -m "fix(numerics): require vector root-map endpoints"
```

---

### Task 2: Make implicit-gradient evidence reproducible

**Files:**
- Create: `scripts/benchmark_implicit_root.py`
- Modify: `tests/unit/test_implicit_root_evidence.py`
- Regenerate: `docs/validation/implicit-root-gradients.json`
- Modify: `docs/60-validation/index.md`

**Interfaces:**
- Consumes: `implicit_bracketed_root`, `ImplicitRootAssumptions`, and the linear, quadratic, and exponential validation cases.
- Produces: `run_benchmark() -> dict[str, Any]`, `--emit`, `--check`, schema version 2, and deterministic algorithmic freshness checks.

- [ ] **Step 1: Replace the bounds-only test with failing recomputation contracts**

Add imports through `importlib.util` or a normal script-module helper and require:

```python
def test_implicit_root_evidence_matches_fresh_algorithmic_metrics() -> None:
    stored = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    current = benchmark_implicit_root.run_benchmark()
    assert benchmark_implicit_root.algorithmic_metrics_match(stored, current)


def test_implicit_root_evidence_records_clean_environment_fields() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert set(payload["environment"]) == {
        "device",
        "git_revision",
        "jax_backend",
        "jax_version",
        "measured_at_utc",
        "platform",
        "python_version",
        "working_tree_dirty",
    }
```

- [ ] **Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_implicit_root_evidence.py
```

Expected: fail because no canonical benchmark module or comparison function exists.

- [ ] **Step 3: Implement the emitter/checker**

Create `scripts/benchmark_implicit_root.py` with:

```python
CASES = (
    ("linear", lambda x, theta: x - theta, 2.0, 1.0),
    ("quadratic", lambda x, theta: x * x - theta, 2.0, 1.0 / (2.0 * math.sqrt(2.0))),
    ("exponential", lambda x, theta: jnp.exp(x) - theta, 2.0, 0.5),
)
FD_STEP = 1.0e-5


def _measure_case(name, residual, theta, analytic):
    solve = lambda value: implicit_bracketed_root(
        residual,
        jnp.asarray(value, dtype=jnp.float64),
        0.0,
        4.0,
        assumptions=ImplicitRootAssumptions(True, True),
        max_steps=96,
        atol=1.0e-14,
        rtol=1.0e-14,
        derivative_residual_atol=1.0e-12,
        derivative_width_atol=1.0e-12,
        derivative_slope_floor=1.0e-8,
    )
    result = solve(theta)
    ad = jax.grad(lambda value: solve(value).root)(theta)
    fd = (solve(theta + FD_STEP).root - solve(theta - FD_STEP).root) / (2.0 * FD_STEP)
    return measured_case_dict(name, result, analytic, ad, fd)
```

Implement `run_benchmark`, `_validate`, `algorithmic_metrics_match`, `--emit`, and `--check` following `scripts/benchmark_rootfinding.py`. Compare root, residual, width, slope, AD, FD, analytic derivative, status, and certification with `rel_tol=1e-12`, `abs_tol=1e-15`; do not compare timestamps or warm wall time.

- [ ] **Step 4: Emit, check, and test**

```bash
env -u VIRTUAL_ENV uv run --no-sync python \
  scripts/benchmark_implicit_root.py --emit
env -u VIRTUAL_ENV uv run --no-sync python \
  scripts/benchmark_implicit_root.py --check
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_implicit_root_evidence.py \
  tests/validation/test_implicit_root_gradients.py
```

Expected: healthy freshness message and all tests pass.

- [ ] **Step 5: Document the command and commit**

Add the `--check` command beside the scalar-root evidence command in `docs/60-validation/index.md`. Run Ruff/format/MyPy on the script and tests, then:

```bash
git add scripts/benchmark_implicit_root.py \
  docs/validation/implicit-root-gradients.json \
  tests/unit/test_implicit_root_evidence.py docs/60-validation/index.md
git commit -m "test(numerics): regenerate implicit root evidence"
```

---

### Task 3: Clarify IFT terminology and current rootfinding contracts

**Files:**
- Modify: `docs/30-decisions/0008-reject-ift-from-core.md`
- Modify: `docs/10-theory/rootfinding.md`
- Create: `tests/integration/test_rootfinding_docs.py`

**Interfaces:**
- Produces: an unambiguous distinction between Information Field Theory and the implicit function theorem, plus prose matching actual IQI selection order.

- [ ] **Step 1: Add failing terminology and algorithm tests**

```python
def test_adr_distinguishes_information_field_theory_from_implicit_function_theorem():
    text = ADR.read_text(encoding="utf-8")
    assert "Information Field Theory" in text
    assert "implicit function theorem" in text
    assert "does not prohibit" in text


def test_rootfinding_docs_describe_actual_interpolation_order():
    text = ROOTFINDING.read_text(encoding="utf-8")
    assert "inverse-quadratic interpolation when three distinct" in text
    assert "otherwise the endpoint secant" in text
    assert "rejected interpolation uses" in text
```

- [ ] **Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/integration/test_rootfinding_docs.py
```

- [ ] **Step 3: Update ADR and theory prose**

Add a terminology note to ADR 0008:

```markdown
:::{note} IFT terminology
This decision uses **IFT** for Information Field Theory. It does not prohibit
dependency-free use of the **implicit function theorem** in a generic numerical
primitive. The latter is governed by its own smoothness, uniqueness,
conditioning, residual, and validation contracts.
:::
```

Change rootfinding selection prose to state: IQI is attempted when its three-point prerequisites exist; otherwise endpoint secant is attempted; a rejected selected interpolant falls back to midpoint.

- [ ] **Step 4: Run GREEN and commit**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/integration/test_rootfinding_docs.py \
  tests/integration/test_theory_index.py
git add docs/30-decisions/0008-reject-ift-from-core.md \
  docs/10-theory/rootfinding.md tests/integration/test_rootfinding_docs.py
git commit -m "docs: distinguish implicit theorem from field theory"
```

---

### Task 4: Publish “How to learn with Jaxstro”

**Files:**
- Create: `docs/00-getting-started/how-to-learn.md`
- Modify: `docs/00-getting-started/index.md`
- Modify: `docs/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Create: `tests/integration/test_learning_docs.py`

**Interfaces:**
- Produces: the canonical predict-compute-audit explanation and reusable labels for later chapters.

- [ ] **Step 1: Add failing content and navigation tests**

Require the page to contain:

```python
required = (
    "# How to learn with Jaxstro: predict, compute, audit",
    "## Predict",
    "## Compute",
    "## Audit",
    "Prediction prevents post-hoc storytelling",
    "A finite output is not yet a scientific result",
    "The audit starts the next prediction",
    "safeguarded_bracketed_root",
    "powerlaw_cdf",
)
```

Assert `myst.yml`, the homepage, Getting Started index, and route manifest all link the page exactly once.

- [ ] **Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/integration/test_learning_docs.py \
  tests/integration/test_docs_gate_wiring.py
```

- [ ] **Step 3: Write the page**

Use this exact teaching sequence:

```markdown
## Predict

Write down units, signs, limiting cases, invariants, conditioning, expected
failure state, and whether the derivative should exist before executing code.

## Compute

Choose an explicit method and inspect tolerances, branches, fixed shapes,
telemetry, and status rather than retaining only a plausible scalar.

## Audit

Compare with an analytic identity or independent method, test convergence,
inspect provenance, and narrow the claim to what the evidence supports.
```

Include a root example and the finite-power-law `alpha=-1` example. Each gets a three-row Predict/Compute/Audit table and a short “failed audit” loop.

- [ ] **Step 4: Integrate navigation and route manifest**

Add `00-getting-started/how-to-learn.md` immediately after the Getting Started index in `docs/myst.yml`. Add the compiled `/jaxstro/00-getting-started/how-to-learn/` route according to the existing manifest schema; update the expected route count in the wiring test rather than leaving the old count hard-coded.

- [ ] **Step 5: Run GREEN and commit**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/integration/test_learning_docs.py \
  tests/integration/test_docs_gate_wiring.py \
  tests/integration/test_readme_examples.py
git add docs/00-getting-started/how-to-learn.md \
  docs/00-getting-started/index.md docs/index.md docs/myst.yml \
  docs/route-manifest.json tests/integration/test_learning_docs.py \
  tests/integration/test_docs_gate_wiring.py
git commit -m "docs: teach predict compute audit"
```

---

### Task 5: Add the science-enabled patterns page

**Files:**
- Create: `docs/10-theory/science-patterns.md`
- Modify: `docs/10-theory/index.md`
- Modify: `docs/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Create: `tests/integration/test_science_patterns_docs.py`

**Interfaces:**
- Produces: cross-module routes from scientific questions to APIs, AD contracts, evidence, and downstream boundaries.

- [ ] **Step 1: Add failing section and route tests**

Require sections for event/equilibrium location, certified equilibrium sensitivity, accumulated quantities, tabulated models, limiting distributions, coordinates/units/spectra, local interactions, inference parameters, and provenance. For each section assert links to one theory page, `40-api/index.md`, and `60-validation/index.md`.

- [ ] **Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/integration/test_science_patterns_docs.py
```

- [ ] **Step 3: Write each scientific pattern using one template**

```markdown
### Locate an event or equilibrium

**Question.** Where does a signed physical residual cross zero?

**Equation.** $G(x;\theta)=0$.

**Primitive.** `safeguarded_bracketed_root` for auditable values;
`implicit_bracketed_root` only for certified sensitivities.

**Transform boundary.** Scalar `jit`; value-shape `vmap`; physical-cost
`lax.map`; IFT only behind the derivative certificate.

**Failure evidence.** Missing brackets, nonfinite trials, exhaustion, and
rejected derivative certificates are typed results.

**Ownership.** The downstream project owns the physical residual and candidate
admissibility; Jaxstro owns generic scalar numerical evidence.
```

Repeat the complete template for every required pattern; do not use “similar to above.”

- [ ] **Step 4: Integrate and commit**

Update homepage cards, theory index, navigation, route manifest, and route-count gate. Run focused docs tests and commit:

```bash
git add docs/10-theory/science-patterns.md docs/10-theory/index.md \
  docs/index.md docs/myst.yml docs/route-manifest.json \
  tests/integration/test_science_patterns_docs.py \
  tests/integration/test_docs_gate_wiring.py
git commit -m "docs: connect science questions to jaxstro"
```

---

### Task 6: Build the safeguarded-root trace figure

**Files:**
- Create: `laboratory/jaxtroviz/rootfinding.py`
- Modify: `laboratory/jaxtroviz/registry.py`
- Modify: `tests/unit/test_figure_registry.py`
- Modify: `docs/10-theory/rootfinding.md`
- Generate: `docs/10-theory/figures/rootfinding-safeguards.webp`
- Modify: `tests/integration/test_rootfinding_docs.py`

**Interfaces:**
- Produces: `root_trace_results()` and `build_rootfinding_safeguards()` from public rootfinding telemetry.

- [ ] **Step 1: Add failing numerical and registry tests**

```python
def test_root_trace_figure_uses_public_solver_telemetry():
    x, residual, result = root_trace_results()
    assert x.shape == residual.shape == (801,)
    assert result.converged
    assert int(result.n_evaluations) == 42
    executed = np.asarray(result.trace.executed)
    assert np.all(np.asarray(result.trace.lo)[executed] <= np.asarray(result.trace.hi)[executed])


def test_root_trace_figure_is_registered():
    spec = FIGURES["rootfinding-safeguards"]
    assert spec.page == "10-theory/rootfinding.md"
    assert spec.site_path == "docs/10-theory/figures/rootfinding-safeguards.webp"
```

If the fresh public solver's exact evaluation count differs, record it in the required metric table and update the asserted measured value; do not force the solver to match stale evidence.

- [ ] **Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_figure_registry.py -k rootfinding
```

- [ ] **Step 3: Implement results and builder**

Use `f(x)=x**2-2` on `[0,2]`. The left panel plots the residual and executed proposals colored by `PROPOSAL_INVERSE_QUADRATIC`, `PROPOSAL_SECANT`, and `PROPOSAL_MIDPOINT`. The right panel plots `lo`, `hi`, and bracket width versus executed iteration on a log scale. Use the shared JaxtroViz palette and label the exact root, verified sign bracket, fallback, and terminal status.

- [ ] **Step 4: Register, render, and embed accessibly**

Register a deterministic `FigureSpec(seed=0, export=ExportSpec(width=9.4, height=4.3))`. Render with:

```bash
env -u VIRTUAL_ENV uv run --no-sync python -m laboratory.jaxtroviz \
  render rootfinding-safeguards
```

Embed with alt text describing the two panels and a caption that says the figure demonstrates bracket preservation and proposal telemetry, not universal speed.

- [ ] **Step 5: Run figure/docs tests and commit**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_figure_registry.py -k rootfinding \
  tests/integration/test_rootfinding_docs.py
git add laboratory/jaxtroviz/rootfinding.py laboratory/jaxtroviz/registry.py \
  docs/10-theory/figures/rootfinding-safeguards.webp \
  docs/10-theory/rootfinding.md tests/unit/test_figure_registry.py \
  tests/integration/test_rootfinding_docs.py
git commit -m "docs: visualize safeguarded root evidence"
```

---

### Task 7: Build the value-map versus certified-IFT figure and lesson

**Files:**
- Modify: `laboratory/jaxtroviz/rootfinding.py`
- Modify: `laboratory/jaxtroviz/registry.py`
- Modify: `tests/unit/test_figure_registry.py`
- Modify: `docs/10-theory/rootfinding.md`
- Generate: `docs/10-theory/figures/rootfinding-value-versus-ift.webp`
- Modify: `tests/integration/test_rootfinding_docs.py`

**Interfaces:**
- Produces: `implicit_comparison_results()` and `build_rootfinding_value_versus_ift()`.

- [ ] **Step 1: Add failing analytic-evidence tests**

Use `f(x,theta)=x**2-theta` at `theta=2`. Require the figure data to contain the analytic root, certified root, branch-selected value path, analytic derivative, AD derivative, central-FD derivative, and a rejected zero-slope certificate. Assert AD/analytic and AD/FD agreement at the existing validation tolerances.

- [ ] **Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_figure_registry.py -k value_versus_ift
```

- [ ] **Step 3: Implement the two-panel figure**

Panel A shows the executed numerical map and states “value-first: audit the executed branch history.” Panel B shows the local implicit relation, slope, and certificate gates and states “IFT: derivative of the certified root relation.” Include a red rejected flat-root marker and never depict rejected evidence as a derivative arrow.

- [ ] **Step 4: Add the curriculum activity**

In `rootfinding.md`, add:

```markdown
### Predict → compute → audit: which derivative are you asking for?

**Predict.** Decide whether the desired quantity is the sensitivity of the
finite executed algorithm or of a unique smooth mathematical root.

**Compute.** Run the value solver and, separately, the certified implicit API.

**Audit.** Compare AD, analytic, and central-FD derivatives; inspect every
certificate predicate; reject the claim if smoothness, uniqueness, residual,
width, or conditioning evidence fails.
```

- [ ] **Step 5: Render, verify, and commit**

Render through the registry, verify accessible embedding and public-data tests, then commit the builder, registry, WebP, lesson, and tests as one slice.

---

### Task 8: Upgrade the homepage and module chapter conventions

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/10-theory/index.md`
- Modify selectively: `docs/10-theory/*.md`
- Create: `tests/integration/test_curriculum_conventions.py`

**Interfaces:**
- Produces: science-capability homepage copy and consistent, selective curriculum affordances without duplicating API reference content.

- [ ] **Step 1: Inventory chapter affordances in a failing test**

Define the substantial chapters as rootfinding, interpolation, regular-grid, splines, linear algebra, distributions, spatial, ODE, quadrature, quantities, and spectra-linked architecture. Require each to include either explicit learning objectives plus an activity, or a documented exemption in a test-side mapping with a concrete reason. Do not require identical boilerplate on short reference pages.

- [ ] **Step 2: Reframe the homepage**

Add five capability cards: explicit quantities/conventions, auditable numerical maps, events/equilibria/inverse mappings, differentiable tabulated models, and provenance-backed claims. Retain the three doors and all current module/API routes.

- [ ] **Step 3: Add chapter learning objectives and concept checks**

For each selected chapter, add two to four objectives and one short Predict/Compute/Audit or concept-check block based on an existing tested example. Link to current evidence rather than creating new measured claims.

- [ ] **Step 4: Run curriculum and existing module-doc tests**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/integration/test_curriculum_conventions.py \
  tests/integration/test_*docs.py \
  tests/integration/test_theory_index.py
```

If the glob expands to an unrelated optional integration test, replace it with the explicit collected module-doc paths recorded by `rg --files tests/integration | rg '_docs.py$'`.

- [ ] **Step 5: Commit the curriculum pass**

Commit only the homepage, theory pages, and their executable content contracts.

---

### Task 9: Publish the package-wide SOTA assessment

**Files:**
- Create: `docs/90-development-log/sota-assessment.md`
- Modify: `docs/90-development-log/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Create: `tests/integration/test_sota_assessment.py`
- Modify: `STATUS.md`

**Interfaces:**
- Produces: evidence-backed strengths, gaps, and ranked Now/Next/Later investments across the complete package and website.

- [ ] **Step 1: Write failing assessment-coverage tests**

Require the page to cover scientific breadth, ownership, numerical robustness, conditioning, AD honesty, JAX transforms, dimensional safety, API cohesion, serialization, performance/compilation evidence, evidence freshness, provenance, curriculum quality, accessibility, discoverability, and downstream reuse. Require headings `Delivered strengths`, `High-confidence gaps`, `Now`, `Next`, `Later`, and `Evidence required`.

- [ ] **Step 2: Perform the evidence inventory**

Inspect every public module export, theory page, validation anchor, provenance registry, figure registration, and focused test family. Record file/test anchors for every strength and gap. Do not assign a numeric score unless the rubric defines observable thresholds for every level; prefer maturity labels `ratified`, `validated`, `implemented`, `experimental`, and `planned`.

- [ ] **Step 3: Write the ranked roadmap**

The page must identify at most five Now items, seven Next items, and five Later items. Each item contains impact, scientific audience, ownership boundary, implementation risk, and evidence gate. Candidate high-impact ideas must be accepted only if supported by the inventory; likely candidates include:

- generated evidence freshness across all manifests;
- consistent compile/JAXPR evidence for transform-heavy kernels;
- dimensional typing/adoption decisions for `jaxstro.quantity`;
- broader certified implicit primitives only after scalar-root adoption evidence;
- curriculum notebooks or exercises that import, rather than duplicate, tested docs code;
- visual coverage for distribution limits, AD contracts, validation triangles, and ownership boundaries.

- [ ] **Step 4: Integrate navigation and status**

Add the page under Development Log, update route manifest/count, and set `STATUS.md` next actions to the top-ranked uncompleted evidence gate.

- [ ] **Step 5: Verify and commit**

Run the assessment content test, navigation gate, validation-doc test, and theory index test. Commit assessment, navigation, manifest, tests, and status.

---

### Task 10: Full website and package verification

**Files:**
- Modify only if a gate exposes a verified defect.

**Interfaces:**
- Consumes: all preceding committed slices.
- Produces: clean checkout, rendered site, review disposition, and final handoff evidence.

- [ ] **Step 1: Run the bounded code and evidence gate**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_bracketed_root.py \
  tests/unit/test_implicit_root.py \
  tests/unit/test_implicit_root_evidence.py \
  tests/unit/test_figure_registry.py \
  tests/integration/test_learning_docs.py \
  tests/integration/test_science_patterns_docs.py \
  tests/integration/test_rootfinding_docs.py \
  tests/integration/test_curriculum_conventions.py \
  tests/integration/test_sota_assessment.py \
  tests/integration/test_docs_gate_wiring.py \
  tests/integration/test_theory_index.py \
  tests/integration/test_validation_docs.py
env -u VIRTUAL_ENV uv run --no-sync python \
  scripts/benchmark_rootfinding.py --check
env -u VIRTUAL_ENV uv run --no-sync python \
  scripts/benchmark_implicit_root.py --check
```

- [ ] **Step 2: Run static gates**

```bash
env -u VIRTUAL_ENV uv run --no-sync ruff check src tests scripts laboratory
env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests scripts laboratory
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
git diff --check
```

- [ ] **Step 3: Render and audit the website**

Use the repository's existing docs gate command from `scripts/check_docs.sh`, which must contain `myst build --html --ci --strict`. Require the updated route count, zero missing-route failures, zero content warnings/errors, and nonempty image alt text. Record build wall time only in a metric table; do not establish a hardware-independent threshold.

- [ ] **Step 4: Request final reviews**

Use at most one already-created reviewer if available. Review:

- numerical and evidence correctness;
- scientific-claim calibration;
- predict-compute-audit pedagogy;
- accessibility and figure truthfulness;
- module/reference preservation;
- package-wide SOTA priorities.

Address every Critical and Important finding before completion.

- [ ] **Step 5: Final commit and handoff**

If review fixes were required, commit them as `fix: address science docs review`. Confirm `git status --short` is empty. Report:

- final commit;
- exact pages and routes added;
- code-review findings and dispositions;
- test, Ruff, format, MyPy, evidence, and docs-build results;
- figure files and source builders;
- science enabled by the new numerical APIs;
- top Now/Next/Later SOTA recommendations;
- remaining limitations and unvalidated claims.
