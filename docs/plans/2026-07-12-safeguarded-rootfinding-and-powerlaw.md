# Safeguarded Rootfinding and Smooth Power-Law Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:test-driven-development`, `superpowers:systematic-debugging`, and
> `superpowers:verification-before-completion`. Work in the normal checkout.

**Goal:** Add production-grade, value-first safeguarded scalar rootfinding for
Gravax integration, then separately repair finite power-law derivatives through
`alpha = -1` without changing the public distribution signatures.

**Architecture:** Rootfinding uses a minimal true-residual sign bracket, a
safeguarded secant proposal with deterministic midpoint fallback, and a
fixed-length `jax.lax.scan` wrapper with `jax.lax.cond`-masked evaluations.
The power-law repair adapts Progenax's smooth `expm1(x)/x` and `log1p(x)/x`
removable-singularity kernels with exponent `e = alpha + 1`.

**Tech stack:** Python 3.11+, JAX/JAX NumPy, jaxtyping, pytest, Ruff, MyPy, MyST.

## Global constraints

- Run every Python, pytest, Ruff, and MyPy command through `uv run --no-sync`.
- Do not add dependencies, use `while_loop`/`custom_root`, or add implicit-root derivatives.
- Preserve unrelated tracked and untracked work; stage explicit paths only.
- Keep Jaxstro domain-agnostic; no Gravax/P²MSM vocabulary in runtime APIs.
- Report measured numerical results with identity, symbol, value, and units.
- Commit rootfinding and power-law work as separate coherent slices.

---

### Task 1: Low-level bracket contract

**Files:**
- Modify: `tests/unit/test_numerics.py`
- Modify: `src/jaxstro/numerics/rootfinding.py`

**Produces:**
- `BracketState(lo, hi, f_lo, f_hi, bracketed)`
- `BracketProposal(x, kind, safeguarded)`
- `initialize_bracket(lo, hi, f_lo, f_hi)`
- `update_bracket(state, x, fx, *, valid=True)`
- `propose_bracketed(state, *, safeguard_fraction)`

- [ ] Write failing tests for ordered finite sign brackets, both endpoint roots,
  both side updates, exact interior roots, `valid=False`, invalid/nonfinite
  updates, denominator/candidate/progress guards, midpoint fallback, and proposal
  kind tie-breaking.
- [ ] Run the new low-level tests and confirm they fail because symbols are absent.
- [ ] Implement the minimal NamedTuple state and signbit-based bracket helpers.
- [ ] Run the low-level tests, then focused existing rootfinding tests.

### Task 2: Fixed-scan solver and telemetry

**Files:**
- Modify: `tests/unit/test_numerics.py`
- Modify: `tests/validation/test_grad_checks.py`
- Modify: `src/jaxstro/numerics/rootfinding.py`

**Produces:**
- `RootTrace(proposal, residual, lo, hi, f_lo, f_hi, proposal_kind, executed, admissible, converged)`
- `BracketedRootResult(root, residual, converged, bracketed, n_evaluations, trace)`
- `safeguarded_bracketed_root(f, lo, hi, *, max_steps, atol, rtol, safeguard_fraction)`

- [ ] Write failing tests for missing brackets, exact roots, exhaustion, masked
  trace slots, no post-convergence evaluations, analytic linear/quadratic/flat/
  kinked/oscillatory cases, no-root cases, float64, JIT, and VMAP.
- [ ] Confirm the tests fail for missing high-level symbols.
- [ ] Implement endpoint initialization plus a fixed `lax.scan`; wrap proposal
  evaluation in `lax.cond(active, ...)`; count only executed function calls.
- [ ] Converge only for exact residual or root-space width
  `hi - lo <= 2 * (atol + rtol * abs(best_x))`; never turn exhaustion into success.
- [ ] Add a value-first gradient contract test while preserving the existing
  bisection-gradient warning and tests.
- [ ] Run focused unit and validation tests.

### Task 3: Rootfinding public surface, evidence, review, and commit

**Files:**
- Modify: `src/jaxstro/numerics/__init__.py`
- Modify: `docs/10-theory/rootfinding.md`
- Modify: `docs/40-api/index.md`
- Modify: `docs/60-validation/index.md`
- Modify: `README.md` only if its public list becomes stale
- Modify: `STATUS.md`
- Add/modify focused API and docs tests as required by discovered conventions

- [ ] Add failing export/docs assertions, then export the result types, state
  helpers, proposal-kind constants, and high-level solver consistently.
- [ ] Replace the deferred hybrid-solver section with the implemented algorithm,
  deterministic kind table, worked analytic example, failure semantics, and AD boundary.
- [ ] Measure hybrid versus bisection evaluation counts, iterations, residuals,
  relative residuals, and warm scalar wall time on representative analytic cases.
- [ ] Run focused pytest, Ruff, format, MyPy, and documentation checks.
- [ ] Request independent numerical, JAX, AD-honesty, API, and Gravax-integration reviews.
- [ ] Address all Critical and Important findings, rerun gates, update `STATUS.md`,
  and commit the rootfinding slice with explicit-path staging.

### Task 4: Smooth finite power-law singularity

**Files:**
- Modify: `tests/unit/test_distributions.py`
- Modify: `tests/validation/test_grad_checks.py`
- Modify: `src/jaxstro/numerics/distributions.py`

**Preserves:**
- `powerlaw_logpdf(x, *, alpha=-1.0, xmin=1.0, xmax=2.0)`
- `powerlaw_cdf(x, *, alpha=-1.0, xmin=1.0, xmax=2.0)`
- `powerlaw_ppf(u, *, alpha=-1.0, xmin=1.0, xmax=2.0)`

- [ ] Write failing exact-limit tests for analytic alpha derivatives of logpdf,
  CDF, and PPF; central-FD agreement around the limit; round trips;
  normalization; support endpoints; JIT; VMAP; and float64.
- [ ] Confirm failures reproduce the alpha-independent `isclose` branch defect.
- [ ] Adapt Progenax's Taylor-guarded `_phi`, `_psi`, stable power integral,
  and inverse kernels with `e = alpha + 1`; derive normalization/CDF/PPF from them.
- [ ] Run focused distribution and gradient validation tests.

### Task 5: Power-law docs, review, verification, and separate commit

**Files:**
- Modify: `docs/10-theory/distributions.md`
- Modify: `docs/40-api/index.md`
- Modify: `docs/60-validation/index.md`
- Modify: `STATUS.md`

- [ ] Document the smooth singularity formulas, limiting derivatives, and support contract.
- [ ] Record measured AD, finite-difference, round-trip, and normalization values in tables.
- [ ] Run focused pytest, Ruff, format, MyPy, and docs checks.
- [ ] Request an independent correctness/AD review, address Important findings,
  rerun gates, and commit this slice separately with explicit-path staging.

### Task 6: Final handoff

- [ ] Recheck the working tree and both commit hashes.
- [ ] Capture a short factual Brain event without editing the Brain repository.
- [ ] Provide exact imports/signatures, result/trace fields, `valid=False` and
  proposal-kind semantics, a capacity-reset interleaving example, commands and
  outcomes, cost tables, limitations, and confirmation of the prohibited items.

