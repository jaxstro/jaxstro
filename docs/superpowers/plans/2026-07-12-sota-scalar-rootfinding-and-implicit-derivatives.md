# SOTA Scalar Rootfinding and Implicit Derivatives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Work inline in the normal checkout;
> do not create a worktree.

**Goal:** Replace the current safeguarded secant baseline with a reusable,
status-rich Brent-Dekker-style scalar bracket state machine, then add a separate
strictly certified `jax.lax.custom_root` API for implicit derivatives of genuine
smooth roots.

**Architecture:** Phase A keeps `BracketState` as minimal true endpoint evidence
and adds a separate `BracketedRootState` for interpolation history, termination
status, and one-evaluation-per-step Brent-Dekker-style proposals. Phase B exposes
`implicit_bracketed_root(f, args, ...)`: it reuses the Phase-A value solver,
returns its full diagnostics as auxiliary data, applies scalar IFT derivatives
through `jax.lax.custom_root`, and fails closed with NaN value/gradient unless a
runtime certificate and explicit caller assumptions both pass.

**Tech Stack:** Python 3.11+, JAX 0.10.1+, JAX NumPy, jaxtyping, pytest, Ruff,
MyPy, MyST.

## Global Constraints

- Run every Python, pytest, Ruff, and MyPy command through
  `env -u VIRTUAL_ENV uv run --no-sync`.
- Preserve all unrelated tracked and untracked work. Never reset, clean, or
  reformat unrelated files. Stage explicit paths only.
- Keep Jaxstro domain-agnostic. Do not import Gravax or add particle, capacity,
  timestep, controller, P²MSM, SDAR, or target-landing concepts to runtime code.
- Add no core dependency. Optimistix and SciPy may inform validation design but
  must not become Jaxstro runtime dependencies.
- Keep canonical iterative control flow fixed-shape with `jax.lax.scan`. Do not
  add `jax.lax.while_loop`.
- Preserve `BracketState(lo, hi, f_lo, f_hi, bracketed)` as minimal sign-bracket
  evidence and preserve `update_bracket(..., valid=False)` as an exact no-op.
- Make one expensive scalar function evaluation per active forward scan slot.
- Keep masked trace slots deterministic: NaN floating evidence, `PROPOSAL_NONE`,
  false masks, and a terminal status code.
- Treat `vmap` as value/shape compatible but not a physical per-lane cost mask.
  Add an explicit `lax.map` wrapper for callers that require per-lane skipping.
- Keep the value-first and implicit-root APIs separate. The existing
  `safeguarded_bracketed_root` must never silently acquire an IFT derivative.
- A certified implicit root represents the derivative of the mathematical root,
  not automatically the derivative of a finite-budget executed controller.
- Stop Phase B if the certificate cannot fail closed under eager, JIT, and VMAP,
  or if certified IFT derivatives do not agree with analytic truth and central
  finite differences.
- Report every measured numerical result in a table with metric identity,
  symbol, value, and units.
- Commit each task as a coherent slice after fresh verification.

## File Structure

- `src/jaxstro/numerics/rootfinding.py`: public facade plus existing bisection,
  Newton, PPF, and monotone-inverse helpers.
- `src/jaxstro/numerics/_bracketed_root.py`: new private owner of bracket state,
  Brent-Dekker-style proposals, statuses, traces, scalar solve, and `lax.map`
  batch solve.
- `src/jaxstro/numerics/_implicit_root.py`: new private owner of explicit-args
  implicit-root assumptions, runtime certificates, `custom_root`, and fail-closed
  results.
- `tests/unit/test_bracketed_root.py`: focused low-level state, proposal,
  termination, scan, and batch contracts migrated from `test_numerics.py`.
- `tests/validation/test_bracketed_root_algorithms.py`: matched-tolerance
  evaluation-count and invariant validation across analytic/adversarial cases.
- `tests/validation/test_implicit_root_gradients.py`: analytic and central-FD
  truth gates for certified and rejected derivatives.
- `scripts/benchmark_rootfinding.py`: reproducible bisection-versus-canonical
  forward cost evidence.
- `docs/10-theory/rootfinding.md`, `docs/40-api/index.md`, and
  `docs/60-validation/index.md`: public algorithm, status, certificate, and AD
  contracts.

---

## Phase A — SOTA value-first bracketed solver

### Task 1: Add explicit root statuses and serialization-complete results

**Files:**
- Create: `src/jaxstro/numerics/_bracketed_root.py`
- Modify: `src/jaxstro/numerics/rootfinding.py:35-329`
- Modify: `src/jaxstro/numerics/__init__.py`
- Create: `tests/unit/test_bracketed_root.py`
- Modify: `tests/unit/test_numerics.py:390-733`
- Modify: `tests/integration/test_api_reference.py`

**Interfaces:**
- Consumes: existing `BracketState`, `BracketProposal`, `RootTrace`,
  `BracketedRootResult`, `initialize_bracket`, `update_bracket`,
  `propose_bracketed`, and `safeguarded_bracketed_root` contracts.
- Produces: public status constants, a serialization-complete result, and the
  private implementation split described below.
- Preserves: the exact fieldwise no-op of
  `update_bracket(state, x, fx, valid=False)`.

- [ ] **Step 1: Write failing status and result-field tests**

  Move the safeguarded classes from `tests/unit/test_numerics.py` into the new
  focused file without changing assertions. Add the following tests before
  changing production code:

  ```python
  def test_result_status_distinguishes_terminal_outcomes() -> None:
      exact = _solve(lambda x: x - 1.0, 0.0, 2.0)
      missing = _solve(lambda x: x * x + 1.0, -1.0, 1.0)
      exhausted = _solve(
          lambda x: x * x - 2.0,
          0.0,
          2.0,
          max_steps=1,
          atol=0.0,
          rtol=0.0,
      )
      nonfinite = _solve(
          lambda x: jnp.where(x == 1.0, jnp.nan, x - 0.25),
          0.0,
          2.0,
          max_steps=8,
          safeguard_fraction=0.49,
      )

      assert exact.status == rootfinding.ROOT_STATUS_EXACT_INTERIOR
      assert missing.status == rootfinding.ROOT_STATUS_MISSING_BRACKET
      assert exhausted.status == rootfinding.ROOT_STATUS_MAX_STEPS
      assert nonfinite.status == rootfinding.ROOT_STATUS_NONFINITE_EVALUATION
  ```

  Add field-order assertions:

  ```python
  def test_result_and_trace_fields_are_checkpoint_stable() -> None:
      assert rootfinding.RootTrace._fields == (
          "proposal",
          "residual",
          "lo",
          "hi",
          "f_lo",
          "f_hi",
          "proposal_kind",
          "executed",
          "admissible",
          "converged",
          "status",
      )
      assert rootfinding.BracketedRootResult._fields == (
          "root",
          "residual",
          "status",
          "converged",
          "bracketed",
          "n_evaluations",
          "residual_scale",
          "final_bracket",
          "trace",
      )
  ```

- [ ] **Step 2: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_bracketed_root.py \
    tests/integration/test_api_reference.py -k "root or bracket"
  ```

  Expected: failures for missing `status`, `residual_scale`, `final_bracket`,
  trace `status`, and public status constants. Confirm that the migrated existing
  tests still collect.

- [ ] **Step 3: Implement exact public status constants and result fields**

  In `_bracketed_root.py`, define stable `int32` identifiers:

  ```python
  ROOT_STATUS_RUNNING = 0
  ROOT_STATUS_EXACT_LO = 1
  ROOT_STATUS_EXACT_HI = 2
  ROOT_STATUS_EXACT_INTERIOR = 3
  ROOT_STATUS_WIDTH_CONVERGED = 4
  ROOT_STATUS_MISSING_BRACKET = 5
  ROOT_STATUS_NONFINITE_EVALUATION = 6
  ROOT_STATUS_MAX_STEPS = 7
  ```

  Define the hard-cutover public types:

  ```python
  class RootTrace(NamedTuple):
      proposal: Float[Array, " steps"]
      residual: Float[Array, " steps"]
      lo: Float[Array, " steps"]
      hi: Float[Array, " steps"]
      f_lo: Float[Array, " steps"]
      f_hi: Float[Array, " steps"]
      proposal_kind: Array
      executed: Array
      admissible: Array
      converged: Array
      status: Array


  class BracketedRootResult(NamedTuple):
      root: Float[Array, ""]
      residual: Float[Array, ""]
      status: Array
      converged: Array
      bracketed: Array
      n_evaluations: Array
      residual_scale: Float[Array, ""]
      final_bracket: BracketState
      trace: RootTrace
  ```

  `residual_scale` is
  `max(abs(initial_f_lo), abs(initial_f_hi))`. `final_bracket` is the last valid
  true-residual bracket, including on exhaustion. Use lower-endpoint tie-breaking
  consistently.

- [ ] **Step 4: Move bracketed implementation ownership without aliases**

  Move the existing bracketed types/functions from `rootfinding.py` to
  `_bracketed_root.py`. Import and re-export the public names from
  `rootfinding.py` and `jaxstro.numerics`. Do not retain duplicate definitions or
  compatibility aliases. Keep existing `bisect`, Newton, PPF, and inverse-table
  code in `rootfinding.py`.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_bracketed_root.py \
    tests/unit/test_numerics.py \
    tests/integration/test_api_reference.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/numerics/_bracketed_root.py \
    src/jaxstro/numerics/rootfinding.py \
    src/jaxstro/numerics/__init__.py \
    tests/unit/test_bracketed_root.py tests/unit/test_numerics.py \
    tests/integration/test_api_reference.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/numerics/_bracketed_root.py \
    src/jaxstro/numerics/rootfinding.py \
    src/jaxstro/numerics/__init__.py \
    tests/unit/test_bracketed_root.py tests/unit/test_numerics.py \
    tests/integration/test_api_reference.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  git diff --check
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/numerics/_bracketed_root.py \
    src/jaxstro/numerics/rootfinding.py src/jaxstro/numerics/__init__.py \
    tests/unit/test_bracketed_root.py tests/unit/test_numerics.py \
    tests/integration/test_api_reference.py
  git commit -m "refactor: add bracketed root status contract"
  ```

### Task 2: Add one-evaluation Brent-Dekker-style low-level state

**Files:**
- Modify: `src/jaxstro/numerics/_bracketed_root.py`
- Modify: `src/jaxstro/numerics/rootfinding.py`
- Modify: `src/jaxstro/numerics/__init__.py`
- Modify: `tests/unit/test_bracketed_root.py`
- Create: `tests/validation/test_bracketed_root_algorithms.py`

**Interfaces:**
- Consumes: `BracketState` true endpoint evidence and Task-1 status constants.
- Produces:
  `BracketHistory`, `BracketedRootState`,
  `initialize_bracketed_root_state`, `propose_bracketed`, and
  `advance_bracketed_root`.
- Preserves: one expensive evaluation between each proposal and advance call;
  `valid=False` preserves bracket, interpolation history, and status exactly.

- [ ] **Step 1: Write failing state/history tests**

  Add:

  ```python
  def test_brent_state_separates_true_bracket_from_interpolation_history() -> None:
      bracket = rootfinding.initialize_bracket(0.0, 2.0, -2.0, 2.0)
      state = rootfinding.initialize_bracketed_root_state(bracket)

      assert state.bracket == bracket
      assert jnp.isnan(state.history.previous_x)
      assert jnp.isnan(state.history.previous_fx)
      assert not bool(state.history.initialized)
      assert state.status == rootfinding.ROOT_STATUS_RUNNING


  def test_invalid_advance_is_exact_full_state_noop() -> None:
      bracket = rootfinding.initialize_bracket(0.0, 2.0, -2.0, 2.0)
      state = rootfinding.initialize_bracketed_root_state(bracket)
      proposal = rootfinding.propose_bracketed(
          state, safeguard_fraction=0.1
      )
      updated = rootfinding.advance_bracketed_root(
          state, proposal, jnp.asarray(0.0), valid=False
      )
      comparisons = jax.tree.map(jnp.array_equal, state, updated)
      assert all(bool(value) for value in jax.tree.leaves(comparisons))
  ```

  Define expected checkpoint fields:

  ```python
  class BracketHistory(NamedTuple):
      previous_x: Float[Array, ""]
      previous_fx: Float[Array, ""]
      previous_previous_x: Float[Array, ""]
      previous_step_was_midpoint: Array
      initialized: Array


  class BracketedRootState(NamedTuple):
      bracket: BracketState
      history: BracketHistory
      status: Array
  ```

- [ ] **Step 2: Write failing proposal-selection tests**

  Add `PROPOSAL_INVERSE_QUADRATIC = 5` and tests covering:

  ```python
  def test_three_distinct_finite_points_enable_inverse_quadratic_proposal() -> None:
      state = _state_with_history(
          lo=0.0,
          hi=1.5,
          f_lo=-2.0,
          f_hi=0.25,
          previous_x=2.0,
          previous_fx=2.0,
          previous_previous_x=1.0,
          previous_step_was_midpoint=False,
      )
      proposal = rootfinding.propose_bracketed(state, safeguard_fraction=0.01)
      assert proposal.kind == rootfinding.PROPOSAL_INVERSE_QUADRATIC
      assert state.bracket.lo < proposal.x < state.bracket.hi


  @pytest.mark.parametrize(
      "reason",
      ["duplicate-point", "nonfinite-iqi", "outside-bracket", "insufficient-progress"],
  )
  def test_rejected_interpolation_uses_deterministic_midpoint(reason) -> None:
      state = _rejection_fixture(reason)
      proposal = rootfinding.propose_bracketed(state, safeguard_fraction=0.1)
      assert proposal.kind == rootfinding.PROPOSAL_MIDPOINT
      assert proposal.x == pytest.approx(
          0.5 * state.bracket.lo + 0.5 * state.bracket.hi
      )
      assert bool(proposal.safeguarded)
  ```

  `_state_with_history` and `_rejection_fixture` are test-only constructors that
  build the exact public NamedTuple fields. The four fixtures must independently
  falsify one guard while keeping other inputs finite.

- [ ] **Step 3: Run the low-level RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_bracketed_root.py -k "state or interpolation or advance"
  ```

  Expected: failures for missing state/history types, initializer, advance
  function, and IQI proposal kind.

- [ ] **Step 4: Implement deterministic Brent-Dekker-style proposal guards**

  Implement `propose_bracketed(state, *, safeguard_fraction)` with this exact
  decision order:

  1. Return exact lower endpoint before exact upper endpoint.
  2. Return `PROPOSAL_NONE` for a missing bracket or terminal state.
  3. Attempt inverse quadratic interpolation only when `lo`, `hi`, and
     `history.previous_x` are pairwise distinct and their residuals are finite
     and pairwise distinct.
  4. Otherwise attempt the finite secant interpolant from the true endpoints.
  5. Accept interpolation only if it is strictly inside the bracket, inside the
     safeguard band
     `[lo + safeguard_fraction*width, hi - safeguard_fraction*width]`, and its
     displacement from the smaller-residual endpoint is less than half the
     previous accepted displacement.
  6. If history is not initialized, use only the in-bracket and safeguard-band
     tests.
  7. Reject every nonfinite, repeated, out-of-band, or insufficient-progress
     interpolant to the overflow-safe midpoint
     `0.5*lo + 0.5*hi`.

  Sanitize every denominator before division so rejected branches remain finite
  under reverse-mode tracing. Compute sign changes with exact zeros plus
  `jnp.signbit`; never multiply endpoint residuals.

- [ ] **Step 5: Implement one-evaluation state advance**

  `advance_bracketed_root(state, proposal, fx, *, valid=True)` must:

  ```python
  admissible = (
      valid
      & state.bracket.bracketed
      & jnp.isfinite(proposal.x)
      & jnp.isfinite(fx)
      & (proposal.x >= state.bracket.lo)
      & (proposal.x <= state.bracket.hi)
  )
  ```

  When inadmissible because `valid=False`, return every field unchanged. When
  `fx` is nonfinite and `valid=True`, preserve bracket/history and set
  `ROOT_STATUS_NONFINITE_EVALUATION`. For an admissible exact root, collapse the
  true bracket and set `ROOT_STATUS_EXACT_INTERIOR`. Otherwise update exactly one
  true endpoint, shift `previous_x` to `previous_previous_x`, store the evaluated
  proposal/residual, and store whether the proposal kind was midpoint.

- [ ] **Step 6: Validate bracket invariants and one-evaluation semantics**

  In `tests/validation/test_bracketed_root_algorithms.py`, add continuous linear,
  quadratic, flat-slope, monotone-kink, oscillatory fixed-point residual, and
  nonmonotone positive-scale cases. For every executed slot assert:

  ```python
  finite = result.trace.executed & result.trace.admissible
  assert jnp.all(result.trace.lo[finite] <= result.trace.hi[finite])
  assert jnp.all(
      (result.trace.f_lo[finite] == 0.0)
      | (result.trace.f_hi[finite] == 0.0)
      | (jnp.signbit(result.trace.f_lo[finite])
         != jnp.signbit(result.trace.f_hi[finite]))
  )
  assert int(result.n_evaluations) == 2 + int(jnp.sum(result.trace.executed))
  ```

- [ ] **Step 7: Run GREEN and commit**

  Run the Task-1 gate plus:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_bracketed_root.py \
    tests/validation/test_bracketed_root_algorithms.py
  ```

  Expected: all tests pass with one function evaluation per active step. Run
  focused Ruff, format, MyPy, and `git diff --check`, then commit:

  ```bash
  git add src/jaxstro/numerics/_bracketed_root.py \
    src/jaxstro/numerics/rootfinding.py src/jaxstro/numerics/__init__.py \
    tests/unit/test_bracketed_root.py \
    tests/validation/test_bracketed_root_algorithms.py
  git commit -m "feat: add Brent-Dekker bracket state machine"
  ```

### Task 3: Cut the high-level solver over and add physical-cost batch mapping

**Files:**
- Modify: `src/jaxstro/numerics/_bracketed_root.py`
- Modify: `src/jaxstro/numerics/rootfinding.py`
- Modify: `src/jaxstro/numerics/__init__.py`
- Modify: `tests/unit/test_bracketed_root.py`
- Modify: `tests/validation/test_bracketed_root_algorithms.py`
- Modify: `tests/integration/test_api_reference.py`

**Interfaces:**
- Produces the canonical scalar signature:

  ```python
  safeguarded_bracketed_root(
      f,
      lo,
      hi,
      *,
      max_steps,
      atol=0.0,
      rtol=1.0e-8,
      safeguard_fraction=0.1,
  ) -> BracketedRootResult
  ```

- Produces the explicit-args batch signature:

  ```python
  map_safeguarded_bracketed_root(
      f,
      args,
      lo,
      hi,
      *,
      max_steps,
      atol=0.0,
      rtol=1.0e-8,
      safeguard_fraction=0.1,
      batch_size=None,
  ) -> BracketedRootResult
  ```

  Here `f(x, arg)` is scalar in `x`; `args`, `lo`, and `hi` share a leading
  batch dimension. The wrapper uses `jax.lax.map`, not `vmap`.

- [ ] **Step 1: Write failing scalar status/trace tests**

  Extend existing tests to require exact endpoint statuses without scan
  evaluations, per-slot terminal status, width convergence only when the returned
  evaluated endpoint is inside `final_bracket`, nonfinite single-evaluation
  exhaustion, and `ROOT_STATUS_MAX_STEPS` without accidental acceptance.

- [ ] **Step 2: Write failing `lax.map` batch tests**

  Add:

  ```python
  def test_lax_map_batch_matches_scalar_solves_and_preserves_shapes() -> None:
      targets = jnp.array([1.0, 2.0, 9.0], dtype=jnp.float64)
      lo = jnp.zeros_like(targets)
      hi = jnp.full_like(targets, 4.0)
      mapped = rootfinding.map_safeguarded_bracketed_root(
          lambda x, target: x * x - target,
          targets,
          lo,
          hi,
          max_steps=64,
          atol=1.0e-12,
          rtol=1.0e-12,
          safeguard_fraction=0.1,
      )
      scalar = jax.tree.map(
          lambda *xs: jnp.stack(xs),
          *[
              rootfinding.safeguarded_bracketed_root(
                  lambda x, target=target: x * x - target,
                  0.0,
                  4.0,
                  max_steps=64,
                  atol=1.0e-12,
                  rtol=1.0e-12,
                  safeguard_fraction=0.1,
              )
              for target in targets
          ],
      )
      comparisons = jax.tree.map(
          lambda x, y: jnp.array_equal(x, y, equal_nan=True), mapped, scalar
      )
      assert all(bool(value) for value in jax.tree.leaves(comparisons))
      assert mapped.trace.proposal.shape == (3, 64)
  ```

  Add a `jax.debug.callback` or effect-token counter test under `jax.disable_jit`
  showing that a lane converging in one step is evaluated once even while another
  lane continues. Do not use a Python counter under `vmap` as evidence.

- [ ] **Step 3: Implement the fixed-scan high-level cutover**

  Initialize `BracketedRootState`, use the Task-2 proposal/advance functions in a
  fixed `lax.scan`, and guard the single residual evaluation with scalar
  `lax.cond(active, evaluate, masked, proposal.x)`. Derive result `status` from
  exact endpoint initialization, exact interior advance, width convergence,
  nonfinite advance, missing bracket, or final max-step exhaustion. Do not infer
  success from the last finite proposal. An executed slot records the status
  produced by that advance; every later masked slot carries the same terminal
  status while keeping `executed=False`, `admissible=False`, and
  `converged=False`.

- [ ] **Step 4: Implement the `lax.map` wrapper**

  Flatten the leading batch dimension, map a scalar closure
  `lambda item: safeguarded_bracketed_root(lambda x: f(x, item.args), ...)`, then
  reshape every result leaf to batch-major shape. Reject nonmatching concrete
  leading dimensions eagerly. Forward `batch_size` to `jax.lax.map` only when it
  is not `None`.

- [ ] **Step 5: Run scalar, JIT, VMAP-value, and LAX-MAP-cost gates**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_bracketed_root.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/integration/test_api_reference.py
  ```

  Expected: scalar/JIT/VMAP values and shapes agree; only `lax.map` carries the
  physical per-lane cost claim.

- [ ] **Step 6: Commit**

  After focused Ruff, format, MyPy, and diff checks, commit:

  ```bash
  git add src/jaxstro/numerics/_bracketed_root.py \
    src/jaxstro/numerics/rootfinding.py src/jaxstro/numerics/__init__.py \
    tests/unit/test_bracketed_root.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/integration/test_api_reference.py
  git commit -m "feat: cut over safeguarded scalar root solver"
  ```

### Task 4: Ratify forward evaluation efficiency and publish the value API

**Files:**
- Modify: `scripts/benchmark_rootfinding.py`
- Modify: `docs/validation/rootfinding-performance.json`
- Modify: `tests/unit/test_benchmark_rootfinding_script.py`
- Modify: `docs/10-theory/rootfinding.md`
- Modify: `docs/40-api/index.md`
- Modify: `docs/60-validation/index.md`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: the canonical Task-3 value solver and status/result contract.
- Produces: reproducible matched-tolerance bisection-versus-hybrid evidence and
  documented cross-project import/signature contracts.

- [ ] **Step 1: Update the benchmark schema test before the benchmark**

  Require each case/method to record `function_evaluations`,
  `executed_iterations`, `final_absolute_residual`, `final_relative_residual`,
  `warm_wall`, `status`, and `converged`, each with explicit units where
  applicable. Require environment, git revision, dirty flag, precision, and
  matched coordinate tolerance.

- [ ] **Step 2: Measure matched-tolerance cases**

  Retain linear, quadratic, flat-slope, monotone-kink, and oscillatory
  fixed-point residual cases. Choose bisection steps separately per case so its
  final full bracket width is no larger than the hybrid coordinate limit. Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/benchmark_rootfinding.py --emit
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/benchmark_rootfinding.py --check
  ```

  Forward cutover passes only if:

  - every method returns a certified bracket-contained root at the matched
    coordinate tolerance;
  - the hybrid uses no more evaluations than bisection on all five cases;
  - the hybrid uses at least 25% fewer evaluations on at least three cases;
  - no trace contains a lost bracket, nonfinite admissible residual, or status
    contradiction;
  - warm wall is recorded but carries no hardware-independent pass threshold.

  If any criterion fails, stop before Phase B. Keep the Task-1 status contract,
  preserve the last green commit, and revise the interpolation acceptance rules
  rather than weakening the benchmark.

- [ ] **Step 3: Update docs and executable API assertions**

  Document all status and proposal identifiers, `BracketState` versus
  `BracketedRootState` ownership, one-evaluation semantics, `valid=False`, scalar
  versus `vmap` versus `lax.map` cost claims, result/trace fields, and the
  measured table. Replace current secant-only prose with the ratified
  Brent-Dekker-style algorithm.

- [ ] **Step 4: Request the Phase-A review gate**

  Request independent review of numerical invariants, JAX control flow,
  evaluation accounting, public API cleanliness, serialization fields, and
  downstream low-level integration suitability. Address every Critical and
  Important finding before proceeding.

- [ ] **Step 5: Run the bounded Phase-A closeout and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_bracketed_root.py \
    tests/unit/test_numerics.py \
    tests/unit/test_benchmark_rootfinding_script.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/validation/test_grad_checks.py::TestRootfindingGradChecks \
    tests/integration/test_api_reference.py \
    tests/integration/test_validation_docs.py \
    tests/integration/test_theory_index.py \
    tests/integration/test_readme_examples.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/numerics tests/unit/test_bracketed_root.py \
    tests/unit/test_numerics.py tests/unit/test_benchmark_rootfinding_script.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/validation/test_grad_checks.py tests/integration/test_api_reference.py \
    scripts/benchmark_rootfinding.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/numerics tests/unit/test_bracketed_root.py \
    tests/unit/test_numerics.py tests/unit/test_benchmark_rootfinding_script.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/validation/test_grad_checks.py tests/integration/test_api_reference.py \
    scripts/benchmark_rootfinding.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  ```

  Run `myst build` from `docs/` and require 61 pages with no content warnings or
  errors. Update `STATUS.md` with the exact metrics and review disposition. Commit:

  ```bash
  git add src/jaxstro/numerics tests/unit/test_bracketed_root.py \
    tests/unit/test_numerics.py tests/unit/test_benchmark_rootfinding_script.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/validation/test_grad_checks.py tests/integration/test_api_reference.py \
    scripts/benchmark_rootfinding.py docs/10-theory/rootfinding.md \
    docs/40-api/index.md docs/60-validation/index.md \
    docs/validation/rootfinding-performance.json README.md CLAUDE.md STATUS.md
  git commit -m "docs: ratify SOTA bracketed rootfinding"
  ```

---

## Phase B — Strictly gated implicit-root derivative API

Phase B starts only after every Phase-A criterion and review gate passes. It does
not alter `safeguarded_bracketed_root` differentiation semantics.

### Task 5: Add explicit assumptions and runtime derivative certificates

**Files:**
- Create: `src/jaxstro/numerics/_implicit_root.py`
- Modify: `src/jaxstro/numerics/rootfinding.py`
- Modify: `src/jaxstro/numerics/__init__.py`
- Create: `tests/unit/test_implicit_root.py`
- Modify: `tests/integration/test_api_reference.py`

**Interfaces:**
- Produces:

  ```python
  class ImplicitRootAssumptions(NamedTuple):
      unique_root: Array
      smooth_branch: Array


  class ImplicitRootCertificate(NamedTuple):
      finite: Array
      primal_converged: Array
      unique_root_asserted: Array
      smooth_branch_asserted: Array
      residual_ok: Array
      slope_ok: Array
      width_ok: Array
      certified: Array
      residual_limit: Float[Array, ""]
      width_limit: Float[Array, ""]
      slope_floor: Float[Array, ""]


  class ImplicitRootResult(NamedTuple):
      root: Float[Array, ""]
      residual: Float[Array, ""]
      slope: Float[Array, ""]
      status: Array
      certified: Array
      certificate: ImplicitRootCertificate
      primal: BracketedRootResult
  ```

  Stable derivative status identifiers:

  ```python
  DERIVATIVE_STATUS_CERTIFIED = 0
  DERIVATIVE_STATUS_PRIMAL_FAILED = 1
  DERIVATIVE_STATUS_ASSUMPTIONS_REJECTED = 2
  DERIVATIVE_STATUS_NONFINITE = 3
  DERIVATIVE_STATUS_RESIDUAL_TOO_LARGE = 4
  DERIVATIVE_STATUS_SLOPE_ILL_CONDITIONED = 5
  DERIVATIVE_STATUS_BRACKET_TOO_WIDE = 6
  ```

- [ ] **Step 1: Write failing certificate truth-table tests**

  Test every failed predicate independently. The all-pass fixture must satisfy:

  ```python
  residual_limit = residual_atol + residual_rtol * primal.residual_scale
  width_limit = width_atol + width_rtol * jnp.abs(primal.root)
  finite = (
      jnp.isfinite(primal.root)
      & jnp.isfinite(primal.residual)
      & jnp.isfinite(slope)
  )
  certified = (
      finite
      & primal.converged
      & assumptions.unique_root
      & assumptions.smooth_branch
      & (jnp.abs(primal.residual) <= residual_limit)
      & (jnp.abs(slope) >= slope_floor)
      & (primal.final_bracket.hi - primal.final_bracket.lo <= width_limit)
  )
  ```

  Assert deterministic status precedence in exactly this order: primal failure,
  rejected assumptions, nonfinite evidence, residual failure, slope failure,
  width failure, certified.

- [ ] **Step 2: Write failing public export and fail-closed tests**

  Add API assertions for all assumption/certificate/result fields and derivative
  status constants. Require an uncertified helper result to expose the original
  `primal` diagnostics while its derivative-facing `root` is NaN.

- [ ] **Step 3: Run RED, implement certificate construction, and run GREEN**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_implicit_root.py \
    tests/integration/test_api_reference.py -k "implicit or root"
  ```

  Expected RED: missing module/types/constants. Implement a private
  `_build_implicit_certificate(...)` using only JAX array operations and the
  exact truth table above. Do not call `custom_root` in this task. Rerun to green,
  run focused Ruff/format/MyPy, and commit:

  ```bash
  git add src/jaxstro/numerics/_implicit_root.py \
    src/jaxstro/numerics/rootfinding.py src/jaxstro/numerics/__init__.py \
    tests/unit/test_implicit_root.py tests/integration/test_api_reference.py
  git commit -m "feat: add implicit root certificate contract"
  ```

### Task 6: Implement `jax.lax.custom_root` behind the certificate

**Files:**
- Modify: `src/jaxstro/numerics/_implicit_root.py`
- Modify: `src/jaxstro/numerics/rootfinding.py`
- Modify: `src/jaxstro/numerics/__init__.py`
- Modify: `tests/unit/test_implicit_root.py`
- Create: `tests/validation/test_implicit_root_gradients.py`

**Interfaces:**
- Consumes: `f(x, args) -> scalar`, explicit differentiable `args`, Phase-A
  scalar solve, and Task-5 certificate types.
- Produces:

  ```python
  implicit_bracketed_root(
      f,
      args,
      lo,
      hi,
      *,
      assumptions,
      max_steps,
      atol,
      rtol,
      safeguard_fraction,
      derivative_residual_atol,
      derivative_residual_rtol=0.0,
      derivative_width_atol,
      derivative_width_rtol=0.0,
      derivative_slope_floor,
  ) -> ImplicitRootResult
  ```

- [ ] **Step 1: Write failing certified analytic-gradient tests**

  Add float64 tests for:

  ```python
  def test_linear_implicit_gradient_matches_analytic_and_fd() -> None:
      def residual(x, theta):
          return x - theta

      def solve(theta):
          return implicit_bracketed_root(
              residual,
              theta,
              0.0,
              4.0,
              assumptions=ImplicitRootAssumptions(True, True),
              max_steps=64,
              atol=1.0e-14,
              rtol=1.0e-14,
              safeguard_fraction=0.1,
              derivative_residual_atol=1.0e-13,
              derivative_width_atol=1.0e-13,
              derivative_slope_floor=1.0e-8,
          ).root

      theta = jnp.asarray(2.0)
      ad = jax.grad(solve)(theta)
      fd = (solve(theta + 1.0e-5) - solve(theta - 1.0e-5)) / 2.0e-5
      assert ad == pytest.approx(1.0, rel=1.0e-10)
      assert ad == pytest.approx(float(fd), rel=1.0e-8)
  ```

  Add positive-root quadratic
  `f(x, theta)=x**2-theta` with analytic derivative
  `1/(2*sqrt(theta))`, and exponential
  `f(x, theta)=exp(x)-theta` with analytic derivative `1/theta`.

- [ ] **Step 2: Write failing rejection-gradient tests**

  Add independent cases for:

  - `unique_root=False`;
  - `smooth_branch=False` on `abs(x)-theta`;
  - zero slope at `(x-theta)**3`;
  - deliberately loose primal residual tolerance;
  - deliberately wide final bracket;
  - nonfinite residual or slope.

  Each case must assert the exact derivative status, `certified=False`, finite
  nested primal diagnostics where applicable, NaN derivative-facing `root`, and
  NaN `jax.grad(lambda theta: result.root)(theta)`.

- [ ] **Step 3: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_implicit_root.py \
    tests/validation/test_implicit_root_gradients.py
  ```

  Expected: failures because `implicit_bracketed_root` is absent.

- [ ] **Step 4: Implement `custom_root` with full primal auxiliary data**

  Implement:

  ```python
  def root_function(x):
      return f(x, args)

  def solve(fn, _initial_guess):
      primal = safeguarded_bracketed_root(
          fn,
          lo,
          hi,
          max_steps=max_steps,
          atol=atol,
          rtol=rtol,
          safeguard_fraction=safeguard_fraction,
      )
      return primal.root, primal

  def tangent_solve(g, y):
      slope = g(jnp.ones_like(y))
      safe = jnp.where(jnp.abs(slope) >= derivative_slope_floor, slope, 1.0)
      return y / safe

  implicit_root, primal = jax.lax.custom_root(
      root_function,
      0.5 * jnp.asarray(lo) + 0.5 * jnp.asarray(hi),
      solve,
      tangent_solve,
      has_aux=True,
  )
  ```

  Evaluate the signed residual and scalar slope at `implicit_root`, build the
  Task-5 certificate, and choose status by the specified precedence. Return the
  certified root unchanged. For failure, use a `lax.cond` false branch returning
  `implicit_root * jnp.nan`; this must make both the value and attempted gradient
  NaN instead of returning a plausible zero gradient. Stop gradients through
  certificate/status diagnostics.

- [ ] **Step 5: Run GREEN, inspect JAXPR, and commit**

  Run the RED command again and require all tests to pass. Add a test that
  source inspection confirms that the public implementation calls
  `jax.lax.custom_root`; `jax.make_jaxpr` of a certified solve and its gradient
  contain no `while` primitive. Run JIT and VMAP value/gradient tests over
  independent scalar parameters. Then run focused Ruff/format/MyPy and commit:

  ```bash
  git add src/jaxstro/numerics/_implicit_root.py \
    src/jaxstro/numerics/rootfinding.py src/jaxstro/numerics/__init__.py \
    tests/unit/test_implicit_root.py \
    tests/validation/test_implicit_root_gradients.py
  git commit -m "feat: add certified implicit scalar roots"
  ```

### Task 7: Adversarially validate the derivative claim

**Files:**
- Modify: `tests/validation/test_implicit_root_gradients.py`
- Modify: `tests/validation/test_grad_checks.py`
- Create: `docs/validation/implicit-root-gradients.json`
- Create: `tests/unit/test_implicit_root_evidence.py`

**Interfaces:**
- Consumes: certified `implicit_bracketed_root` from Task 6.
- Produces: reproducible analytic/AD/central-FD evidence with explicit units and
  fail-closed adversarial cases.

- [ ] **Step 1: Add tolerance-tightening and branch-stability tests**

  For linear, quadratic, and exponential cases, solve at coordinate tolerances
  `1e-8`, `1e-10`, `1e-12`, and `1e-14`. Record root error, residual, bracket
  width, slope magnitude, AD derivative, central-FD derivative, and relative
  AD/FD discrepancy. Require certified claims only when:

  ```python
  relative_gradient_error = abs(ad - fd) / max(abs(fd), 1.0e-14)
  assert relative_gradient_error <= 1.0e-6
  assert abs(result.residual) <= result.certificate.residual_limit
  assert (
      result.primal.final_bracket.hi - result.primal.final_bracket.lo
      <= result.certificate.width_limit
  )
  ```

  Require derivative convergence under tolerance tightening: the last two
  certified gradients must differ by no more than `1e-8` relative.

- [ ] **Step 2: Add adversarial rejection cases**

  Add multiple-root `x**3-x-theta` with `unique_root=False`, nonsmooth
  `abs(x)-theta` with `smooth_branch=False`, near-singular
  `epsilon*(x-theta)` below the slope floor, nonfinite function values, and a
  branch-changing piecewise residual. These cases must not produce a certified
  derivative.

- [ ] **Step 3: Emit and schema-test the evidence artifact**

  Store one row per case/tolerance in
  `docs/validation/implicit-root-gradients.json`. Each measured quantity must be
  represented as `{"value": ..., "unit": ...}`. Include JAX version, device,
  precision, git revision, dirty flag, FD step, and certificate thresholds. The
  unit test must recompute the deterministic root/status/derivative fields and
  reject stale evidence; warm timings may be recorded without an equality gate.

- [ ] **Step 4: Run the derivative scientific gate and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_implicit_root.py \
    tests/unit/test_implicit_root_evidence.py \
    tests/validation/test_implicit_root_gradients.py \
    tests/validation/test_grad_checks.py
  ```

  Expected: all certified cases meet analytic and FD criteria; every adversarial
  case fails closed with the intended status. Commit:

  ```bash
  git add tests/unit/test_implicit_root.py \
    tests/unit/test_implicit_root_evidence.py \
    tests/validation/test_implicit_root_gradients.py \
    tests/validation/test_grad_checks.py \
    docs/validation/implicit-root-gradients.json
  git commit -m "test: validate implicit scalar root derivatives"
  ```

### Task 8: Publish the implicit API and complete the cross-project handoff

**Files:**
- Modify: `docs/10-theory/rootfinding.md`
- Modify: `docs/40-api/index.md`
- Modify: `docs/60-validation/index.md`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `tests/integration/test_api_reference.py`
- Modify: `tests/integration/test_validation_docs.py`
- Modify: `STATUS.md`

**Interfaces:**
- Publishes the exact Task-5/6 signatures, fields, statuses, certificate
  predicates, and derivative meaning.
- Preserves the value-first no-IFT contract of `safeguarded_bracketed_root`.

- [ ] **Step 1: Write failing documentation/API assertions**

  Require the API page to contain the exact implicit signature, every result and
  certificate field, every derivative status, the phrase “derivative of the
  mathematical root,” and the warning that this is not automatically the
  derivative of an executed finite-budget controller.

- [ ] **Step 2: Document both solver lanes without overclaiming**

  Add a comparison table:

  | Need | API | Derivative contract |
  | --- | --- | --- |
  | Auditable forward root or finite-budget integration | `safeguarded_bracketed_root` | Value-first; no implicit derivative claim |
  | Independent batched forward roots with physical lane masking | `map_safeguarded_bracketed_root` | Value-first; `lax.map` cost semantics |
  | Unique smooth mathematical root with strict runtime certificate | `implicit_bracketed_root` | Scalar IFT derivative through `custom_root` |

  Explain that caller-supplied `unique_root` and `smooth_branch` are explicit
  scientific assertions, not facts inferred by Jaxstro. Link the reproducible
  forward and derivative evidence artifacts.

- [ ] **Step 3: Request the Phase-B review gate**

  Request independent review of IFT sign/algebra, `custom_root` tangent solve,
  certificate sufficiency, failure gradients, JIT/VMAP behavior, public API
  versatility, and claim honesty. Address every Critical and Important finding.

- [ ] **Step 4: Run the final bounded gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_bracketed_root.py \
    tests/unit/test_implicit_root.py \
    tests/unit/test_implicit_root_evidence.py \
    tests/unit/test_numerics.py \
    tests/unit/test_benchmark_rootfinding_script.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/validation/test_implicit_root_gradients.py \
    tests/validation/test_grad_checks.py \
    tests/integration/test_api_reference.py \
    tests/integration/test_validation_docs.py \
    tests/integration/test_theory_index.py \
    tests/integration/test_readme_examples.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/numerics tests/unit/test_bracketed_root.py \
    tests/unit/test_implicit_root.py tests/unit/test_implicit_root_evidence.py \
    tests/unit/test_numerics.py tests/unit/test_benchmark_rootfinding_script.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/validation/test_implicit_root_gradients.py \
    tests/validation/test_grad_checks.py tests/integration/test_api_reference.py \
    scripts/benchmark_rootfinding.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/numerics tests/unit/test_bracketed_root.py \
    tests/unit/test_implicit_root.py tests/unit/test_implicit_root_evidence.py \
    tests/unit/test_numerics.py tests/unit/test_benchmark_rootfinding_script.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/validation/test_implicit_root_gradients.py \
    tests/validation/test_grad_checks.py tests/integration/test_api_reference.py \
    scripts/benchmark_rootfinding.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  ```

  Run `myst build` from `docs/` and require 61 pages with no content warnings or
  errors. Do not substitute the hour-scale release suite unless a focused failure
  crosses subsystem boundaries.

- [ ] **Step 5: Update status, capture Brain, and commit**

  Update `STATUS.md` with the forward evaluation table, certified derivative
  table, rejected cases, review disposition, and downstream handoff. Capture a
  factual Brain event with the sanctioned `brain` CLI; do not edit the Brain
  repository manually. Commit:

  ```bash
  git add src/jaxstro/numerics docs/10-theory/rootfinding.md \
    docs/40-api/index.md docs/60-validation/index.md \
    docs/validation/implicit-root-gradients.json README.md CLAUDE.md STATUS.md \
    tests/unit/test_bracketed_root.py tests/unit/test_implicit_root.py \
    tests/unit/test_implicit_root_evidence.py tests/unit/test_numerics.py \
    tests/unit/test_benchmark_rootfinding_script.py \
    tests/validation/test_bracketed_root_algorithms.py \
    tests/validation/test_implicit_root_gradients.py \
    tests/validation/test_grad_checks.py tests/integration/test_api_reference.py \
    tests/integration/test_validation_docs.py scripts/benchmark_rootfinding.py
  git commit -m "docs: publish certified implicit scalar roots"
  ```

  Final handoff must provide exact imports/signatures, status and certificate
  tables, value-versus-implicit derivative semantics, focused command outputs,
  forward evaluation evidence, analytic/AD/FD derivative evidence, remaining
  rejected domains, final commit hash, and a clean-tree check.
