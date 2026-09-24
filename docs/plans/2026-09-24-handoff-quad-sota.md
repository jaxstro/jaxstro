# Handoff: jaxstro cleanup and the quad SOTA program (2026-09-24)

Paste the section "Prompt for the next session" into a fresh session opened in
`~/projects/jaxstro-dev/jaxstro`. The rest of this file is the reference it
points to.

## Prompt for the next session

You are continuing Anna's jaxstro cleanup and the program to make `jaxstro.quad`
state of the art and fast. Work in ENGINEERING mode unless a scientific choice
comes up.

Read first, in this order:

1. `CLAUDE.md` (writing rule, JAX/AD contracts, load-bearing invariants).
2. `docs/plans/2026-09-24-handoff-quad-sota.md` (this file).
3. `docs/plans/2026-09-24-cleanup-ledger.md`: the task list, 110 tasks, keyed
   by ID. It is the source of truth; update its rows (status, evidence,
   commits) as you work and commit it with the change. The claude.ai artifact it
   was exported from is a frozen snapshot.

Working rules Anna set (all still in force):

- Targeted tests only. Choose owner tests by grepping all of `tests/` for the
  changed symbols (`rg -l "<symbol>" tests -g '*.py'`), never by a directory or
  file-name glob: twice on 2026-09-24 a test outside `tests/unit/quad/` broke CI.
  Never pass pytest an empty path list (it runs the whole suite). No full suite
  unless Anna asks. Other sessions run timed benchmarks on this machine: keep
  local runs small and never run diagnostics while a timed run is going.
- Clean up tests you touch: replace exact-phrase and count pins with structural
  checks in the same commit.
- Do not remove methods because no downstream package uses them yet; jaxstro is
  meant to be a versatile differentiable numerical-methods package.
- Scientific assumptions, pass/fail tolerances and statistical constants are
  Anna's to approve: propose them with the measured basis and wait.
- Before claiming a fix, show fresh command output. Report errors plainly.
- Push in batches after `ruff check --no-fix`, `ruff format --check` and
  `mypy src/jaxstro` pass locally; read the CI result later. The full gate is
  `.github/workflows/full-gate.yml`: 8 parallel stages from
  `bash scripts/check.sh <stage>`. `tests.yml` is the fast gate on push.
- Commits end with the Co-Authored-By line from the session's attribution
  instructions.

Start with QD-11 (design approved 2026-09-24, below). Before writing solver
code, re-run the ε-algorithm prototype described below to confirm the numbers,
then implement Phase 1 test-first.

## State at handoff

- `main` at `3de93bc` (pushed). Full gate run 36063431163 green on all 8 stages.
- Last green run covers: Gaussian-rule weight fix and caching, sampled
  integration axis fix and nonuniform Simpson, `fixed()` zero-width derivative,
  multidim replay fail-closed, AdaptiveSmolyak capacity blocking, complement-pair
  adaptive regions (QD-02 Phase A), Romberg scope docs, helper dedupes, RQMC
  ln(4/alpha) interval, figure check (bytes on macOS arm64, render-only
  elsewhere), CI split into parallel stages.
- Uncommitted in jaxstro: only the pre-existing `.gitignore` edit (not ours; do
  not stage it).
- Downstream, committed locally but NOT pushed (each repo also carries other
  sessions' unpushed commits, so ask Anna before pushing):
  - fluxax `0321bcc` (numerics owner-module imports), repo 6 ahead of origin.
  - progenax `cc3a0e1` (owner-module imports, jaxstro.quad cutover), 5 ahead.
- gravax migrated its imports in `551d4238` on branch
  `codex/dynamic-direct-hierarchy` (not `main`). Its frozen
  `validation/evidence/audit-2026-09-14/` tree still has a flat import; leave it.

## Next tasks, in recommended order

1. **QD-11, QAGS-style extrapolation (approved design).**
   - New method `quad.GaussKronrodExtrapolated(pair=21)`, new
     `ErrorKind.EXTRAPOLATION`; `GaussKronrod` unchanged.
   - Port QUADPACK `qagse` + `qelg` into the fixed-capacity controller in
     `src/jaxstro/quad/_adaptive.py` (regions are `(1 + t, 1 - t)` pairs, see
     `ReferencePartition`): track bisection depth per region, record the global
     sum as a sequence entry at each new smallest level, Wynn table of at most
     50 entries.
   - Safeguards (required): extrapolation error
     `|e2 - e1| + |e1 - e0| + 5 eps |r|`; use the extrapolated result only if its
     error beats the plain adaptive error; divergence and roundoff tests end with
     `DIVERGENCE_SUSPECTED` (currently reserved, now emitted) or
     `ROUNDOFF_LIMITED`.
   - Phase 1: `gradient="stop"` only; `gradient="replay"` raises a clear error.
     Phase 2 (separate design, needs Anna): replay through the ε-table with each
     recorded sum stored as a replayable formula.
   - Prototype to reproduce first: GK21 (nodes and Kronrod weights from
     `jaxstro.quad._gk.gauss_kronrod_data(GaussKronrod(21))`) summed over the
     dyadic partition `[a, a+h_k], [a+h_k, a+2h_k], ..., [a+(b-a)/2, b]` with
     `h_k = (b-a) 2^-k`, k = 1..15, then Wynn's ε-algorithm on that sequence.
     Measured 2026-09-24 at 15 levels: x^-0.9 on [0,1] 5.3e-16 (raw 1.6e-1);
     (x-1)^-0.5 on [1,2] 6.8e-15 (raw 9.0e-5); x^-0.5 ln x on [0,1] 4.4e-16;
     1/(x ln^2 x) on [0,1/2] 6.3e-3 with a naive estimate of 2.8e-5, which is why
     the safeguards are required.
   - Tests: those four integrands, x^-0.99, a `RightInfinite` case, the honesty
     contract (CONVERGED implies true error within tolerance), and evaluation
     counts against `scipy.integrate.quad` (QAGS). Note scipy QAGS itself
     reports false convergence on 1/(x ln^2 x) at epsrel 1e-3 (6.1e-3 error);
     document if the safeguards cannot catch it. Any new pass/fail tolerance
     needs Anna's approval.
   - Docs: `docs/20-methods/approximation-integration/adaptive-quadrature.md`
     (endpoint section and table), `docs/50-api/approximation-integration/quad.md`,
     contract registry (`scripts/build_contract_registry.py`, run the docs gate).
2. **ARC-03 and ARC-09 (approved): remove jaxstro's flat `numerics` re-exports,
   delete `numerics/integration.py`, `numerics/quadrature.py` and the `trapz` /
   `cumulative_trapz` aliases**; move any unique rule into `jaxstro.quad`; add a
   test that fails if a flat name returns; then
   `rg -g '*.py' "from jaxstro\.numerics import" ..` across the workspace.
   Unblocked locally now that gravax migrated, but gravax `main` still has the
   old imports: tell Anna before landing, since gravax installs jaxstro editable.
   Also update `docs/50-api/approximation-integration/quad.md` ("Compatibility
   boundary", "Migrating to jaxstro.quad") and `tests/unit/quad/test_sampled.py`
   (it imports `jaxstro.numerics.integration`).
3. **QD-12 open items**: (4) AdaptiveSmolyak returns MAX_INDICES even when the
   true error is 1.3e-13 (cos(100 x0), max_nodes=256), because an unevaluable
   frontier index blocks convergence; its reported estimate (5.6e-17) does not
   reflect the blocked index. (5) Zero-width derivative policy differs: 1-D
   gives f(bound), multidim gives NaN (INVALID_INPUT). Both need Anna.
4. **QD-14 (needs an API decision)**: tanh-sinh complement abscissae (retain
   nodes while 1 - |t| is representable, down to ~1e-300) and an optional
   offset-aware integrand signature (Boost's f(x, xc)) so finite endpoints
   a != 0 can resolve below the spacing of a; today that limit is ~1e-8 for
   (x-1)^-0.5.
5. **QD-10 multidim hazards**: AdaptiveSmolyak converges on all-zero initial
   surpluses with epsabs=0; adaptive tensor Clenshaw-Curtis discards its refined
   candidates; Genz-Malik estimate underestimates on discontinuous integrands
   (same as scipy).
6. **QD-07 speed and coverage**: O(n) Gauss rules (Hale-Townsend /
   Glaser-Liu-Rokhlin), FFT Clenshaw-Curtis weights (Waldvogel), hash-based Owen
   scrambling (Burley 2020, measured 20x slower than LMS today), lattice rules,
   Gauss-Radau/Lobatto, QAGI/QAWO/QAWS/QAWC.
7. **QD-08 file splits**: `_tensor.py` 1,339 lines (fixed rules / CC ladder /
   hash cache / controller), `_sparse.py` 1,237 (dyadic rules / fixed index set /
   frontier / cache+controller), `qmc.py` 1,010 (method dataclasses, budget
   validators, adaptive inspection); shared result builders; one dispatch table
   in `integrate.py`. Behaviour-preserving; prove with owner tests and evidence
   `--check` scripts.

## Decisions Anna made on 2026-09-24 (do not re-ask)

- CI-10 split gate into parallel jobs; CI-11 fast gate on push; DOC-07 quad
  labelled experimental with a qualification track to follow.
- ARC-09 owner-module paths only; ARC-03 cut callers over and delete shims.
- Gaussian rules: remove NumPy hermgauss; parity tolerances 64 eps nodes,
  16 n eps weights, 1e-13 tail integrals.
- Freshness policy extended to the Kronrod golden (1 ULP on macOS arm64,
  rtol 1e-13 / atol 1e-14 elsewhere).
- QD-02 endpoint design (complement-pair regions); moving a rounded node one
  ulp inside with the dominance rule for tanh-sinh.
- Romberg: option 3, smooth integrands only, documented; two-level agreement
  was measured and rejected (cos(200x) still accepted at epsrel 1e-8).
- Figure check: exact bytes on macOS arm64, render-only elsewhere.
- RQMC empirical-Bernstein two-sided interval uses ln(4/alpha_k).
- QD-11 design above.

## Lessons from this session

- A heuristic that drops quadrature nodes without accounting for their mass
  produced false CONVERGED results (commit 31f1c31, reverted in d542757). Run an
  adversarial review of numerical changes before building on them.
- XLA on CPU flushes subnormals: `nextafter(0.0, 1.0)` becomes 0 inside JAX.
- `jnp.where` substituting a constant NaN is dropped by reverse-mode transpose;
  a multiplicative mask on the input tangents survives.
- Golub-Welsch eigenvector weights are accurate only to ~eps absolute; the
  Christoffel form keeps relative accuracy in the tails.
- Below Nyquist, dyadic samples of an oscillatory integrand look smooth; no
  stopping rule on those samples alone can detect it.
