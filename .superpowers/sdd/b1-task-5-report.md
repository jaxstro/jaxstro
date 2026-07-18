# B1 Task 5 implementation report

Date: 2026-07-18

Branch: `codex/quad-phase-b`

Implementation base: `0af8089`

Verdict: implementation and local gates GREEN. The subsequent whole-rung review
is GREEN, so Phase B1 is complete.

## Scope and ownership

Task 5 added validation, transformation, provenance, and release evidence for
the existing B1 deterministic multidimensional methods. A final review
correction changed only the non-stop capability message in
`src/jaxstro/quad/integrate.py` so it names the Phase B1 stop-only boundary and
directs replay users to Phase B4. It did not add a runtime dependency,
implement replay derivatives, add quantities, change a numerical rule, or
broaden the accepted public API.

The tested domain is the dimensionless unit hypercube `[0, 1]^d`. The
parameter rules are:

```text
a_i = 0.35 + 0.05 i
u_i = i / (d + 1)
```

The JSON artifact stores those inputs as exact rational strings. All runtime
tests use float64 unless the transformation matrix explicitly selects float32.

## Independent truth provenance

`scripts/generate_quad_b1_reference.py` evaluates analytic closed forms inside
a generator-owned 100-decimal-digit `mp.workdps(...)` context and reports 80
decimal digits. Rational conversion and every formula evaluation occur inside
that guarded context, which is independent of and restores the caller's
global precision. The generator never calls a Jaxstro method or an external
quadrature routine. The six formula IDs are:

```text
genz-unit-hypercube-six-family-v1:oscillatory
genz-unit-hypercube-six-family-v1:product-peak
genz-unit-hypercube-six-family-v1:corner-peak
genz-unit-hypercube-six-family-v1:gaussian
genz-unit-hypercube-six-family-v1:continuous
genz-unit-hypercube-six-family-v1:discontinuous-first-two-axes
```

The immutable artifact contains 24 records: six families at dimensions
2, 4, 6, and 8. It stores schema version 1, formula-set ID, exact inputs,
decimal truth, generator version, reported precision, working precision, and
generator source SHA-256.

```text
generator SHA-256:
ce950bb6fff540bf5e989bd08f49960fce23dac1dd294439cb054fc169f4d8e0

artifact SHA-256:
227f7d9477de44f46a3448000f251e7ebfe885954f1c9e8e6ab87d293c644c30
```

The caller-precision mutation test failed before the repair because 15- and
37-digit callers produced different records. It now proves byte-identical
records under both callers, exact caller-context restoration, and a frozen
full-precision decimal oracle that differs from the former binary64-derived
tail. An independent 200-digit audit matched every emitted 80-digit record.
The direct JAX closed-form redundancy check passed all 24 records. Artifact
freshness is byte-exact and source-hash-sensitive.

`mpmath==1.3.0` and `scipy==1.16.0` are exactly pinned in the development-only
`reference` dependency group. Both remain absent from
`project.dependencies`; the runtime dependency boundary is unchanged.

## Frozen method controls and truth gates

| Method | Runtime dimensions | Control | External absolute-error gates |
| --- | --- | --- | --- |
| Fixed tensor | 2, 4 | `TensorProduct(GaussianRule(12))`; exact work `12**d` | `2e-8` for four smooth Genz families |
| Adaptive tensor | 2, 4 | `AdaptiveTensorClenshawCurtis(initial_level=2)`; `max_evaluations=32768`; `epsabs=epsrel=1e-8` | `5e-7` smooth; `5e-5` nonsmooth |
| Adaptive cubature | 2, 4, 6, 8 | `AdaptiveCubature(GenzMalik())`; `max_evaluations=500000`; `max_regions=4096`; `epsabs=epsrel=1e-8` | `5e-7` smooth; `5e-5` nonsmooth |

Analytic anchors reuse the existing smooth `gaussian` gate. No threshold,
capacity, rule, level, dimension, or tolerance was loosened after observing a
result.

Structural preflight is separate from runtime certification. Exact
adaptive-tensor initial counts for dimensions 2 through 8 are:

```text
65, 425, 2625, 15625, 90625, 515625, 2890625
```

Exact Genz-Malik point counts are:

```text
17, 33, 57, 93, 149, 241, 401
```

Every exact-capacity case is accepted and every one-under-capacity case raises
`ValueError` without materializing a runtime payload.

## Truth claim matrix

The 72 original runtime cases consist of three analytic anchors and six Genz
families per method/dimension. Every analytic anchor passes its smooth external
truth gate. The accepted Genz claims are:

| Method | Smooth Genz claims | Continuous claims | Discontinuous claims |
| --- | --- | --- | --- |
| Fixed tensor | all four families at d=2,4 | none | none |
| Adaptive tensor | all four families at d=2,4 | d=2 | none |
| Adaptive cubature | all four families at d=2,4,6,8 | d=2,4,8 | d=2,4,6,8 |

Passing an external truth gate is not the same statement as satisfying the
controller's internal `1e-8` estimator tolerance. Several d=6/d=8 cubature
rows have accurate external truth residuals but honestly return
`MAX_EVALUATIONS`. Their statuses are retained below.

## Exact limitation evidence

Fixed Gaussian-12 tensor has no embedded estimator and makes no high-accuracy
claim for unresolved kinks or jumps:

| Family, d | Absolute residual | Status | Evaluations |
| --- | ---: | --- | ---: |
| continuous, 2 | `4.020689850376957e-4` | `ERROR_ESTIMATE_UNAVAILABLE` | 144 |
| continuous, 4 | `3.3746643580634395e-4` | `ERROR_ESTIMATE_UNAVAILABLE` | 20,736 |
| discontinuous, 2 | `1.455241594837775e-2` | `ERROR_ESTIMATE_UNAVAILABLE` | 144 |
| discontinuous, 4 | `4.33358357839128e-2` | `ERROR_ESTIMATE_UNAVAILABLE` | 20,736 |

Adaptive-tensor limitations under the practical 32,768-evaluation control:

| Family, d | Residual | Evaluations / refinements / level | Frontier norm | Status |
| --- | ---: | --- | ---: | --- |
| discontinuous, 2 | `1.21599337392575e-3` | 24,961 / 9 / 7 | `5.352858148793382e-3` | `MAX_EVALUATIONS` |
| continuous, 4 | `3.9738172472236766e-4` | 32,385 / 4 / 4 | `6.916251122028871e-4` | `MAX_EVALUATIONS` |
| discontinuous, 4 | `3.218803677795348e-2` | 32,385 / 4 / 4 | `2.3432820553677375e-2` | `MAX_EVALUATIONS` |

Adaptive-cubature has one external truth limitation:

| Family, d | Residual | Evaluations / refinements / level / regions | Estimator norm | Status |
| --- | ---: | --- | ---: | --- |
| continuous, 6 | `1.0946951546741968e-4` | 499,895 / 1,677 / 14 / 1,678 | `2.654924527754773e-4` | `MAX_EVALUATIONS` |

Its region budget was not exhausted; the evaluation budget was.

## Exact per-case logical work and status

Cell format is `status; evaluations/refinements/level/active_regions`.

```text
Statuses:
C = CONVERGED
M = MAX_EVALUATIONS
U = ERROR_ESTIMATE_UNAVAILABLE

Cases:
A0 = analytic constant
A2 = analytic product moment
Ae = analytic separable exponential
O  = oscillatory
P  = product peak
K  = corner peak
G  = gaussian
Cn = continuous
D  = discontinuous
```

| Method, d | A0 | A2 | Ae | O | P | K | G | Cn | D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed, 2 | U;144/0/0/0 | U;144/0/0/0 | U;144/0/0/0 | U;144/0/0/0 | U;144/0/0/0 | U;144/0/0/0 | U;144/0/0/0 | U;144/0/0/0 | U;144/0/0/0 |
| fixed, 4 | U;20736/0/0/0 | U;20736/0/0/0 | U;20736/0/0/0 | U;20736/0/0/0 | U;20736/0/0/0 | U;20736/0/0/0 | U;20736/0/0/0 | U;20736/0/0/0 | U;20736/0/0/0 |
| adaptive tensor, 2 | C;65/0/2/0 | C;65/0/2/0 | C;225/2/3/0 | C;65/0/2/0 | C;225/2/3/0 | C;225/2/3/0 | C;225/2/3/0 | M;24961/9/7/0 | M;24961/9/7/0 |
| adaptive tensor, 4 | C;2625/0/2/0 | C;2625/0/2/0 | C;29889/4/3/0 | C;2625/0/2/0 | C;4825/1/3/0 | C;29889/4/3/0 | C;29889/4/3/0 | M;32385/4/4/0 | M;32385/4/4/0 |
| cubature, 2 | C;17/0/0/1 | C;17/0/0/1 | C;391/11/4/12 | C;85/2/2/3 | C;119/3/2/4 | C;935/27/6/28 | C;255/7/3/8 | C;3859/113/17/114 | C;7701/226/30/227 |
| cubature, 4 | C;57/0/0/1 | C;337725/2962/17/2963 | C;11799/103/7/104 | C;2451/21/5/22 | C;855/7/3/8 | C;108357/950/13/951 | C;19095/167/8/168 | C;451383/3959/22/3960 | C;26277/230/30/231 |
| cubature, 6 | C;149/0/0/1 | M;499895/1677/16/1678 | M;499895/1677/12/1678 | M;499895/1677/11/1678 | C;1639/5/3/6 | M;499895/1677/17/1678 | M;499895/1677/11/1678 | M;499895/1677/14/1678 | C;84483/283/32/284 |
| cubature, 8 | C;401/0/0/1 | M;499245/622/15/623 | M;499245/622/11/623 | M;499245/622/10/623 | C;401/0/0/1 | M;499245/622/17/623 | M;499245/622/10/623 | M;499245/622/12/623 | M;499245/622/30/623 |

For cubature, every row satisfies
`evaluations = N * (1 + 2 * refinements)` and
`active_regions = refinements + 1`. Fixed tensor has exact `12**d` work.

## JAX transformation matrix

The 51-case integration module covers:

- all three B1 methods;
- eager and separately compiled JIT;
- JIT-of-VMAP with two lanes;
- float32 and float64;
- scalar real, array real, and scalar complex payloads;
- exact status/work identities;
- exact zero stop-gradient and zero JVP tangent; and
- all four tested non-stop strings for each method, including `replay`,
  rejected with the exact B1 capability message.

The standalone transformation gate passed 51 cases in 25.17 s. No replay
derivative is implemented or implied.

## Resource evidence and bounded cache lifetime

The validation autouse fixture clears JAX compilation caches plus adaptive
tensor and cubature host metadata caches after every runtime case, then runs
Python garbage collection. This is part of the committed combined gate.

Timed results:

| Campaign | Result | Elapsed | Peak RSS |
| --- | --- | ---: | ---: |
| Adaptive tensor practical matrix | 18 passed | 214.22 s | 1,094,713,344 B |
| Adaptive cubature claim+limitation matrix | 36 passed | 20.16 s | 346,193,920 B |
| Exact 72-row evidence capture | complete | 239.35 s | 1,037,664,256 B |
| Task 5 validation + transformations | 164 passed | 290.21 s | 1,083,047,936 B |
| B1 deterministic scientific gate | 451 passed | 390.01 s | 3,973,087,232 B |
| Complete Quad engineering matrix | 788 passed | 1,035.80 s | 4,755,963,904 B |

The original adaptive-tensor 250,000-evaluation setting remains only an
incomplete non-default stress record. Fresh d=2/d=4 representatives peaked at
12,413,124,608/842,678,272 bytes. A combined run completed 11 cases with no
threshold miss before a bounded stop at 684.01 s and 16,738,811,904-byte peak
RSS. It is not a release-ready default scientific gate.

## TDD and verification

The initial RED proved the missing truth infrastructure:

```text
2 failed, 111 deselected in 0.22 s
```

The failures were exactly the missing reference dependency/artifact contracts;
there were no collection errors.

Final commands and results:

```text
uv run --locked --group reference python
  scripts/generate_quad_b1_reference.py --check
fresh tests/validation/data/quad-b1-genz-reference.json

uv run --no-sync pytest -q
  tests/validation/test_quad_multidim_deterministic.py
  tests/integration/test_quad_multidim_deterministic_transforms.py
164 passed in 290.21 s

uv run --no-sync pytest -q
  tests/unit/quad/test_tensor.py
  tests/unit/quad/test_adaptive_tensor.py
  tests/unit/quad/test_genz_malik.py
  tests/unit/quad/test_cubature.py
  tests/validation/test_quad_multidim_deterministic.py
  tests/integration/test_quad_multidim_deterministic_transforms.py
451 passed in 390.01 s

uv run --no-sync pytest -q tests/unit/quad
  tests/integration/test_quad_multidim_transforms.py
  tests/integration/test_quad_multidim_deterministic_transforms.py
  tests/validation/test_quad_multidim_deterministic.py
788 passed in 1035.80 s

uv run --no-sync ruff check src tests
All checks passed!

uv run --no-sync ruff format --check src tests
341 files already formatted

uv run --no-sync mypy src/jaxstro
Success: no issues found in 128 source files
```

The independent Task 5 review then exposed that the original generator inherited
the default 15-digit global `mpmath` context before formatting 80 decimal
digits. The precision/provenance repair followed a second strict TDD cycle:

```text
RED:
uv run --no-sync pytest -q
  tests/validation/test_quad_multidim_deterministic.py
  -k 'reference_generator_owns_precision or
      reference_artifact_records_reported'
2 failed, 113 deselected in 0.29 s

GREEN:
uv run --locked --group reference python
  scripts/generate_quad_b1_reference.py --check
fresh tests/validation/data/quad-b1-genz-reference.json

uv run --no-sync pytest -q
  tests/validation/test_quad_multidim_deterministic.py
  -k 'reference or direct_closed_forms'
28 passed, 87 deselected in 5.32 s

uv run --no-sync pytest -q
  tests/validation/test_quad_multidim_deterministic.py -k limitation
8 passed, 107 deselected in 63.71 s

uv run --no-sync pytest -q
  tests/integration/test_quad_multidim_deterministic_transforms.py
51 passed in 43.27 s

uv run --no-sync ruff check
  scripts/generate_quad_b1_reference.py
  tests/validation/test_quad_multidim_deterministic.py
All checks passed!

uv run --no-sync ruff format --check
  scripts/generate_quad_b1_reference.py
  tests/validation/test_quad_multidim_deterministic.py
2 files already formatted

uv run --no-sync mypy src/jaxstro
Success: no issues found in 128 source files
```

All frozen limitation residuals remained within their original strict
regression tolerances, so none was changed. Method controls, scientific
thresholds, and the development-only dependency boundary are unchanged. The
only runtime-source correction is the exact fail-closed capability message:

```text
<Method> supports only gradient="stop" in Phase B1;
gradient="replay" is introduced in Phase B4
```

All tested non-stop strings must produce that exact method-specific message.

## B4 carry-forward and next checkpoint

B4 must benchmark adaptive tensor separately in dimensions 5, 6, 7, and 8
with compile time, warm runtime, process/device memory, dtype, payload, and
capacity evidence. It must disclose the intrinsic tensor frontier and
fixed-capacity `O(C d)` metadata growth.

B4 must also benchmark cubature payload shape, dtype, reachable store
capacity, compile/warm runtime, and process/device memory. B1 certifies only
the bounded scalar-payload cases above; it is not a universal
payload/dtype/store-memory safety claim.

The whole-rung B1 numerical-method, JAX, API, provenance, and test-quality
review is GREEN. Pause at the completed B1 boundary; begin B2 only on explicit
continuation.
