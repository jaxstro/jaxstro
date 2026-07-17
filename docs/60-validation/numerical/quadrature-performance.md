# Quadrature performance and comparison evidence

## Purpose

This report separates numerical correctness from performance. Every library is judged against declared mathematical truth before any timing ratio is interpreted.

```{admonition} Reading rule
:class: important
A faster result is not a better result unless its value, status, work accounting, and derivative checks are warranted.
```

## Comparison label definitions

| Label | Meaning | Records |
| --- | --- | ---: |
| `exact` | Same embedded rule family and order. | 11 |
| `strong_match` | Closely matched global refinement capacity. | 6 |
| `node_matched` | Same local node count with different estimators. | 10 |
| `family_matched` | Same broad method family; algorithms differ. | 26 |
| `capability` | Related capability only; no algorithmic equivalence claim. | 9 |
| `best_method` | Predeclared practical choice using the frozen library-specific adapter settings. | 16 |

## Cases and truth

The catalog includes smooth, vector-valued, localized, nonsmooth, endpoint-singular, improper, oscillatory, expensive, narrow-feature, and nonfinite cases. Truth comes from analytic derivations or an independent NumPy Gauss-Legendre convergence ladder.

```{math}
\varepsilon_{\mathrm{obs}} = \lVert \widehat{I} - I \rVert_{\infty}.
```

## Accuracy and calibration

49 of 78 records warrant primal timing interpretation. 30 warrant JVP timing, and 18 warrant a direct two-library reverse-mode comparison.

There are 15 records with available derivative truth that fail at least one JVP gate. Records without declared derivative truth are explicitly ineligible for AD comparisons. Reported-error ratios are calibration diagnostics, not automatic bound claims.

## Work

Reported and normalized evaluations are retained separately. In particular, Quadax Clenshaw-Curtis interval work is converted to actual node evaluations before comparable-work analysis.

## Compile, warm, VMAP, and AD timing

Lowering, compilation, warm scalar execution, VMAP batches of 16 and 128, JVP, and supported reverse mode are measured separately with synchronized outputs and interleaved library order. Every method-case record is measured in a fresh Python process so internal compilation caches cannot leak between records.

Using a descriptive stability threshold of $\operatorname{MAD}/\operatorname{median} \le 0.10$, 559 of 752 supported timed library-mode measurements are stable. Automatic regression decisions use the stricter predeclared ratio, minimum-case, and two-MAD separation rules.

```{admonition} Timing scope
:class: note
Wall time is informational for this recorded CPU environment and is never used as a deterministic freshness gate.
```

## Failure semantics

Jaxstro fails closed on nonfinite integrand samples. Quadax 0.2.13 masks nonfinite samples to zero, so that case is recorded as a semantic difference and excluded from performance claims.

## Primary matched timing ratios

Each timing ratio is $t_{\mathrm{jaxstro}}/t_{\mathrm{quadax}}$; each work ratio is $N_{\mathrm{jaxstro}}/N_{\mathrm{quadax}}$. Values above one therefore favor Quadax for that metric. The parenthetical timing label states whether the Jaxstro slowdown exceeds twice the larger MAD; it is not a winner declaration.

| Case | Family | Compile | Warm | VMAP 128 | JVP | Work |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `smooth_exponential` | `gauss_kronrod` | not warranted | not warranted | not warranted | not warranted | not warranted |
| `smooth_exponential` | `clenshaw_curtis` | 1.07 | 0.71 (not separated) | 0.02 (not separated) | 0.77 (not separated) | 1.00 |
| `smooth_exponential` | `romberg` | 1.66 | 1.63 (separated) | 2.69 (separated) | 1.67 (separated) | 1.00 |
| `localized_gaussian` | `gauss_kronrod` | not warranted | not warranted | not warranted | not warranted | not warranted |
| `localized_gaussian` | `clenshaw_curtis` | 1.20 | 0.85 (not separated) | 0.23 (not separated) | not warranted | 1.22 |
| `breakpoint_kink` | `gauss_kronrod` | not warranted | not warranted | not warranted | not warranted | not warranted |
| `breakpoint_kink` | `clenshaw_curtis` | 1.26 | 0.81 (not separated) | 0.02 (not separated) | not warranted | 1.00 |
| `oscillatory_cosine` | `gauss_kronrod` | not warranted | not warranted | not warranted | not warranted | not warranted |
| `oscillatory_cosine` | `clenshaw_curtis` | 1.20 | 1.38 (separated) | 0.60 (not separated) | 1.21 (separated) | 1.85 |
| `oscillatory_cosine` | `romberg` | 1.61 | 2.14 (separated) | 1.56 (separated) | 2.30 (separated) | 1.00 |
| `expensive_identity` | `gauss_kronrod` | not warranted | not warranted | not warranted | not warranted | not warranted |
| `expensive_identity` | `clenshaw_curtis` | 1.06 | 0.73 (not separated) | 0.01 (not separated) | 0.79 (not separated) | 1.00 |
| `expensive_identity` | `romberg` | 1.77 | 1.74 (separated) | 6.32 (separated) | 2.03 (separated) | 1.00 |

```{admonition} Scope of this table
:class: note
The table is restricted to the predeclared primary float64, family-matched, representative, ratio-eligible lane. Capability-only and practical-choice records remain in the machine-readable artifact but cannot drive matched-method superiority claims.
```

## Environment

Source revision: `2378e4d19d0e63bd13e17f8dbf6b499128f1ae02`

- `backend`: `cpu`
- `cpu_model`: `Apple M2 Max / Mac14,5`
- `device`: `cpu:0`
- `device_kind`: `cpu`
- `jax_version`: `0.10.1`
- `jaxlib_version`: `0.10.1`
- `machine`: `arm64`
- `numpy_version`: `2.4.6`
- `operating_system`: `macOS-26.1-arm64-arm-64bit-Mach-O`
- `processor`: `arm`
- `python_version`: `3.13.7`
- `quadax_version`: `0.2.13`

## Optimization decision

Status: `optimization_required`.

The frozen primary-lane trigger assessment is recorded in the baseline payload. Profile before changing runtime code.

Fired triggers: `warm`, `vmap_128`, `ad_jvp`.

## Warranted limitations

- CPU wall time is hardware- and load-dependent and is not a CI gate.
- Family-matched labels do not imply identical algorithms or failure semantics.
- Jaxstro replay derivatives and Quadax adaptive-loop derivatives have different policies.
- No backend-portable peak device-memory metric is claimed.
- The nonfinite case intentionally exposes Quadax zero-substitution behavior.
