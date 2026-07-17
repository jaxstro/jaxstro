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
| `best_method` | Independent practical choice for each library. | 16 |

## Cases and truth

The catalog includes smooth, vector-valued, localized, nonsmooth, endpoint-singular, improper, oscillatory, expensive, narrow-feature, and nonfinite cases. Truth comes from analytic derivations or an independent NumPy Gauss-Legendre convergence ladder.

```{math}
\varepsilon_{\mathrm{obs}} = \lVert \widehat{I} - I \rVert_{\infty}.
```

## Accuracy and calibration

45 of 78 records warrant direct performance interpretation. Reported-error ratios are calibration diagnostics, not automatic bound claims.

## Work

Reported and normalized evaluations are retained separately. In particular, Quadax Clenshaw-Curtis interval work is converted to actual node evaluations before comparable-work analysis.

## Compile, warm, VMAP, and AD timing

Lowering, compilation, warm scalar execution, VMAP batches of 16 and 128, JVP, and supported reverse mode are measured separately with synchronized outputs and interleaved library order.

```{admonition} Timing scope
:class: note
Wall time is informational for this recorded CPU environment and is never used as a deterministic freshness gate.
```

## Failure semantics

Jaxstro fails closed on nonfinite integrand samples. Quadax 0.2.13 masks nonfinite samples to zero, so that case is recorded as a semantic difference and excluded from performance claims.

## Environment

Source revision: `7c3a61243d4d9a2d1f19ce8b43b19adbba2371c8`

- `backend`: `cpu`
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

Status: `baseline_pending_independent_review`.

Approved optimization gates are evaluated after fairness review.

## Warranted limitations

- CPU wall time is hardware- and load-dependent and is not a CI gate.
- Family-matched labels do not imply identical algorithms or failure semantics.
- Jaxstro replay derivatives and Quadax adaptive-loop derivatives have different policies.
- No backend-portable peak device-memory metric is claimed.
- The nonfinite case intentionally exposes Quadax zero-substitution behavior.
