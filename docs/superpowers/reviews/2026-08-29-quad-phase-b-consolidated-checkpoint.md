# Jaxstro Quad Phase B consolidated checkpoint review

**Scope:** Read-only review of the complete live Phase B surface from the B0
Hyperrectangle owner (`cd7afd3`) through snapshot `9b595dd`.

## Verdict

- Critical findings: none.
- Important findings: none.
- Release disposition: eligible for the complete release gate after the two
  minor dispositions below.

## Reviewed evidence

- B1 tensor and Genz-Malik contracts, work/status boundaries, and transform
  tests.
- B2 exact dyadic sparse coalescing, admissible frontier construction, and
  sparse reference validation.
- B3 Joe-Kuo Sobol data, prefix reuse, scramble/key ownership, fixed-look
  Student-t intervals, and sequential empirical-Bernstein calibration.
- B4 accepted-formula replay, first-order JVP/VJP/Jacobian boundaries,
  heterogeneous quantity representation invariance, canonical/legacy API
  identity, comparison labels, and MyST claim boundaries.
- Observed-memory artifact at source revision `1048b5b`: 72 supported
  fresh-process CPU cases, 24 explicit randomized-array contract rejections,
  and maximum replay-minus-primal peak RSS of 45,973,504 bytes (43.8 MiB),
  below the frozen 10 GiB materiality criterion.

## Minor findings and dispositions

1. **Stale unreachable B4 wording** in `src/jaxstro/quad/integrate.py` said
   replay was introduced in Phase B4. The helper was unreachable after the
   public replay path existed. **Resolved:** removed the dead helper and
   verified the dispatch/replay matrix (50 focused tests).
2. **Observed-memory source distinction.** The memory check validates the
   run-card identity, materiality threshold, and generator hash; it does not
   claim the artifact was measured at a later documentation-only commit.
   **Disposition:** retained the artifact's exact measurement revision
   (`1048b5b`) and record that `82863bd` and `9b595dd` modified only evidence,
   documentation, contracts, and their tests—not numerical owners or the
   measurement harness. This is an honest provenance boundary, not a stale
   measurement claim.

## Claim boundary

Phase B is unusually strong as an evidence-rich JAX-native finite-
hyperrectangle quadrature design: exact sparse coalescing, bounded calibrated
RQMC, accepted-formula first-order replay, typed failure/work records, and
heterogeneous alpha quantity normalization are coherently implemented and
validated. It does not establish universal accuracy, performance, memory, or
external-library superiority; it does not support non-hyperrectangular
geometry, randomized vector/complex confidence intervals, higher-order replay
derivatives, or a separate CPU device-memory metric.
