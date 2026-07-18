# Quad Phase B memory optimization addendum

**Status:** diagnostic complete; runtime change not authorized by this evidence

## Trigger

The corrected immutable baseline in
`docs/validation/quad-multidim-performance-baseline.json`, emitted from source
commit `e091640`, reports the following deterministic Sobol replay memory
proxies at level \(8\):

| Dimension | Proxy bytes | Family-median ratio |
| ---: | ---: | ---: |
| 2 | 8,192 | 0.50 |
| 4 | 12,288 | 0.75 |
| 8 | 20,480 | 1.25 |
| 16 | 36,864 | 2.25 |

The frozen trigger is \(2.0\), so the dimension-16 record is eligible for an
optimization investigation.

## Exact owner

The owner is `_qmc_formula` in `src/jaxstro/quad/integrate.py`. Accepted-formula
replay materializes a coordinate array with shape
\((R2^m,d)\), together with formula weights and an active mask. For the
deterministic record, \(R=1\), \(m=8\), and \(d=16\).

## Interpretation

The proxy is not an observed process or device-memory measurement. It is an
analytic storage model, and its growth is the expected
\(O(2^m d)\) cost of retaining explicit coordinates. The triggered record uses
only \(36{,}864\) bytes. Its ratio exceeds the family median because dimension
16 contains eight times as many coordinate values as dimension 2, not because
the implementation has demonstrated superlinear growth or a leak.

The baseline therefore supports an optimization *eligibility* claim, not an
optimization *necessity* claim.

## Candidate local change

If a future measured campaign shows material process or device-memory pressure,
the narrow candidate is a QMC-specific replay primitive that regenerates Sobol
coordinates in bounded chunks during the backward pass instead of retaining
the full coordinate array. It must not alter:

- point identity or ordering;
- scrambling and key semantics;
- accepted level or replicate count;
- value, status, error, or work evidence;
- first-order replay derivatives;
- the explicit rejection of higher derivatives.

## Required regression gates

Before any runtime change is authorized, add process and device-memory
measurements for dimensions \(2,4,8,16\), levels \(8,12,16\) where feasible,
scalar and array payloads, and deterministic plus scrambled formulas. A change
must then pass:

1. bitwise point/formula identity for deterministic Sobol;
2. seeded identity for all scramble families;
3. the complete replay transformation matrix;
4. raw and heterogeneous-quantity representation invariance;
5. no more than \(5\%\) warm-runtime regression;
6. at least \(1.5\) times measured peak-memory improvement in one declared
   material case;
7. two independent optimized suites from distinct clean commits.

## Stop condition and disposition

Stop this optimization branch now. No runtime code changes are warranted while
the only trigger is a \(36{,}864\)-byte analytic proxy with expected linear
dimension scaling. Reopen the addendum only after an observed memory campaign
meets the material-case gate above.

This disposition preserves the immutable trigger evidence and avoids claiming
that a relative family-median threshold alone demonstrates a practical
bottleneck.
