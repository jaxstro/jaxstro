# Phase B observed replay-memory run card

## Question and decision

Does accepted-formula Sobol replay create a material observed memory burden
relative to the matched primal formula? A case is material only when its
fresh-process replay peak RSS exceeds the matched primal peak RSS by at least
**10 GiB**. If no material case appears, make no runtime change. If one does,
the only candidate remains bounded regeneration of Sobol coordinates during
backward replay, subject to the approved addendum's 1.5x peak-memory and 5%
warm-runtime gates.

## Frozen configuration

- Platform: the emitted artifact records macOS version, Python, JAX, jaxlib,
  backend, device, precision, and source revision.
- Dimensions: 2, 4, 8, and 16.
- Sobol levels: 8, 12, and 16, subject only to an explicit infeasible-process
  record rather than silent omission.
- Formulae: deterministic `Sobol` and eight-replicate LMS-plus-shift
  `ScrambledSobol`.
- Payloads: scalar and a four-component real array for deterministic Sobol.
  Randomized QMC array payloads are recorded as an intentional Phase B contract
  rejection because its calibrated interval is real-scalar only.
- Modes: primal value and first-order replay gradient.
- Protocol: one fresh Python process per supported case, compile and one warm
  execution timed separately, with macOS `/usr/bin/time -l` measuring maximum
  resident set size for that process.
- Memory metrics: peak process RSS in bytes, paired replay-minus-primal RSS in
  bytes, and the 10-GiB materiality decision. The active CPU backend exposes no
  reliable separate device-memory metric; this is recorded, not inferred.

The baseline artifact remains immutable. This campaign writes a separate
evidence artifact and never changes numerical tolerance, calibration, or
runtime behavior merely to alter a memory result.
