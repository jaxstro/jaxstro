---
title: Atmosphere Capabilities
description: >-
  What jaxstro can currently do with local atmosphere spectra, what each dataset
  contributes, and where the downstream package boundary begins.
---

`jaxstro.atmospheres` is the shared atmosphere-spectrum layer for the ecosystem.
Its job is deliberately small and important:

```text
stellar atmosphere parameters + local library metadata -> spectrum + provenance
```

It does not compute colors, magnitudes, bolometric corrections, extinction,
survey observables, or fit decisions. Those live above jaxstro. Fluxax owns
filters and photometry; population and stellar-parameter packages decide which
library candidate they want to request.

## The mental model

Think of the atmosphere system as three nested layers.

```text
source archives       processed artifacts        runtime boundary
-------------         -------------------        ----------------
zip/tar/text/gzip ->  catalog + Zarr store  ->   SpectrumResult
                      validation ledger          status + provenance
```

The source archives are scientific inputs and are preserved. The processed
artifacts are local, gitignored cache products that make runtime access
deterministic and inspectable. The runtime boundary returns a spectrum; it does
not silently turn spectra into photometry.

## Capability matrix

```{list-table} Local atmosphere libraries
:header-rows: 1
:label: tbl-atmosphere-capabilities

* - Dataset
  - Current local state
  - Count
  - Native spectral coordinate
  - Native flux column
  - Runtime backend
* - PHOENIX/NewEra low-resolution v3
  - Processed
  - 38,352
  - wavelength, nm
  - flux-lambda array from the released low-resolution product
  - `NewEraBackend`
* - BOSZ 2025-recomputed bridge subset
  - Processed
  - 3,303
  - wavelength, angstrom
  - released resampled `flux` and `continuum`
  - `BoszBackend`
* - Sonora Diamondback 2024
  - Processed
  - 1,440 valid spectra; 4 macOS resource-fork entries skipped
  - wavelength, micron
  - `W/m2/m`
  - artifact-ready; backend not yet implemented
* - TLUSTY BSTAR 2007, `vturb=2`
  - Processed
  - 981
  - frequency, Hz
  - `F_nu`, `erg s-1 cm-2 Hz-1`
  - artifact-ready; backend not yet implemented
* - TLUSTY BSTAR 2007, `vturb=10` with C/N variants
  - Processed
  - 551
  - frequency, Hz
  - `F_nu`, `erg s-1 cm-2 Hz-1`
  - artifact-ready; backend not yet implemented
* - TLUSTY OSTAR 2002
  - Processed
  - 690
  - frequency, Hz
  - `F_nu`, `erg s-1 cm-2 Hz-1`
  - artifact-ready; backend not yet implemented
```

The processed counts are validated local artifact counts, not promises that
model families agree physically in their overlaps. The overlap tools are
diagnostics: they check finite spectra and shared wavelength domains, then record
normalized differences so a human can decide what the mismatch means.

## Why processed does not always mean runtime-ready

Processed artifacts answer: "Can jaxstro read, validate, index, and preserve this
library without re-parsing source archives?"

Runtime backends answer a different question: "Can jaxstro safely select a local
cell and return a `SpectrumResult` for this family under an explicit interpolation
and unit policy?"

NewEra and BOSZ already have runtime backends. Sonora and TLUSTY have validated
processed artifacts, coverage reporting, and provenance ledgers; their runtime
backends should be added only against those proven schemas.

This separation prevents a common failure mode: treating "the files were parsed"
as if it also proved the runtime interpolation policy.

## Dataset-specific lessons

### Sonora

Sonora filenames encode gravity in SI acceleration, for example `g31` means
`31 m s^-2`. `AtmosphereParams.logg` is cgs, so the converter stores both:

```text
g_m_s2 = 31
logg = log10(31 * 100)
```

The converter also skips archive members such as `__MACOSX/spectra/._*.spec`.
Those are macOS resource-fork sidecars, not spectra. The skip list is recorded
in validation JSON so the count difference is visible: the archive listing has
1,444 `.spec`-like entries, but only 1,440 valid spectra.

### TLUSTY

TLUSTY flux products are natively frequency-domain `F_nu` tables. jaxstro keeps
that native representation in the processed artifacts. Wavelength ranges in
coverage reports are derived metadata:

```text
lambda_nm = c_nm_s / frequency_hz
```

The real TLUSTY files are ragged: not every spectrum in one named dataset shares
the same frequency grid. The processed Zarr store therefore uses deterministic
`gridNNN` subgroups. Each subgroup has its own `frequency_hz`, `flux_fnu`, and
coordinate arrays; the catalog row tells you which subgroup and row to read.

That design is less visually tidy than a single giant rectangle, but it is
scientifically honest. It avoids resampling or interpolating the archive product
inside the converter.

## Inspecting local coverage

Use the coverage report when you want to answer "what can this checkout see?"

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data python \
  scripts/report_atmosphere_coverage.py --data-dir data --format markdown
```

The report is catalog-first. A processed catalog appears even if no runtime
backend is implemented yet; `backend_available` tells you whether
`AtmosphereLibrary.spectrum(...)` can delegate to a backend today.

## Validation gates

The artifact validation tests deliberately check the local processed files, not
only synthetic fixtures:

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
  tests/validation/test_atmospheres_local_artifacts.py
```

Those tests assert the measured counts, source archive preservation, units,
finite sampled flux, readback validation, and processed coverage status. They are
the quickest way to tell whether a local atmosphere cache is trustworthy enough
for backend work.
