---
title: TLUSTY Flux Data Processing
description: >-
  How staged TLUSTY OSTAR/BSTAR flux tar files are converted into local
  processed artifacts without changing released frequency/flux semantics.
---

TLUSTY flux files are staged as source tar archives under `data/atmospheres/tlusty/`.
The converter reads `*.flux.gz` members, parses Fortran-style numeric columns,
and writes local processed artifacts without deleting the source tar files.

Use this page when you need to rebuild or validate the local TLUSTY cache. Use
[](../20-architecture/atmosphere-capabilities.md) for the higher-level capability
matrix and runtime/backend status.

## Local layout

```text
data/atmospheres/tlusty/
  ostars-2002/raw/
  bstars-2007/vturb-2/raw/
  bstars-2007/vturb-10-cn/raw/
  processed/
    catalog.parquet
    catalog_fragments/
    validation/
    tlusty_flux.zarr/
```

The released coordinate is frequency in Hz and the released flux is
`F_nu` in `erg s-1 cm-2 Hz-1`. Wavelength ranges in the catalog are diagnostic
coverage metadata derived as `c / nu`; they are not a unit-converted replacement
for the raw frequency grid.

## The important schema detail

The real TLUSTY files are not one rectangular array per named dataset. Frequency
grids can differ across spectra, even within one BSTAR or OSTAR product. The
processed Zarr layout therefore uses deterministic grid subgroups:

```text
tlusty_flux.zarr/
  bstar_2007_vturb_2/
    grid000/
      frequency_hz
      flux_fnu
      teff
      logg
      vturb_km_s
    grid001/
    ...
  bstar_2007_vturb_10_cn/
    grid000/
    ...
  ostar_2002/
    grid000/
    ...
```

Each `gridNNN` subgroup is rectangular because all rows in that subgroup share
one `frequency_hz` array. The catalog row records `zarr_group`,
`zarr_subgroup`, and `zarr_row`, so runtime code can find the exact native grid
without resampling in the converter.

This is a pedagogical example of the atmosphere policy: preserve the released
scientific coordinate first; add interpolation or spectral-density transforms
only in a tested runtime backend.

## Conversion

Dry-run a staged directory:

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data python \
  scripts/convert_tlusty_flux.py \
  --raw-dir data/atmospheres/tlusty/bstars-2007/vturb-2/raw \
  --dataset tlusty_bstar_2007_vturb_2 \
  --dry-run
```

Convert that directory:

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data python \
  scripts/convert_tlusty_flux.py \
  --raw-dir data/atmospheres/tlusty/bstars-2007/vturb-2/raw \
  --dataset tlusty_bstar_2007_vturb_2 \
  --group-name bstar_2007_vturb_2 \
  --overwrite
```

The three controlled local batches are:

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data python \
  scripts/convert_tlusty_flux.py \
  --raw-dir data/atmospheres/tlusty/bstars-2007/vturb-2/raw \
  --dataset tlusty_bstar_2007_vturb_2 \
  --group-name bstar_2007_vturb_2 \
  --overwrite

env -u VIRTUAL_ENV uv run --no-sync --extra data python \
  scripts/convert_tlusty_flux.py \
  --raw-dir data/atmospheres/tlusty/bstars-2007/vturb-10-cn/raw \
  --dataset tlusty_bstar_2007_vturb_10_cn \
  --group-name bstar_2007_vturb_10_cn \
  --overwrite

env -u VIRTUAL_ENV uv run --no-sync --extra data python \
  scripts/convert_tlusty_flux.py \
  --raw-dir data/atmospheres/tlusty/ostars-2002/raw \
  --dataset tlusty_ostar_2002 \
  --group-name ostar_2002 \
  --overwrite
```

The current measured processed counts are:

```text
tlusty_bstar_2007_vturb_2      981
tlusty_bstar_2007_vturb_10_cn  551
tlusty_ostar_2002              690
```

Validation JSON records readback status, source tar hashes, member provenance,
and float32 storage roundoff. Values far below the float32 dynamic range may
underflow to zero; this is recorded by the roundoff ledger.

## Validation

After conversion, run:

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
  tests/validation/test_atmospheres_local_artifacts.py
```

The TLUSTY portion asserts:

- all three datasets appear in the shared processed catalog.
- row counts are 981, 551, and 690.
- raw source tars still exist.
- validation JSON records source tar hashes, `readback_ok=true`, and
  `raw_deleted=false`.
- each Zarr subgroup has a positive `frequency_hz` array and finite sampled
  `flux_fnu`.
- coverage reports list the datasets as processed with wavelength ranges derived
  from frequency.

## Runtime status

TLUSTY is processed and visible to `AtmosphereLibrary.from_local("data")`.
Runtime backend work must still choose and test the spectral-density policy for
returning a wavelength-domain `SpectrumResult`. The processed artifact keeps raw
`frequency_hz` and `F_nu` intact so that policy remains explicit.
