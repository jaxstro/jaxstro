---
title: TLUSTY Flux Data Processing
description: >-
  How staged TLUSTY OSTAR/BSTAR flux tar files are converted into local
  processed artifacts without changing released frequency/flux semantics.
---

TLUSTY flux files are staged as source tar archives under `data/atmospheres/tlusty/`.
The converter reads `*.flux.gz` members, parses Fortran-style numeric columns,
and writes local processed artifacts without deleting the source tar files.

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

## Conversion

Dry-run a staged directory:

```bash
uv run --extra data python scripts/convert_tlusty_flux.py \
  --raw-dir data/atmospheres/tlusty/bstars-2007/vturb-2/raw \
  --dataset tlusty_bstar_2007_vturb_2 \
  --dry-run
```

Convert that directory:

```bash
uv run --extra data python scripts/convert_tlusty_flux.py \
  --raw-dir data/atmospheres/tlusty/bstars-2007/vturb-2/raw \
  --dataset tlusty_bstar_2007_vturb_2 \
  --group-name bstar_2007_vturb_2 \
  --overwrite
```

Validation JSON records readback status, source tar hashes, member provenance,
and float32 storage roundoff. Values far below the float32 dynamic range may
underflow to zero; this is recorded by the roundoff ledger.
