---
title: PHOENIX/NewEra Data Processing
description: >-
  How the local NewEra low-resolution text spectra are converted into compact
  processed artifacts without vendoring raw atmosphere data.
---

The local PHOENIX/NewEra low-resolution spectra are source data, not runtime
artifacts. Each raw text file alternates a short model metadata row with one very
long flux-vector row. The local extracted cache is large enough that conversion
must be batch-oriented and deletion-safe.

## Local layout

```text
data/atmospheres/newera/
  manifest/
    list_of_available_NewEraV3_models.txt
  raw/
    PHOENIX-NewEraV3-LowRes-SPECTRA/
  processed/
    manifest.parquet
    catalog.parquet
    catalog_fragments/
    validation/
    newera_lowres_v3.zarr/
```

Everything under `data/atmospheres/` is gitignored. The raw text files and
processed artifacts are local cache products, not package data.

## Precision policy

The converter parses each raw flux row as `float64` for validation, then stores
the spectra as `float32` in the processed Zarr artifact.

This is the default because the released text rows carry limited decimal
precision, while `float32` keeps roughly seven significant decimal digits and
cuts dense storage in half relative to `float64`. Downstream interpolation or
band integration may cast the stored arrays back to `float64` for accumulation.

The converter records the maximum absolute and relative `float64 -> float32`
roundoff per source file in `processed/validation/*.json`. Raw text should only
be deleted after the per-file validation ledger reports `readback_ok: true`.

## Conversion

Install the optional data tooling and run a dry-run first:

```bash
uv run --extra data python scripts/convert_newera_lowres.py --dry-run
```

Convert one source file, validate it, and delete only that validated raw file:

```bash
uv run --extra data python scripts/convert_newera_lowres.py \
  --file PHOENIX-NewEraV3-LowRes-SPECTRA.Z-0.0.alpha=0.6.txt \
  --delete-raw-after-validate
```

Continue batch-by-batch. If conversion fails, the raw source file remains in
place. If validation succeeds and deletion is enabled, only the converted raw
`.txt` file is removed; the upstream tarball is not touched.
