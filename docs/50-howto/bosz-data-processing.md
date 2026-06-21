---
title: BOSZ Data Processing
description: >-
  How local BOSZ 2025-recomputed resampled spectra are converted into compact
  processed artifacts without vendoring archive data.
---

BOSZ spectra are staged as local source data under `data/atmospheres/`. The STScI
HLSP page labels the archive path as `bosz2024`, but its 2025 Sept. 24 update
states that the models were recomputed and earlier 2024 files should be
replaced.

## Local layout

```text
data/atmospheres/bosz/2025-recomputed/
  provenance/
  wavelength/
    bosz2024_wave_r10000.txt
  scripts/
  raw/
    r10000/
      m+0.00/
  processed/
    catalog.parquet
    catalog_fragments/
    validation/
    bosz_2025_recomputed.zarr/
```

Everything under `data/atmospheres/` is gitignored. Raw gzip spectra and
processed artifacts are local cache products, not package data.

## Precision policy

The converter parses the two released resampled columns as `float64` for
validation, then stores both arrays as `float32` in the processed Zarr artifact.
The shared wavelength grid is stored as `float64`.

This mirrors the NewEra policy: dense spectra are the storage bottleneck, while
the wavelength coordinate is small and should remain precise. Downstream
interpolation or integration may cast spectra back to `float64` for accumulation.

The converter preserves the released BOSZ resampled columns as `flux` and
`continuum` arrays. Unit-normalization and surface-flux interpretation belong at
the downstream physical interface, not in the raw cache converter.

## Conversion

Install the optional data tooling and run a dry-run first:

```bash
uv run --extra data python scripts/convert_bosz_resampled.py --dry-run
```

Convert the staged bridge subset:

```bash
uv run --extra data python scripts/convert_bosz_resampled.py --overwrite
```

For large BOSZ expansion, convert in batches by resolution, metallicity, and
composition. Give each batch a stable group name:

```bash
uv run --extra data python scripts/convert_bosz_resampled.py \
  --raw-dir data/atmospheres/bosz/2025-recomputed/raw/r10000/m-0.50 \
  --wavelength data/atmospheres/bosz/2025-recomputed/wavelength/bosz2024_wave_r10000.txt \
  --group-name r10000_mminus0_50_v2 \
  --delete-raw-after-validate
```

Raw files are deleted only after the batch readback validation succeeds.
