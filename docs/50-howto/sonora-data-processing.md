---
title: Sonora 2024 Data Processing
description: >-
  How staged Sonora Diamondback spectra are converted into local processed
  artifacts while preserving raw archive provenance.
---

Sonora 2024 spectra are staged as source data under `data/atmospheres/sonora/`.
The converter reads the staged `spectra.zip` archive and writes local
Zarr/Parquet artifacts without deleting the source zip.

## Local layout

```text
data/atmospheres/sonora/2024/
  provenance/
  raw/
    spectra.zip
  processed/
    catalog.parquet
    catalog_fragments/
    validation/
    sonora_2024.zarr/
```

Filename gravity is preserved as `g_m_s2`. The converter also records cgs `logg`
as `log10(g_m_s2 * 100)` for compatibility with `AtmosphereParams`. Wavelengths
remain in microns and fluxes remain in the released `W/m2/m` column.

## Conversion

Dry-run first:

```bash
uv run --extra data python scripts/convert_sonora_2024.py --dry-run
```

Convert the staged archive:

```bash
uv run --extra data python scripts/convert_sonora_2024.py --overwrite
```

The source zip is not deleted. Validation JSON records readback status,
float32 roundoff, the source zip hash, and member provenance.
