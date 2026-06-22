---
title: Sonora 2024 Data Processing
description: >-
  How staged Sonora Diamondback spectra are converted into local processed
  artifacts while preserving raw archive provenance.
---

Sonora 2024 spectra are staged as source data under `data/atmospheres/sonora/`.
The converter reads the staged `spectra.zip` archive and writes local
Zarr/Parquet artifacts without deleting the source zip.

Use this page when you need to rebuild or validate the local Sonora cache. Use
[](../20-architecture/atmosphere-capabilities.md) when you want the broader
"which atmosphere libraries are available?" map.

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

That gravity conversion is easy to miss. Sonora filenames encode acceleration in
`m s^-2`; stellar-atmosphere `logg` conventionally means `log10(cm s^-2)`. The
factor of 100 is the unit conversion from meters to centimeters.

## Source-member policy

The local `spectra.zip` may contain macOS resource-fork sidecars:

```text
__MACOSX/spectra/._*.spec
```

Those entries look like spectra by suffix, but they are not model spectra. The
converter skips `__MACOSX/` and `._*` entries, records the skipped member names
in validation JSON, and reports both valid and skipped counts during dry-run.

The current staged archive reports:

```text
valid_count=1440
skipped_count=4
```

The processed catalog count is therefore 1,440, not the raw archive's
1,444 `.spec`-like entries.

## Conversion

Dry-run first:

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data python \
  scripts/convert_sonora_2024.py --dry-run
```

Convert the staged archive:

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data python \
  scripts/convert_sonora_2024.py --overwrite
```

The source zip is not deleted. Validation JSON records readback status,
float32 roundoff, the source zip hash, and member provenance.

## Validation

After conversion, run the local artifact validation:

```bash
env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
  tests/validation/test_atmospheres_local_artifacts.py
```

The Sonora portion asserts:

- `catalog.parquet` exists and has 1,440 rows.
- `sonora_2024.zarr` exists and sampled flux values are finite.
- wavelength unit is `micron`.
- flux unit is `W/m2/m`.
- cgs `logg` spans approximately `3.49..5.50`.
- `spectra.zip` is still present.
- validation JSON records `valid_count=1440`, `skipped_count=4`,
  `readback_ok=true`, and `raw_deleted=false`.

## Runtime status

Sonora is processed and visible to `AtmosphereLibrary.from_local("data")` as
`sonora_2024`. At the time of this document, it is artifact-ready but does not
yet have an implemented `SonoraBackend`. A library query can report Sonora
coverage, but `AtmosphereLibrary.spectrum(...)` should not delegate to Sonora
until the backend interpolation policy is added and tested.
