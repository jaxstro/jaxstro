---
title: Release notes
description: >-
  Changelog, versioning, and deprecation policy for jaxstro — including the
  dx-outside cumulative_trapz reconciliation and the A_RAD correction.
---

This section will hold the changelog, the semantic-versioning scheme, and the
deprecation policy that downstream packages pin against (`jaxstro>=X,<Y`). The
canonical changelog lives in the repository's `CHANGELOG.md`; this page will render
and annotate it for the docs site.

Two reconciliations from the 0.1.0 line are worth flagging here because they can
shift downstream numbers:

- **`cumulative_trapz` standardized to dx-outside.** Former dx-inside call sites may
  drift by ~1 ulp; this is the expected rounding difference, not a regression — see
  [](../10-theory/cumulative-trapz.md).
- **`A_RAD` corrected** to $7.565733250\times10^{-15}\ \erg\,\mathrm{cm^{-3}\,K^{-4}}$,
  derived as $4\sigma_\mathrm{SB}/c$ from the CODATA 2018 values rather than rounded
  independently (principle [9](../10-theory/index.md#p9-correctness)).

## Release evidence

Before a release or downstream migration, record the exact commands and results
used to qualify the branch. The normal local gate is:

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
```

For the full Phase-C style gate, use the repository script:

```bash
bash scripts/check.sh
```

The release note should name any intentionally skipped gate, the reason it was
skipped, and the narrower command that still covers the changed subsystem.

## Data packaging policy

Do not commit large external scientific data products to the repo or wheel. Local
mirrors such as PHOENIX/NewEra atmosphere products belong under gitignored data
or cache directories. Tests should use tiny synthetic fixtures or compact
metadata manifests unless a small redistributable upstream product is explicitly
approved and documented.

The full keep-a-changelog record remains in the repository's `CHANGELOG.md`.
