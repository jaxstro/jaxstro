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

:::{warning} Planned — not yet written
This section is a stub. The full keep-a-changelog record is in the repository's
`CHANGELOG.md`.
:::
