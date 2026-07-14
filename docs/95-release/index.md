---
title: Release notes
description: >-
  Changelog, versioning, and deprecation policy for jaxstro — including the
  dx-outside cumulative_trapz reconciliation and the A_RAD correction.
---

This section summarizes the numerical changes that downstream packages must
understand. The canonical keep-a-changelog record remains in the repository's
`CHANGELOG.md`; the [](./checklist.md) turns publication into explicit,
evidence-carrying local and remote gates.

Two reconciliations from the 0.1.0 line are worth flagging here because they can
shift downstream numbers:

- **`cumulative_trapz` standardized to dx-outside.** Former dx-inside call sites may
  drift by ~1 ulp; this is the expected rounding difference, not a regression — see
  [](../20-methods/approximation-integration/cumulative-trapz.md).
- **`A_RAD` corrected** to $7.565733250\times10^{-15}\ \erg\,\mathrm{cm^{-3}\,K^{-4}}$,
  derived as $4\sigma_\mathrm{SB}/c$ from the CODATA 2018 values rather than rounded
  independently (principle [9](../20-methods/methods.md#p9-correctness)).

## Release evidence

Before a release or downstream migration, record the exact commands and results
used to qualify the branch. The normal local gate is:

```bash
bash scripts/check.sh
```

For a Pages candidate, also verify the production base path and rendered DOM:

```bash
BASE_URL=/jaxstro bash scripts/check_docs.sh
```

The release note should name any intentionally skipped gate, the reason it was
skipped, and the narrower command that still covers the changed subsystem.

## Data packaging policy

Do not commit large external scientific data products to the repo or wheel. Local
mirrors such as PHOENIX/NewEra atmosphere products belong under gitignored data
or cache directories. Tests should use tiny synthetic fixtures or compact
metadata manifests unless a small redistributable upstream product is explicitly
approved and documented.

No checklist row authorizes a remote action. Pushes, Pages configuration,
deployment, tags, archives, and package-index uploads remain separate approvals.
