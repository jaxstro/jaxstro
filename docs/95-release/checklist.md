---
title: Release checklist
description: >-
  Evidence and authorization gates for publishing jaxstro documentation,
  archives, and Python distributions without making irreversible claims early.
---

# Release checklist

This checklist separates repeatable local qualification from irreversible remote
actions. A checked local row does not authorize a push, tag, GitHub setting
change, Zenodo deposit, or PyPI upload.

## 1. Permanent identity — hard stop

- [x] Anna settled **jaxstro vs jaxstro-core** on 2026-07-12: the permanent
  distribution and import name is **`jaxstro`**; `jaxstro-core` is rejected.
- [x] The chosen distribution name agrees across `pyproject.toml`, wheel/sdist
  names, installation documentation, release notes, and downstream constraints.
- [x] No test upload or production upload occurred before that decision.
- [ ] PyPI project ownership and publication metadata are verified immediately
  before the later, separately authorized upload.

**Stop gate:** every PyPI operation requires Anna's separate, explicit
authorization. PyPI names and published artifacts cannot be treated like a local
draft. The progenax publication path remains blocked until jaxstro's first
compatible release is published through that later authorized process.

## 2. Version, history, and scientific evidence

- [ ] The version in `pyproject.toml`, `CITATION.cff`, and the release heading is
  identical and no longer marked unreleased.
- [ ] `CHANGELOG.md` describes public APIs, numerical corrections, known policy
  gaps, and downstream migration requirements without overstating validation.
- [ ] Provenance pages are fresh:

  ```bash
  uv run --no-sync python scripts/build_provenance_registry.py --check
  ```

- [ ] The exact release commit and complete local evidence are recorded.

## 3. Local quality gates

- [ ] The release mirror passes with its exact counts and timing recorded:

  ```bash
  bash scripts/check.sh
  ```

- [ ] The base-path documentation build passes before Pages publication:

  ```bash
  BASE_URL=/jaxstro bash scripts/check_docs.sh
  ```

- [ ] The public-repository hygiene scan has no unexplained machine-local paths,
  credentials, private notes, raw datasets, build trees, or bundled PDFs.
- [ ] Supported Python versions pass the scheduled/manual GitHub Actions full
  gate; the required pull-request aggregate check is green.

## 4. Distribution contents

- [ ] Build both artifacts from a clean checkout:

  ```bash
  uv build
  ```

- [ ] Inspect the wheel and sdist contents. They include the license, typing
  marker, and intended package sources, and exclude tests, local atmosphere data,
  docs build output, caches, laboratory products, agent state, internal audits,
  and planning files. The exclusions are ratcheted in `pyproject.toml`.
- [ ] Install the wheel in a clean environment and import every advertised
  top-level module.
- [ ] Validate distribution metadata and the long description with an
  appropriate package-index checker before any upload.

## 5. Documentation and Pages

- [x] `.github/workflows/pages.yml` uses the full docs gate, the `/jaxstro` base
  path, requires a deployable root `index.html`, and uploads `docs/_build/html`.
- [x] Anna explicitly authorized the push containing the Pages workflow on
  2026-07-12.
- [x] GitHub Pages source is set once to **GitHub Actions**, using the reviewed
  repository-settings or API operation.
- [ ] The deployed site at `https://jaxstro.github.io/jaxstro/` is checked in the
  rendered DOM for navigation, assets, equations, figures, and internal links.

## 6. Citation, archive, and contributor surface

- [ ] `CITATION.cff` validates and points to the permanent repository and chosen
  release version.
- [ ] `CONTRIBUTING.md` matches the current setup, scientific evidence rules, and
  verification commands.
- [ ] The release tag is created only from the qualified commit.
- [ ] Zenodo integration is configured deliberately; the resulting DOI is added
  to citation metadata and release notes only after the archive exists.

## 7. Downstream ecosystem

- [ ] Fluxax, Stellax, Startrax, Progenax, Gravax, Radax, and Hydrax consumers are
  searched for actual version/import constraints; missing local repositories are
  recorded rather than assumed compatible.
- [ ] Required downstream floors are updated only after the jaxstro release is
  available from the selected channel.
- [ ] Progenax's blocked PyPI item is resumed only after its declared jaxstro
  dependency can be installed in a clean environment.

## 8. Remote-action authorization

Request and record separate explicit authorization for each operation:

1. pushing reviewed commits to `origin/main`;
2. configuring GitHub Pages and verifying deployment;
3. creating and pushing a version tag;
4. creating a GitHub/Zenodo release; and
5. uploading to TestPyPI or PyPI.

If any remote result differs from the locally verified artifact, stop and review
the discrepancy. Do not repair a deployment by weakening the local gates.
