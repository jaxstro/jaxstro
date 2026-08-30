# Jaxstro Science and Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Produce a release-ready Jaxstro candidate with a reproducible delivery gate and a bounded, evidence-complete scientifically-qualified core, without claiming that all importable code or JAX backends are qualified.

**Architecture:** PUBLIC_MODULES becomes the single owner of the root-module inventory; the existing contract registry remains the sole owner of numerical-method claims; and scripts/check.sh is the exact release mirror. A lockfile-backed local MyST CLI replaces the global executable. Legacy numerical paths remain until a cycle-free internal owner move and pinned multi-consumer evidence authorize a hard cut.

**Tech Stack:** CPython 3.13, JAX/JAXlib, Equinox, Diffrax, Optimistix, SymPy, uv, Hatchling, pytest, Ruff, MyPy, GitHub Actions, Node 24, npm, MyST 1.10.1.

**Spec:** docs/superpowers/specs/2026-08-30-science-release-readiness-design.md

## Global Constraints

- Work from a clean commit; record exact commit, command output, platform, and x64 setting before changing a release or scientific qualification claim.
- The first supported release target is **CPython 3.13 on Ubuntu x86_64 CPU with JAX_ENABLE_X64=1**.
- Do not advertise GPU acceleration, operating-system independence, or untested Python versions as qualified support.
- Keep Jaxstro runtime dependencies unchanged; MyST is a development-only Node dependency and must not enter wheel metadata.
- jaxstro.units remains canonical. Do not start a quantity migration or remove a legacy numerical owner in this plan.
- Do not weaken numerical tolerances, omit a failing gate, regenerate measured evidence, tag, push, upload, or edit a downstream repository without separate authorization.
- The qualified core is exactly jaxstro.units, jaxstro.numerics.safeguarded_bracketed_root, jaxstro.numerics.implicit_bracketed_root, and jaxstro.numerics.universal_kepler_step; quadrature remains experimental.

---

## File structure and delivery boundaries

| File | Responsibility |
| --- | --- |
| src/jaxstro/_public.py | Canonical tuple of public root submodules. |
| src/jaxstro/contracts/profiles.py | Static selection of existing qualified contracts; no second schema. |
| scripts/check_distribution.py | Standard-library wheel/sdist contents, metadata, and clean-interpreter checks. |
| scripts/check.sh | Exact local release mirror, including Node docs tooling and both artifacts. |
| package.json, package-lock.json, .nvmrc | Reproducible development-only MyST executable on Node 24. |
| .github/workflows/full-gate.yml | Exact `release-mirror` plus separately scoped slow scientific-validation job. |
| docs/70-project/release/support.md | Explicit support and non-claim policy. |
| docs/60-validation/qualified-core.md | Rendered evidence boundary for the named scientific profile. |

Tasks 1-3 are one release-integrity delivery. Task 4 is a separately reviewable scientific-qualification delivery. Task 5 qualifies a clean candidate. Task 6 is an owner decision boundary, not a Jaxstro source change.

### Task 1: Canonical public surface and truthful support policy

**Files:**
- Create: src/jaxstro/_public.py
- Create: docs/70-project/release/support.md
- Create: tests/integration/test_public_surface.py
- Modify: src/jaxstro/__init__.py:14-44
- Modify: pyproject.toml:17-25
- Modify: README.md:5-60
- Modify: docs/70-project/direction/architecture.md:13-22
- Modify: docs/50-api/api.md:28-46
- Modify: docs/70-project/release/release.md:20-42
- Modify: docs/myst.yml:309-313
- Modify: docs/route-manifest.json
- Modify: tests/integration/test_architecture_docs.py:5-37
- Modify: tests/integration/test_api_reference.py:5-56
- Modify: tests/unit/test_contract_manifests.py:3-36

**Interfaces:**
- Consumes: current root exports and the direct-import list in tests/integration/test_api_reference.py.
- Produces: PUBLIC_MODULES: tuple[str, ...], used by the root namespace and later clean-install checks.

- [ ] **Step 1: Write the failing public-surface and support test**

Create tests/integration/test_public_surface.py:

~~~python
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tomllib

import jaxstro
from jaxstro._public import PUBLIC_MODULES

ROOT = Path(__file__).resolve().parents[2]


def test_root_exports_the_canonical_public_modules() -> None:
    expected = {
        "astrometry", "atmospheres", "constants", "contracts", "coords",
        "evidence", "geometry", "jaxconfig", "numerics", "params",
        "provenance", "quad", "quantity", "spatial", "spectra", "testing",
        "units",
    }
    assert set(PUBLIC_MODULES) == expected
    assert set(jaxstro.__all__) == {"DEFAULT_UNITS", *expected}
    for name in PUBLIC_MODULES:
        assert getattr(jaxstro, name) is importlib.import_module(f"jaxstro.{name}")


def test_release_support_page_names_only_qualified_support() -> None:
    text = (ROOT / "docs/70-project/release/support.md").read_text(encoding="utf-8")
    for phrase in (
        "CPython 3.13", "Ubuntu x86_64 CPU", "JAX_ENABLE_X64=1",
        "not a qualified support claim", "GPU", "experimental",
    ):
        assert phrase in text

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "Operating System :: OS Independent" not in project["project"]["classifiers"]
    assert "Programming Language :: Python :: 3" not in project["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.13" in project["project"]["classifiers"]

    routes = json.loads((ROOT / "docs/route-manifest.json").read_text(encoding="utf-8"))
    assert routes["70-project/release/support.md"] == "/support"
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_public_surface.py -q

Expected: FAIL because jaxstro._public and the support page do not exist.

- [ ] **Step 3: Implement the single root-surface owner**

Create src/jaxstro/_public.py:

~~~python
"""Canonical public root-module inventory for Jaxstro."""

PUBLIC_MODULES: tuple[str, ...] = (
    "astrometry", "atmospheres", "constants", "contracts", "coords",
    "evidence", "geometry", "jaxconfig", "numerics", "params",
    "provenance", "quad", "quantity", "spatial", "spectra", "testing",
    "units",
)
~~~

In src/jaxstro/__init__.py, import PUBLIC_MODULES, set __all__ = ["DEFAULT_UNITS", *PUBLIC_MODULES], and make __getattr__ accept only names in PUBLIC_MODULES. Do not add root callable re-exports.

Write docs/70-project/release/support.md with the following policy:

~~~markdown
## Qualified support for 0.1.0

The release qualification target is CPython 3.13 on Ubuntu x86_64 CPU with
JAX_ENABLE_X64=1. The release mirror exercises this configuration.

Installation requires Python >=3.13. GPU, TPU, macOS, Windows, and Python
versions other than CPython 3.13 are not a qualified support claim. JAX may
make an installation run on another backend; that is not evidence that Jaxstro
numerical contracts have been qualified there.

jaxstro.quad is public but experimental. Its method pages define accepted domains,
statuses, replay boundary, and non-claims; it is not in the qualified core.
~~~

Remove `Operating System :: OS Independent` and the generic `Programming Language :: Python :: 3` classifier from `pyproject.toml`; retain `Programming Language :: Python :: 3.13` and `Programming Language :: Python :: 3 :: Only`. Replace the README GPU-accelerated slogan with JAX-native, evidence-bounded, and differentiability-aware wording. Replace the badge alt text with `Requires Python >=3.13; CPython 3.13 release-qualified`, and replace the installation requirement with `Python >=3.13 is required; only CPython 3.13 on Ubuntu x86_64 CPU with JAX_ENABLE_X64=1 is release-qualified.` List all seven direct runtime dependencies; link the support page; add every `PUBLIC_MODULES` name to the architecture inventory; and add the support page to release navigation in `docs/myst.yml` and `docs/route-manifest.json` with the stable route `/support`.

In `tests/integration/test_architecture_docs.py`, `tests/integration/test_api_reference.py`, and `tests/unit/test_contract_manifests.py`, replace each handwritten public-module tuple with `from jaxstro._public import PUBLIC_MODULES` and derive `{f"jaxstro.{name}" for name in PUBLIC_MODULES}` where fully qualified contract paths are required. Remove the now-redundant one-off `jaxstro.jaxconfig` import assertion from `test_api_reference.py`. The prose remains authored documentation; the tests, import smoke, contract inventory, and root namespace now consume the same owner instead of hand-copied inventories.

- [ ] **Step 4: Run focused public-surface tests**

Run:

~~~bash
env -u VIRTUAL_ENV uv run --no-sync pytest \
  tests/integration/test_public_surface.py \
  tests/integration/test_api_reference.py \
  tests/integration/test_architecture_docs.py \
  tests/unit/test_contract_manifests.py -q
~~~

Expected: PASS. Every advertised root module imports and documentation, route manifest, and package metadata carry the same bounded support story.

- [ ] **Step 5: Commit the public-surface slice**

~~~bash
git add src/jaxstro/_public.py src/jaxstro/__init__.py pyproject.toml README.md docs/70-project/release/support.md docs/70-project/direction/architecture.md docs/50-api/api.md docs/70-project/release/release.md docs/myst.yml docs/route-manifest.json tests/integration/test_public_surface.py tests/integration/test_architecture_docs.py tests/integration/test_api_reference.py tests/unit/test_contract_manifests.py
git commit -m "docs: define qualified public support surface"
~~~

### Task 2: Check both distribution artifacts in clean interpreters

**Files:**
- Create: scripts/check_distribution.py
- Create: tests/unit/test_check_distribution.py
- Modify: pyproject.toml:1-3,75-89
- Modify: scripts/check.sh:36-43
- Modify: tests/integration/test_release_readiness.py:99-132

**Interfaces:**
- Consumes: jaxstro._public.PUBLIC_MODULES, project version, wheel/sdist paths, and a clean interpreter path.
- Produces: `python scripts/check_distribution.py --wheel PATH --sdist PATH --python PATH --provenance PATH`, which exits nonzero for malformed contents, missing build provenance, or failed public-module imports.

- [ ] **Step 1: Write failing artifact-checker tests**

Create tests/unit/test_check_distribution.py:

~~~python
from __future__ import annotations

import importlib.util
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_distribution", ROOT / "scripts/check_distribution.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_check_wheel_requires_typing_license_and_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "jaxstro-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("jaxstro/__init__.py", "")
        archive.writestr("jaxstro/py.typed", "")
        archive.writestr(
            "jaxstro-0.1.0.dist-info/METADATA",
            "Name: jaxstro\nVersion: 0.1.0\nRequires-Python: >=3.13\n"
            "License-Expression: Apache-2.0\n",
        )
        archive.writestr("jaxstro-0.1.0.dist-info/licenses/LICENSE", "Apache-2.0")
    checker.check_wheel(wheel, expected_version="0.1.0")


def test_check_sdist_requires_metadata_and_rejects_generated_tree(tmp_path: Path) -> None:
    sdist = tmp_path / "jaxstro-0.1.0.tar.gz"
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "PKG-INFO").write_text(
        "Name: jaxstro\nVersion: 0.1.0\nRequires-Python: >=3.13\n"
        "License-Expression: Apache-2.0\n",
        encoding="utf-8",
    )
    (payload / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
    (payload / "pyproject.toml").write_text("[project]", encoding="utf-8")
    package = payload / "src/jaxstro"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "py.typed").write_text("", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        for source in payload.rglob("*"):
            if source.is_file():
                archive.add(source, arcname=f"jaxstro-0.1.0/{source.relative_to(payload)}")
    checker.check_sdist(sdist, expected_version="0.1.0")


def test_check_sdist_rejects_internal_workspaces(tmp_path: Path) -> None:
    sdist = tmp_path / "jaxstro-0.1.0.tar.gz"
    payload = tmp_path / "AGENTS.md"
    payload.write_text("internal", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="jaxstro-0.1.0/AGENTS.md")
    with pytest.raises(ValueError, match="forbidden release member"):
        checker.check_sdist(sdist, expected_version="0.1.0")


def test_check_sdist_rejects_generated_documentation(tmp_path: Path) -> None:
    sdist = tmp_path / "jaxstro-0.1.0.tar.gz"
    payload = tmp_path / "index.html"
    payload.write_text("generated", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="jaxstro-0.1.0/docs/_build/html/index.html")
    with pytest.raises(ValueError, match="forbidden release member"):
        checker.check_sdist(sdist, expected_version="0.1.0")


def test_check_sdist_rejects_a_wrong_archive_root(tmp_path: Path) -> None:
    sdist = tmp_path / "jaxstro-0.1.0.tar.gz"
    payload = tmp_path / "PKG-INFO"
    payload.write_text("Name: jaxstro\n", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="wrong-root/PKG-INFO")
    with pytest.raises(ValueError, match="invalid sdist root"):
        checker.check_sdist(sdist, expected_version="0.1.0")


def test_check_provenance_requires_pinned_backend(tmp_path: Path) -> None:
    provenance = tmp_path / "build-provenance.txt"
    provenance.write_text(
        "uv 0.8.22\nPython 3.13.0\nhatchling==1.31.0\n", encoding="utf-8"
    )
    checker.check_provenance(provenance)
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_check_distribution.py -q

Expected: FAIL because scripts/check_distribution.py does not exist.

- [ ] **Step 3: Implement the artifact checker and wire both artifacts into the gate**

Implement `check_provenance(path: Path) -> None` to require the three expected build-provenance lines, and expose it through the required `--provenance` CLI argument. Implement the artifact interface as follows:

~~~python
def check_wheel(path: Path, *, expected_version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        _require(names, "jaxstro/__init__.py")
        _require(names, "jaxstro/py.typed")
        _require_metadata(names, archive.read, expected_version)
        _require_license(names)
        _reject_forbidden(names)


def check_sdist(path: Path, *, expected_version: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        expected_root = f"jaxstro-{expected_version}"
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            if member.isdir():
                continue
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or not member_path.parts
                or member_path.parts[0] != expected_root
                or not member.isfile()
            ):
                raise ValueError(f"invalid sdist root or member: {member.name}")
            members[member_path.relative_to(expected_root).as_posix()] = member
        names = set(members)
        _reject_forbidden(names)
        for name in ("src/jaxstro/__init__.py", "src/jaxstro/py.typed", "LICENSE", "pyproject.toml", "PKG-INFO"):
            _require(names, name)
        pkg_info = members["PKG-INFO"]
        _require_metadata(
            names,
            lambda _: archive.extractfile(pkg_info).read(),
            expected_version,
            metadata_name="PKG-INFO",
        )
~~~

Import `PurePosixPath` from `pathlib`. Make `_require_metadata` parse RFC 822 metadata with `email.message_from_bytes`, require `Name: jaxstro`, the expected `Version`, `Requires-Python: >=3.13`, and `License-Expression: Apache-2.0` in both `METADATA` and `PKG-INFO`. Before stripping paths, require every non-directory sdist member to be a regular file rooted exactly at `jaxstro-{expected_version}/`; reject absolute, parent-traversal, wrong-root, mixed-root, and symlink members with `ValueError(f"invalid sdist root or member: {member.name}")`. Reject a relative member beginning with `.github/`, `docs/audits/`, `docs/plans/`, `docs/superpowers/`, `docs/_build/`, `laboratory/`, `tests/`, `.mypy_cache/`, `.pytest_cache/`, or containing `/__pycache__/`; reject `AGENTS.md`, `CLAUDE.md`, or `STATUS.md` with `ValueError(f"forbidden release member: {name}")`.

In `pyproject.toml`, pin the build requirement to `hatchling==1.31.0`. Add `/docs/_build`, `/.mypy_cache`, `/.pytest_cache`, and `/**/__pycache__` to the sdist exclusion list. The release criterion is deterministic runtime resolution and a fixed PEP 517 backend version; do not claim byte-identical artifacts across arbitrary builders. Extend `test_sdist_excludes_internal_and_nonruntime_workspaces` to require those four excludes.

The --python path must execute this after each clean installation:

~~~python
import importlib
from jaxstro._public import PUBLIC_MODULES

for module in PUBLIC_MODULES:
    importlib.import_module(f"jaxstro.{module}")
~~~

Replace the wheel-only end of scripts/check.sh with a `mktemp -d` artifact directory, `uv build --python 3.13 -o "$ARTIFACT_DIR"`, checker invocation, and two clean CPython 3.13 environments: `uv venv --python 3.13 "$WHEEL_VENV"` installs the wheel and `uv venv --python 3.13 "$SDIST_VENV"` installs the sdist. Record `uv --version`, `python --version`, and the literal pinned `hatchling==1.31.0` in `$ARTIFACT_DIR/build-provenance.txt`; the checker must validate it is present. Trap cleanup must delete only these generated paths. Do not write `dist/` in the repository.

- [ ] **Step 4: Run artifact and release-readiness tests**

Run:

~~~bash
env -u VIRTUAL_ENV uv run --no-sync pytest \
  tests/unit/test_check_distribution.py \
  tests/integration/test_release_readiness.py -q
~~~

Expected: PASS. Extend the source-level release test to require the exact Hatchling pin, all generated-tree sdist exclusions, `uv build --python 3.13 -o`, `check_distribution.py`, `--wheel`, `--sdist`, build provenance, and two clean CPython 3.13 installs.

- [ ] **Step 5: Commit the artifact qualification slice**

~~~bash
git add pyproject.toml scripts/check_distribution.py scripts/check.sh tests/unit/test_check_distribution.py tests/integration/test_release_readiness.py
git commit -m "ci: qualify wheel and sdist release artifacts"
~~~

### Task 3: Lock the documentation CLI and make full CI exact

**Files:**
- Create: package.json
- Create: package-lock.json
- Create: .nvmrc
- Modify: scripts/check_docs.sh:27-38
- Modify: scripts/check.sh:24-29
- Modify: .gitignore:9-27
- Modify: .github/workflows/pages.yml:32-37
- Modify: .github/workflows/full-gate.yml
- Modify: tests/integration/test_docs_gate_wiring.py:16-77
- Modify: tests/integration/test_release_readiness.py:13-98

**Interfaces:**
- Consumes: Node 24 and package-lock.json.
- Produces: `npx --no-install myst` as the only MyST invocation; a `release-mirror` job that invokes `bash scripts/check.sh` on every main push, scheduled run, and manual run; and a separately named scheduled/manual `scientific-validation` job that retains slow validation coverage.

- [ ] **Step 1: Write failing tooling and CI parity tests**

Add these assertions to the existing integration tests:

~~~python
def test_docs_cli_is_lockfile_backed() -> None:
    package = (REPO_ROOT / "package.json").read_text(encoding="utf-8")
    docs_gate = (REPO_ROOT / "scripts/check_docs.sh").read_text(encoding="utf-8")
    assert '"mystmd": "1.10.1"' in package
    assert (REPO_ROOT / "package-lock.json").is_file()
    assert "npx --no-install myst build --html --ci --strict" in docs_gate
    assert "exec npx --no-install myst start" in docs_gate
    assert "\n  myst build" not in docs_gate
    assert "\n  exec myst start" not in docs_gate

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "node_modules/" in gitignore


def test_full_gate_runs_the_local_release_mirror_on_main_push() -> None:
    workflow = (REPO_ROOT / ".github/workflows/full-gate.yml").read_text(encoding="utf-8")
    assert "push:" in workflow
    assert "branches: [main]" in workflow
    assert "release-mirror:" in workflow
    assert "bash scripts/check.sh" in workflow
    assert "npm install --global" not in workflow
    assert 'JAX_ENABLE_X64: "1"' in workflow
    assert "timeout-minutes: 60" in workflow
    assert "scientific-validation:" in workflow
    assert "github.event_name != 'push'" in workflow
    assert "pytest tests/validation -q" in workflow
~~~

Replace the existing job-name/string slicing in `test_docs_gate_wiring.py` and `test_release_readiness.py`; it is coupled to the old `docs:` and `test-matrix:` topology. The new assertions must establish: (1) the local gate installs Node dependencies before `bash scripts/check_docs.sh`, (2) the `release-mirror` job invokes only `bash scripts/check.sh` for release work, and (3) the explicitly separate scientific-validation job runs `pytest tests/validation -q`. Do not retain tests that parse assumed job ordering.

In `test_pages_workflow_uses_the_verified_docs_gate_and_site_output`, replace the global-MyST assertion with `npm ci --ignore-scripts`, assert `npm install --global` is absent, and assert the Pages path filter contains `package.json`, `package-lock.json`, and `.nvmrc`. Keep the existing Node 24, pinned uv action, documentation-gate, and artifact-output assertions.

- [ ] **Step 2: Run focused tests to verify they fail**

Run: env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_docs_gate_wiring.py tests/integration/test_release_readiness.py -q

Expected: FAIL because MyST is globally installed and full CI does not mirror the local release gate on main pushes.

- [ ] **Step 3: Implement the deterministic documentation toolchain**

Create .nvmrc containing 24 and package.json:

~~~json
{
  "private": true,
  "engines": {"node": "24.x"},
  "devDependencies": {"mystmd": "1.10.1"},
  "scripts": {"docs:check": "bash scripts/check_docs.sh"}
}
~~~

Run `npm install --package-lock-only --ignore-scripts` once and commit the generated lockfile. Add `node_modules/` to `.gitignore`. Replace both direct MyST commands in `scripts/check_docs.sh` with `npx --no-install myst`. Run `npm ci --ignore-scripts` immediately before the docs gate in `scripts/check.sh`; Pages must do the same and remove the global install. Add `package.json`, `package-lock.json`, and `.nvmrc` to the Pages path trigger so a changed docs toolchain is deployed through the same gate.

Replace the existing release-approximation jobs in `full-gate.yml` with a `release-mirror` job triggered by push to `main`, the existing weekly schedule, and `workflow_dispatch`. Preserve the existing workflow environment exactly: `JAX_ENABLE_X64: "1"`, `XLA_PYTHON_CLIENT_PREALLOCATE: "false"`, `XLA_PYTHON_CLIENT_ALLOCATOR: platform`, and `OMP_NUM_THREADS: "1"`. The job checks out, sets Node 24 and uv/Python 3.13, declares `timeout-minutes: 60`, then contains exactly this release command:

~~~yaml
- name: Run the exact local release mirror
  run: bash scripts/check.sh
~~~

Add a separate `scientific-validation` job with `if: github.event_name != 'push'`, `timeout-minutes: 30`, the pinned `setup-uv` action and CPython 3.13 setup, `uv sync --locked --extra dev`, then `uv run --no-sync pytest tests/validation -q`. It runs on the weekly and manually dispatched workflow only; it is scientific coverage, not an assertion that the release mirror contains slow data-dependent validation. Keep `tests.yml` as a fast pull-request signal. Do not duplicate selected release-gate commands in the `release-mirror` job.

- [ ] **Step 4: Run focused tests and lock checks**

Run:

~~~bash
npm ci --ignore-scripts
env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_docs_gate_wiring.py tests/integration/test_release_readiness.py -q
env -u VIRTUAL_ENV uv lock --check
~~~

Expected: PASS. `npm ci` creates only ignored `node_modules`; `uv.lock` stays unchanged; structural tests no longer depend on deleted CI job names; and the slow scientific-validation lane remains explicit.

- [ ] **Step 5: Commit the reproducible-gate slice**

~~~bash
git add .gitignore package.json package-lock.json .nvmrc scripts/check_docs.sh scripts/check.sh .github/workflows/pages.yml .github/workflows/full-gate.yml tests/integration/test_docs_gate_wiring.py tests/integration/test_release_readiness.py
git commit -m "ci: mirror the locked local release gate on main"
~~~

### Task 4: Qualify a small scientific core from existing contracts

**Files:**
- Create: src/jaxstro/contracts/profiles.py
- Create: docs/60-validation/qualified-core.md
- Create: tests/validation/test_qualified_core.py
- Modify: src/jaxstro/contracts/__init__.py
- Modify: src/jaxstro/contracts/_core.py:1-82
- Modify: src/jaxstro/numerics/_contracts.py
- Modify: tests/validation/test_bracketed_root_algorithms.py:1-34
- Modify: docs/validation/contracts.json (generated contract inventory)
- Modify: docs/50-api/research-infrastructure/contracts.md (generated contract reference)
- Modify: docs/myst.yml:268-278
- Modify: docs/route-manifest.json
- Modify: docs/70-project/development/package-assessment-scorecard.md:28-45

**Interfaces:**
- Consumes: get_module_contract, get_callable_contract, MaturityLevel, EvidenceKind, and SupportLevel from jaxstro.contracts.
- Produces: QUALIFIED_CORE_V1: tuple[str, ...] and QUALIFIED_CORE_MODULES_V1: tuple[str, ...]; both contain registry import paths only.

- [ ] **Step 1: Write the failing qualified-core profile test**

Create tests/validation/test_qualified_core.py:

~~~python
from __future__ import annotations

from jaxstro.contracts import (
    EvidenceKind,
    ExecutionBoundary,
    MaturityLevel,
    SupportLevel,
    get_callable_contract,
    get_module_contract,
)
from jaxstro.contracts.profiles import QUALIFIED_CORE_MODULES_V1, QUALIFIED_CORE_V1


def test_qualified_core_v1_is_evidence_complete() -> None:
    assert QUALIFIED_CORE_MODULES_V1 == ("jaxstro.units",)
    assert QUALIFIED_CORE_V1 == (
        "jaxstro.numerics.safeguarded_bracketed_root",
        "jaxstro.numerics.implicit_bracketed_root",
        "jaxstro.numerics.universal_kepler_step",
    )
    module = get_module_contract(QUALIFIED_CORE_MODULES_V1[0])
    assert module.maturity is MaturityLevel.VALIDATED
    assert len(module.evidence) == 1
    assert module.evidence[0].kind is EvidenceKind.UNIT_TEST
    assert module.evidence[0].target == "tests/unit/test_units.py"
    assert module.non_ownership
    assert module.execution_boundary is ExecutionBoundary.STATIC
    assert "CGS" in module.dimensional_policy
    for path in QUALIFIED_CORE_V1:
        contract = get_callable_contract(path)
        assert contract.maturity is MaturityLevel.VALIDATED
        assert contract.evidence and contract.limitations and contract.boundaries
        assert any(item.kind is EvidenceKind.VALIDATION_TEST for item in contract.evidence)
    kepler = get_callable_contract("jaxstro.numerics.universal_kepler_step")
    transforms = {item.transform: item.support for item in kepler.transforms}
    assert transforms == {
        "jit": SupportLevel.SUPPORTED,
        "vmap": SupportLevel.SUPPORTED,
        "jvp": SupportLevel.CONDITIONAL,
        "vjp": SupportLevel.CONDITIONAL,
    }
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_qualified_core.py -q

Expected: FAIL because jaxstro.contracts.profiles does not exist. The live safeguarded-root contract also lacks a linked validation test, so preserve that criterion rather than lowering it.

- [ ] **Step 3: Add the profile and its rendered scientific boundary**

Create src/jaxstro/contracts/profiles.py:

~~~python
"""Named, evidence-complete selections from the public contract registry."""

QUALIFIED_CORE_MODULES_V1: tuple[str, ...] = ("jaxstro.units",)

QUALIFIED_CORE_V1: tuple[str, ...] = (
    "jaxstro.numerics.safeguarded_bracketed_root",
    "jaxstro.numerics.implicit_bracketed_root",
    "jaxstro.numerics.universal_kepler_step",
)
~~~

Export both names from `src/jaxstro/contracts/__init__.py`. Do not add a class, registry, decorator, or copied method descriptions.

Extend `module_contract` in `src/jaxstro/contracts/_core.py` with the optional keyword-only argument `evidence: tuple[EvidenceReference, ...] = ()`, pass it directly to `ModuleContract`, and import `EvidenceKind` and `EvidenceReference`. On the existing `units` record, provide exactly:

~~~python
evidence=(
    EvidenceReference(
        "units.scale-conversion-default",
        EvidenceKind.UNIT_TEST,
        "tests/unit/test_units.py",
        "CGS/default identity, named-system scales, and exact to/from-CGS conversion behavior.",
    ),
),
~~~

Before editing numerical-contract metadata, append these public-solver validation cases to `tests/validation/test_bracketed_root_algorithms.py`:

~~~python
@pytest.mark.parametrize(
    ("f", "lo", "hi", "expected"),
    [
        (lambda x: x - 0.3, 0.0, 1.0, 0.3),
        (lambda x: x**2 - 2.0, 0.0, 2.0, jnp.sqrt(2.0)),
    ],
)
def test_public_safeguarded_solver_converges_on_analytic_roots(f, lo, hi, expected) -> None:
    result = rootfinding.safeguarded_bracketed_root(
        f, lo, hi, max_steps=64, atol=1.0e-7, rtol=0.0
    )
    assert bool(result.bracketed)
    assert bool(result.converged)
    assert float(jnp.abs(result.root - expected)) <= 2.0e-6
    assert float(jnp.abs(result.residual)) <= 2.0e-6


def test_public_safeguarded_solver_reports_typed_missing_and_nonfinite_cases() -> None:
    missing = rootfinding.safeguarded_bracketed_root(
        lambda x: x**2 + 1.0, -1.0, 1.0, max_steps=8
    )
    nonfinite = rootfinding.safeguarded_bracketed_root(
        lambda x: jnp.where((x > 0.0) & (x < 2.0), jnp.nan, x - 1.0),
        0.0,
        2.0,
        max_steps=8,
    )
    assert missing.status == rootfinding.ROOT_STATUS_MISSING_BRACKET
    assert not bool(missing.converged)
    assert nonfinite.status == rootfinding.ROOT_STATUS_NONFINITE_EVALUATION
    assert not bool(nonfinite.converged)
~~~

Then, and only then, add an `EvidenceReference` named `root.safeguarded_bracketed_root.public-validation` with `EvidenceKind.VALIDATION_TEST`, target `tests/validation/test_bracketed_root_algorithms.py`, and the claim `"The public solver converges on analytical roots and returns typed missing-bracket and nonfinite results."` to the evidence tuple constructed only for `safeguarded_bracketed_root` in `src/jaxstro/numerics/_contracts.py`. Add its id to that callable's value-first boundary evidence ids. Do not change the algorithm, tolerance defaults, benchmark, or maturity enum.

Keep the conditional local to `_value_root_contract`; the exact shape is:

~~~python
public_validation = (
    EvidenceReference(
        "root.safeguarded_bracketed_root.public-validation",
        EvidenceKind.VALIDATION_TEST,
        "tests/validation/test_bracketed_root_algorithms.py",
        "The public solver converges on analytical roots and returns typed missing-bracket and nonfinite results.",
    ),
) if name == "safeguarded_bracketed_root" else ()
# use (evidence.id, *(item.id for item in public_validation)) for the boundary
# and (evidence, *public_validation, performance) for CallableContract.evidence
~~~

Write `docs/60-validation/qualified-core.md`, linking each import path to its API and exact evidence target: `tests/unit/test_units.py`, `tests/validation/test_bracketed_root_algorithms.py`, `tests/validation/test_implicit_root_gradients.py`, and `tests/validation/test_kepler_gradients.py`. State explicitly that `jaxstro.units` is a static-module exception: its qualified contract is ownership/non-ownership, CGS dimensional policy, and scale/conversion/default evidence; it makes no JAX-transform or numerical-failure claim. Add it to `docs/myst.yml` and `docs/route-manifest.json` with the stable route `/qualified-core`. Include this exact non-claim:

~~~markdown
This profile does not qualify every importable callable, a downstream physical
model, automatic differentiation across status or discrete-route changes, or
any GPU, TPU, macOS, Windows, or unlisted Python runtime.

jaxstro.quad.fixed and jaxstro.quad.integrate remain experimental and are not
promoted by this profile.
~~~

Add the page to validation navigation. In the scorecard, link foundation claims to this profile and retain the recorded unclassified-callable count as a limit.

- [ ] **Step 4: Regenerate only the deterministic contract artifacts**

Run:

~~~bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --emit
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --check
~~~

Expected: PASS. Only `docs/validation/contracts.json` and
`docs/50-api/research-infrastructure/contracts.md` change. Do not run a
numerical benchmark, regenerate a numerical validation artifact, or emit the
unrelated evidence index.

- [ ] **Step 5: Run profile and numerical evidence tests**

Run:

~~~bash
env -u VIRTUAL_ENV uv run --no-sync pytest \
  tests/unit/test_units.py \
  tests/validation/test_bracketed_root_algorithms.py \
  tests/validation/test_qualified_core.py \
  tests/validation/test_grad_checks.py \
  tests/validation/test_implicit_root_gradients.py \
  tests/validation/test_kepler_gradients.py \
  tests/unit/test_build_contract_registry_script.py \
  tests/integration/test_assessment_scorecard.py -q
~~~

Expected: PASS. This validates existing independent numerical evidence; it does not establish a downstream physical-model claim.

- [ ] **Step 6: Commit the scientific-profile slice**

~~~bash
git add src/jaxstro/contracts/profiles.py src/jaxstro/contracts/__init__.py src/jaxstro/contracts/_core.py src/jaxstro/numerics/_contracts.py docs/50-api/research-infrastructure/contracts.md docs/60-validation/qualified-core.md docs/myst.yml docs/route-manifest.json docs/70-project/development/package-assessment-scorecard.md docs/validation/contracts.json tests/validation/test_bracketed_root_algorithms.py tests/validation/test_qualified_core.py tests/unit/test_build_contract_registry_script.py
git commit -m "docs: define the qualified scientific core"
~~~

### Task 5: Run frozen release qualification from a clean commit

**Files:**
- Modify: docs/70-project/release/checklist.md:31-76
- Modify: CHANGELOG.md:1-20
- Modify: CITATION.cff:1-30 only if a separately approved release-version decision changes it.
- Modify: tests/integration/test_release_readiness.py:86-105

**Interfaces:**
- Consumes: a clean commit containing Tasks 1-4 and exact scripts/check.sh behavior.
- Produces: commit-specific local and CI qualification evidence for candidate SHA `X`, plus a later documentation-only attestation commit `Y`; never a tag, archive, GitHub release, or package-index upload.

- [ ] **Step 1: Freeze the candidate before running it**

Run:

~~~bash
git status --short
git rev-parse HEAD
git diff --check
~~~

Expected: empty status, one immutable SHA, and no whitespace errors. Stop if the tree is dirty.

- [ ] **Step 2: Run the exact release mirror once**

Run: JAX_ENABLE_X64=1 bash scripts/check.sh

Expected: lock, Python dependencies, lint, format, MyPy, four generated-registry checks, locked docs build and rendered-site checks, non-slow tests, ML integration, and wheel/sdist inspection and imports all exit 0.

- [ ] **Step 3: Verify candidate X in the matching CI lanes after separately authorized push**

After an explicitly authorized push of `X` to `main`, verify that the `release-mirror` Actions run was triggered from and reports exactly `X`, and that it ran `bash scripts/check.sh`. If a slow scientific-validation result is needed for a broader numerical claim, manually dispatch its separately named workflow lane on `X` and record it as scientific evidence, not as an unstated release-mirror step. A Pages build, PR fast suite, another revision scheduled run, or the later attestation commit is insufficient.

- [ ] **Step 4: Record observed evidence only as an attestation of X**

Fill local rows in `docs/70-project/release/checklist.md` with the literal `candidate_sha: X`, date, Python/JAX platform, command, counts, elapsed time, and the matching release-mirror run identifier. Add this policy directly above the row:

~~~markdown
This record attests only to `candidate_sha`. The later commit that records this
evidence is not a qualified release candidate; any future tag or publication
must target `candidate_sha` after separate authorization.
~~~

If any command fails, record it and stop; write a regression test before a minimal root-cause repair. Do not check a remote-action row.

- [ ] **Step 5: Ratchet the attestation policy and commit Y**

Add a `test_release_readiness.py` assertion that `checklist.md` contains both `candidate_sha:` and the exact sentence `The later commit that records this evidence is not a qualified release candidate`. This tests the provenance policy without pretending a source-level test can qualify an arbitrary SHA.

~~~bash
git add docs/70-project/release/checklist.md CHANGELOG.md tests/integration/test_release_readiness.py
git commit -m "docs: record release candidate qualification"
~~~

This creates attestation commit `Y`. Do not tag `Y`, call it qualified, push it, create a GitHub/Zenodo release, or upload an artifact without the separate authorization in `docs/70-project/release/checklist.md`. A future authorized tag must resolve to `X`, not `Y`.

### Task 6: Migrate Progenax to canonical Jaxstro quadrature owners

**Approval boundary:** This task changes the Progenax repository. Obtain explicit authorization to edit that repository before beginning it; this plan change does not authorize the edit.

**Files:**
- Create: /Users/anna/projects/jaxstro-dev/progenax/tests/unit/test_jaxstro_quad_ownership.py
- Modify: /Users/anna/projects/jaxstro-dev/progenax/src/progenax/numerics.py:1-30
- Modify: /Users/anna/projects/jaxstro-dev/progenax/src/experimental/gravoturb/inference/projected_logp.py:25-30
- Modify: /Users/anna/projects/jaxstro-dev/progenax/src/experimental/gravoturb/theory/log_correlations.py:21-27,63-68,102-107
- Modify: /Users/anna/projects/jaxstro-dev/progenax/src/experimental/gravoturb/theory/counts_in_cells.py:28-34
- Modify: /Users/anna/projects/jaxstro-dev/progenax/tests/experimental/unit/test_log_correlations.py:72-100,159-170
- Modify: /Users/anna/projects/jaxstro-dev/progenax/tests/experimental/unit/test_projected_logp.py:146-155
- Modify: /Users/anna/projects/jaxstro-dev/progenax/uv.lock

**Interfaces:**
- Consumes: jaxstro.quad.cumulative_trapezoid(y, x=None, *, dx=1.0, axis=-1), jaxstro.quad.gauss_hermite_nodes(n), and jaxstro.quad.hermite_coefficients(map_fn, n_max, n_quad=128).
- Produces: the unchanged public Progenax name `progenax.numerics.cumulative_trapz`, now bound directly to `jaxstro.quad.cumulative_trapezoid`; all Gravoturb source and tests use canonical `jaxstro.quad` imports; and a lockfile whose editable Jaxstro record includes `sympy` and passes `uv lock --check`.

- [ ] **Step 1: Write the failing ownership ratchet**

Create /Users/anna/projects/jaxstro-dev/progenax/tests/unit/test_jaxstro_quad_ownership.py:

~~~python
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODULES = frozenset((
    "jaxstro.numerics.integration",
    "jaxstro.numerics.quadrature",
))


def _is_legacy_module(module: str) -> bool:
    return any(
        module == legacy or module.startswith(f"{legacy}.")
        for legacy in LEGACY_MODULES
    )


def _legacy_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.name for alias in node.names if _is_legacy_module(alias.name)
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if _is_legacy_module(node.module):
                modules.add(node.module)
            modules.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if _is_legacy_module(f"{node.module}.{alias.name}")
            )
    return modules


def test_progenax_uses_only_canonical_jaxstro_quad_imports() -> None:
    python_files = tuple((ROOT / "src").rglob("*.py")) + tuple(
        (ROOT / "tests").rglob("*.py")
    )
    offenders: dict[str, list[str]] = {}
    for path in python_files:
        legacy = _legacy_imports(path)
        if legacy:
            offenders[path.relative_to(ROOT).as_posix()] = sorted(legacy)
    assert offenders == {}
~~~

- [ ] **Step 2: Run the ownership ratchet to verify it fails**

Run:

~~~bash
cd /Users/anna/projects/jaxstro-dev/progenax
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_jaxstro_quad_ownership.py -q
~~~

Expected: FAIL and list the four source files plus the two experimental tests that still import legacy Jaxstro paths.

- [ ] **Step 3: Make the exact canonical-import substitutions and refresh the owned lock**

In src/progenax/numerics.py, preserve the Progenax API name with:

~~~python
from jaxstro.quad import cumulative_trapezoid as cumulative_trapz
~~~

Update its docstring to identify jaxstro.quad.cumulative_trapezoid as the owner
while retaining the current dx-outside and keyword-compatibility statement.

Use these three replacements without wrappers or changed arguments:

~~~python
from jaxstro.quad import gauss_hermite_nodes, hermite_coefficients

from jaxstro.quad import hermite_coefficients as _quadrature_hermite_coefficients
~~~

The two-symbol import belongs in projected_logp.py; the aliased import belongs
in both log_correlations.py and counts_in_cells.py. Replace every explanatory
reference to jaxstro.numerics.quadrature in log_correlations.py with
jaxstro.quad. Change the four direct imports in test_log_correlations.py and
the one in test_projected_logp.py to from jaxstro.quad import hermite_coefficients.
Do not change coefficient order, n_max, n_quad, tolerances, test fixtures, or
the public Progenax function name.

The live Progenax lock currently fails `uv lock --check` because the editable
Jaxstro record omits required `sympy`. After the import substitutions, run:

~~~bash
cd /Users/anna/projects/jaxstro-dev/progenax
env -u VIRTUAL_ENV uv lock
env -u VIRTUAL_ENV uv lock --check
rg -n 'name = "jaxstro"|name = "sympy"|\{ name = "sympy" \}' uv.lock
~~~

Expected: PASS. Inspect the lock diff before proceeding: the Jaxstro package
record and its `requires-dist` list name `sympy>=1.12`, and the resolved `sympy`
package is present. Do not hand-edit `uv.lock` or accept unrelated resolver
changes without identifying their cause.

- [ ] **Step 4: Run focused parity, transform, and consumer-workflow tests**

Run:

~~~bash
cd /Users/anna/projects/jaxstro-dev/progenax
env -u VIRTUAL_ENV uv sync --locked --extra dev --extra experimental
env -u VIRTUAL_ENV uv run --no-sync pytest -q \
  tests/unit/test_jaxstro_quad_ownership.py \
  tests/unit/test_numerics.py \
  tests/experimental/unit/test_log_correlations.py \
  tests/experimental/unit/test_density_hermite.py \
  tests/experimental/unit/test_projected_logp.py \
  tests/experimental/unit/test_counts_in_cells.py \
  tests/experimental/unit/test_grads_cic.py
env -u VIRTUAL_ENV uv run --no-sync ruff check \
  src/progenax/numerics.py \
  src/experimental/gravoturb/inference/projected_logp.py \
  src/experimental/gravoturb/theory/log_correlations.py \
  src/experimental/gravoturb/theory/counts_in_cells.py \
  tests/unit/test_jaxstro_quad_ownership.py \
  tests/experimental/unit/test_log_correlations.py \
  tests/experimental/unit/test_projected_logp.py
env -u VIRTUAL_ENV uv run --no-sync mypy \
  src/progenax \
  src/experimental/gravoturb/inference/projected_logp.py \
  src/experimental/gravoturb/theory/log_correlations.py \
  src/experimental/gravoturb/theory/counts_in_cells.py
~~~

Expected: PASS. The selected tests retain cumulative-trapezoid values and
gradients, Hermite analytic identities, Gravoturb coefficient gradients, and
the projected-bandpower workflow while exercising canonical imports. The locked
environment is now evidence for the live sibling dependency graph, not merely
the import substitution.

- [ ] **Step 5: Prove that the source migration is complete and commit it separately**

Run:

~~~bash
cd /Users/anna/projects/jaxstro-dev/progenax
rg -n "jaxstro\.numerics\.(integration|quadrature)" src tests \
  --glob '*.py' \
  --glob '!test_jaxstro_quad_ownership.py'
~~~

Expected: no output. Then commit only the source, test, and documentation
comments listed in this task:

~~~bash
git add uv.lock src/progenax/numerics.py src/experimental/gravoturb/inference/projected_logp.py src/experimental/gravoturb/theory/log_correlations.py src/experimental/gravoturb/theory/counts_in_cells.py tests/unit/test_jaxstro_quad_ownership.py tests/experimental/unit/test_log_correlations.py tests/experimental/unit/test_projected_logp.py
git commit -m "refactor: use canonical jaxstro quadrature owners"
~~~

Keep Jaxstro legacy modules intact. `jaxstro.quad` currently imports Hermite
helpers from `jaxstro.numerics.quadrature`, so consumer migration cannot
authorize deletion. A separate future deletion plan must first move those
helpers to a cycle-free canonical implementation owner and prove Jaxstro's own
legacy/canonical identity, array-value, JIT, and AD seams; only then can clean,
immutable revisions of at least two independent consumers establish the external
deletion condition. Quantity adoption remains subject to the existing parity,
serialization, transform, warm-cost, and migration-diff gate.

## Self-review

**Spec coverage:** Task 1 aligns root API, support prose, route manifest, architecture test, and package metadata. Task 2 validates both artifact metadata and payloads with a fixed PEP 517 backend, while explicitly not claiming bit-identical builds. Task 3 locks MyST, ignores its local install tree, makes CI release parity exact, and preserves slow scientific validation as a separate lane. Task 4 attaches only public-surface evidence to the bounded core and routes its rendered boundary. Task 5 distinguishes qualified candidate `X` from attestation `Y`. Task 6 migrates every observed Progenax legacy integration/quadrature import and repairs its sibling lock without deleting Jaxstro compatibility modules. No new method family or framework is introduced.

**Placeholder scan:** Every introduced file, route, replacement symbol, test target, backend version, lock action, and CI lane has an exact owner and command. A later Jaxstro deletion remains deliberately outside this plan until a cycle-free internal-owner move passes identity/value/JIT/AD tests and two independent pinned consumers are qualified.

**Type consistency:** `PUBLIC_MODULES` is a tuple of module-name strings used by the root package, architecture test, and installed-artifact command. `QUALIFIED_CORE_V1` and `QUALIFIED_CORE_MODULES_V1` are contract import-path tuples used only with `get_callable_contract` and `get_module_contract`, respectively. `module_contract(..., evidence=())` retains existing module records while allowing only the named units evidence addition.

## Execution handoff

Plan complete and saved to docs/superpowers/plans/2026-08-30-science-release-readiness.md. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
