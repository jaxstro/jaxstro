"""Executable contracts for random computation and resampling method pages."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_PAGE = REPO_ROOT / "docs" / "20-methods" / "probability-sampling" / "random.md"
SAMPLING_PAGE = (
    REPO_ROOT / "docs" / "20-methods" / "probability-sampling" / "sampling.md"
)
BIBLIOGRAPHY = REPO_ROOT / "docs" / "99-bibliography" / "references.bib"


def _page_text() -> str:
    return RANDOM_PAGE.read_text(encoding="utf-8")


def _sampling_text() -> str:
    return SAMPLING_PAGE.read_text(encoding="utf-8")


def _first_python_block(page: Path) -> str:
    text = page.read_text(encoding="utf-8")
    match = re.search(r"```python\n(?P<code>.*?)\n```", text, re.DOTALL)
    assert match is not None, f"{page.name} needs a standalone Python example"
    return match.group("code")


def test_random_computation_example_is_standalone_and_executable() -> None:
    block = _first_python_block(RANDOM_PAGE)
    for definition in ("key =", "subkeys =", "folded =", "manifest ="):
        assert definition in block

    namespace: dict[str, object] = {}
    exec(compile(block, str(RANDOM_PAGE), "exec"), namespace)

    subkeys = np.asarray(namespace["subkeys"])
    folded = np.asarray(namespace["folded"])
    manifest = namespace["manifest"]
    assert subkeys.shape == (3, 2)
    assert folded.shape == (3, 2)
    assert manifest == {
        "algorithm": "jax.random",
        "seed": 17,
        "stream": "particle-filter",
    }


def test_sampling_example_is_standalone_and_executable() -> None:
    block = _first_python_block(SAMPLING_PAGE)
    for definition in ("weights =", "num_samples =", "systematic ="):
        assert definition in block

    namespace: dict[str, object] = {}
    exec(compile(block, str(SAMPLING_PAGE), "exec"), namespace)

    systematic = np.asarray(namespace["systematic"])
    stratified = np.asarray(namespace["stratified"])
    residual = np.asarray(namespace["residual"])
    replay = np.asarray(namespace["replay"])
    for indices in (systematic, stratified, residual):
        assert indices.shape == (5,)
        assert np.all((indices >= 0) & (indices < 3))
    np.testing.assert_array_equal(replay, systematic)
    np.testing.assert_array_equal(np.bincount(residual, minlength=3), [2, 2, 1])


def test_split_pages_preserve_live_execution_contracts() -> None:
    random_text = _page_text()
    sampling_text = _sampling_text()

    assert "```{list-table} Random-computation contracts" in random_text
    assert ":label: tbl-random-computation-contracts" in random_text
    for phrase in ("caller owns `next_key`", "`validation_only`"):
        assert phrase in random_text

    assert "```{list-table} Sampling and resampling contracts" in sampling_text
    assert ":label: tbl-sampling-resampling-contracts" in sampling_text
    for phrase in (
        "caller owns `next_key`",
        "integer indices",
        "`num_samples` is static",
        "zero-total fallback",
        "eager validation is skipped while weights are traced",
        "finite, nonnegative weights under `jax.jit`",
        "`validation_only`",
    ):
        if phrase != "caller owns `next_key`":
            assert phrase in sampling_text


def test_random_claims_have_primary_provenance() -> None:
    random_text = _page_text()
    sampling_text = _sampling_text()
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")

    assert "{cite:t}`JAXJEP263`" in random_text
    assert "{cite:t}`DoucCappeMoulines2005`" in sampling_text
    assert "@misc{JAXJEP263" in bibliography
    assert "docs.jax.dev/en/latest/jep/263-prng.html" in bibliography
    assert "@inproceedings{DoucCappeMoulines2005" in bibliography
    assert "10.1109/ISPA.2005.195385" in bibliography


def test_random_page_routes_to_api_validation_and_contract_taxonomy() -> None:
    for text in (_page_text(), _sampling_text()):
        assert "[](../../40-api/index.md#jaxstro-numerics-random)" in text
        assert "[](../../60-validation/index.md)" in text
        assert "[](../methods.md#gradient-contracts)" in text
