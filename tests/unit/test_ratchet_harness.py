"""Mechanism self-tests for the canonical ratchet harness (jaxstro.testing.ratchet).

Each generic primitive is exercised against a tiny, self-contained fixture written to
``tmp_path`` (or a small committed fixture under ``_ratchet_fixtures/`` for the collect-only
subprocess check). These tests are content-free in the same sense the harness is: they
assert the MECHANISM, not any ecosystem-package symbol. Pure stdlib, ms-scale — NO jax, NO
real-suite scan — so they live in the fast inner loop (tests/unit).
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from jaxstro.testing import ratchet

# Bind the non-``test_``-prefixed primitives as module-level names for readability. The
# ``test_body_has_assert`` primitive is deliberately referenced via ``ratchet.`` so pytest
# does not mis-collect the harness function itself as a test (it takes a ``node_id`` arg).
assert_no_stale = ratchet.assert_no_stale
assert_partition = ratchet.assert_partition
has_nearby_citation = ratchet.has_nearby_citation
resolve_node_ids = ratchet.resolve_node_ids
scan_module_numeric_literals = ratchet.scan_module_numeric_literals

# ======================================================================================
# assert_partition
# ======================================================================================


def test_assert_partition_passes_for_clean_partition():
    """Two disjoint buckets whose union equals all_symbols -> no error."""
    all_symbols = {"a", "b", "c", "d"}
    bucket1 = {"a": "x", "b": "y"}
    bucket2 = {"c", "d"}
    # Should not raise.
    assert_partition(all_symbols, bucket1, bucket2, label="demo")


def test_assert_partition_raises_on_missing_symbol():
    """A symbol in all_symbols but in no bucket -> AssertionError."""
    all_symbols = {"a", "b", "c"}
    bucket1 = {"a", "b"}
    with pytest.raises(AssertionError, match="not categorized"):
        assert_partition(all_symbols, bucket1, label="demo")


def test_assert_partition_raises_on_overlap():
    """A symbol in two buckets -> AssertionError (ambiguous status)."""
    all_symbols = {"a", "b", "c"}
    bucket1 = {"a", "b"}
    bucket2 = {"b", "c"}
    with pytest.raises(AssertionError, match="BOTH|overlap"):
        assert_partition(all_symbols, bucket1, bucket2, label="demo")


def test_assert_partition_raises_on_stale_entry():
    """A bucket entry not in all_symbols (stale) -> AssertionError."""
    all_symbols = {"a", "b"}
    bucket1 = {"a", "b", "zombie"}
    with pytest.raises(AssertionError, match="stale|no longer"):
        assert_partition(all_symbols, bucket1, label="demo")


# ======================================================================================
# assert_no_stale
# ======================================================================================


def test_assert_no_stale_passes_when_clean():
    mapping = {"a": "test_a", "b": "test_b"}
    universe = {"a", "b", "c"}
    assert_no_stale(mapping, universe, label="demo")  # no raise


def test_assert_no_stale_raises_on_stale_key():
    mapping = {"a": "test_a", "ghost": "test_ghost"}
    universe = {"a", "b"}
    with pytest.raises(AssertionError, match="stale|no longer|ghost"):
        assert_no_stale(mapping, universe, label="demo")


# ======================================================================================
# test_body_has_assert
# ======================================================================================


def _write_test_file(tmp_path: Path, body: str) -> Path:
    src = textwrap.dedent(body)
    p = tmp_path / "fixture_test_module.py"
    p.write_text(src)
    return p


def test_body_has_assert_false_for_no_assert(tmp_path):
    """A test fn whose body only calls a function (no assert) -> False."""
    p = _write_test_file(
        tmp_path,
        """
        def helper():
            return 1

        def test_x():
            helper()
        """,
    )
    assert ratchet.test_body_has_assert(f"{p}::test_x") is False


def test_body_has_assert_true_for_bare_assert(tmp_path):
    """A bare ``assert`` statement in the body -> True."""
    p = _write_test_file(
        tmp_path,
        """
        def test_y():
            x = 1 + 1
            assert x == 2
        """,
    )
    assert ratchet.test_body_has_assert(f"{p}::test_y") is True


def test_body_has_assert_true_for_assert_helper(tmp_path):
    """A recognized assert-helper call (np.testing.assert_allclose) -> True."""
    p = _write_test_file(
        tmp_path,
        """
        import numpy as np

        def test_z():
            np.testing.assert_allclose(1.0, 1.0)
        """,
    )
    assert ratchet.test_body_has_assert(f"{p}::test_z") is True


def test_body_has_assert_handles_class_method(tmp_path):
    """A file::Class::method node id resolves to the method body."""
    p = _write_test_file(
        tmp_path,
        """
        class TestThing:
            def test_method(self):
                assert True
        """,
    )
    assert ratchet.test_body_has_assert(f"{p}::TestThing::test_method") is True


# ======================================================================================
# scan_module_numeric_literals
# ======================================================================================


def _write_module(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "fixture_module.py"
    p.write_text(textwrap.dedent(body))
    return p


def test_scan_yields_only_citable_literals(tmp_path):
    """A citable 2.35 is yielded; trivial 1.0/0.5 and a small int 3 are skipped."""
    p = _write_module(
        tmp_path,
        """
        ALPHA = 2.35
        SCALE = 1.0
        HALF = 0.5
        INDEX = 3
        """,
    )
    found = dict(
        scan_module_numeric_literals(
            str(p), trivial={0.0, 1.0, 2.0, 0.5, -1.0, -0.5}, small_int_max=12
        )
    )
    assert set(found) == {2.35}


def test_scan_respects_small_int_max(tmp_path):
    """An int 20 is skipped when small_int_max>=20, yielded when small_int_max<20."""
    p = _write_module(
        tmp_path,
        """
        BIG_INDEX = 20
        """,
    )
    skipped = list(
        scan_module_numeric_literals(str(p), trivial=set(), small_int_max=50)
    )
    assert skipped == []
    yielded = dict(
        scan_module_numeric_literals(str(p), trivial=set(), small_int_max=12)
    )
    assert set(yielded) == {20.0}


def test_scan_skips_bool_and_tiny(tmp_path):
    """Booleans and |v| < 1e-9 are never yielded."""
    p = _write_module(
        tmp_path,
        """
        FLAG = True
        TINY = 1e-12
        REAL = 4.74
        """,
    )
    found = dict(scan_module_numeric_literals(str(p), trivial=set(), small_int_max=12))
    assert set(found) == {4.74}


def test_scan_yields_signed_negative_literal(tmp_path):
    """A negative literal (AST UnaryOp(USub, Constant)) is yielded with its sign.

    The signed value -2.35 must survive (NOT an unsigned 2.35 duplicate): Task-3 provenance
    matching compares literal value-text, so the sign is load-bearing.
    """
    p = _write_module(
        tmp_path,
        """
        coeff = -2.35  # cite
        """,
    )
    pairs = list(scan_module_numeric_literals(str(p), trivial=set(), small_int_max=12))
    values = [v for v, _ in pairs]
    assert values == [-2.35]  # signed, exactly one entry (no unsigned duplicate)


# ======================================================================================
# has_nearby_citation
# ======================================================================================


def _literal_line(path: Path, text: str) -> int:
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if text in line:
            return i
    raise AssertionError(f"literal {text!r} not found in {path}")


def test_has_nearby_citation_true_for_comment_above(tmp_path):
    """A citation comment one line above the literal -> True."""
    p = _write_module(
        tmp_path,
        """
        # Salpeter (1955) ApJ 121, 161
        ALPHA = 2.35
        """,
    )
    lineno = _literal_line(p, "2.35")
    assert has_nearby_citation(str(p), lineno) is True


def test_has_nearby_citation_false_without_comment(tmp_path):
    """The same literal with no citation comment -> False."""
    p = _write_module(
        tmp_path,
        """
        ALPHA = 2.35
        """,
    )
    lineno = _literal_line(p, "2.35")
    assert has_nearby_citation(str(p), lineno) is False


def test_has_nearby_citation_true_for_docstring(tmp_path):
    """A literal inside a function whose docstring cites a source -> True."""
    p = _write_module(
        tmp_path,
        '''
        def relation():
            """Tout et al. (1996), Table 1."""
            coeff = 7.23
            return coeff
        ''',
    )
    lineno = _literal_line(p, "7.23")
    assert has_nearby_citation(str(p), lineno) is True


def test_module_docstring_citation_does_not_whitelist_whole_file(tmp_path):
    """A MODULE-level docstring citation must NOT whitelist a module-level literal.

    Regression for the AST-tripwire neutering bug: when ``_cited_docstring_spans`` included
    ``ast.Module``, a single citation token in the module docstring covered the ENTIRE file,
    so a brand-new uncited module-level coefficient stayed green. The module docstring's span
    must NOT provenance an arbitrary module-level literal.
    """
    p = _write_module(
        tmp_path,
        '''
        """Module that cites Tout et al. (1996), Table 1 in its docstring."""

        SOME_FUDGE = 0.73219
        ''',
    )
    lineno = _literal_line(p, "0.73219")
    assert has_nearby_citation(str(p), lineno) is False


def test_function_docstring_citation_still_whitelists_its_body(tmp_path):
    """The legitimate case still works: a literal inside a citation-bearing FUNCTION
    docstring's body IS covered (only the over-broad ast.Module span was removed)."""
    p = _write_module(
        tmp_path,
        '''
        """Module docstring with NO citation token."""

        def relation():
            """Tout et al. (1996), Table 1."""
            coeff = 7.23
            return coeff
        ''',
    )
    lineno = _literal_line(p, "7.23")
    assert has_nearby_citation(str(p), lineno) is True


# ======================================================================================
# str | Path broadening (the SoTA delta: path primitives accept pathlib.Path)
# ======================================================================================


def test_path_argument_accepted_by_scan_and_citation(tmp_path):
    """``scan_module_numeric_literals`` and ``has_nearby_citation`` accept a Path (not just
    str) and behave identically to the str form (the str | Path annotation broadening)."""
    p = _write_module(
        tmp_path,
        """
        # Salpeter (1955)
        ALPHA = 2.35
        """,
    )
    # Path object, not str:
    found = dict(scan_module_numeric_literals(p, trivial=set(), small_int_max=12))
    assert set(found) == {2.35}
    lineno = _literal_line(p, "2.35")
    assert has_nearby_citation(p, lineno) is True


# ======================================================================================
# resolve_node_ids (genuinely runs pytest --collect-only in a subprocess)
# ======================================================================================

_FIXTURES_DIR = Path(__file__).resolve().parent / "_ratchet_fixtures"
_REAL_NODE = f"{_FIXTURES_DIR / 'fixture_collect_target.py'}::test_real_target"
_BOGUS_NODE = f"{_FIXTURES_DIR / 'fixture_collect_target.py'}::test_does_not_exist"


def test_resolve_node_ids_keeps_only_real():
    """A real node id resolves; a bogus one does not."""
    resolved = resolve_node_ids([_REAL_NODE, _BOGUS_NODE], rootdir=str(_FIXTURES_DIR))
    assert _REAL_NODE in resolved
    assert _BOGUS_NODE not in resolved


def test_resolve_node_ids_rejects_import_broken_file(tmp_path):
    """A cited test in a file that fails to import is NOT reported as resolved.

    When the test's FILE fails to collect (e.g. a bad import), pytest emits a collection
    ERROR / ``found no collectors for <id>`` rather than ``not found: <id>``. The harness
    must treat that id as UNRESOLVED, not silently pass it through (anti-theater).
    """
    broken = tmp_path / "fixture_broken_import.py"
    broken.write_text(
        textwrap.dedent(
            """
            import this_module_does_not_exist  # noqa: F401

            def test_x():
                assert True
            """
        )
    )
    node_id = f"{broken}::test_x"
    resolved = resolve_node_ids([node_id], rootdir=str(tmp_path))
    assert node_id not in resolved


# ======================================================================================
# SyntaxError fallback branches (a malformed source must not explode the scanners)
# ======================================================================================


def test_scan_yields_nothing_for_malformed_syntax(tmp_path):
    """A file that fails to ``ast.parse`` -> the scanner yields nothing (no exception)."""
    p = _write_module(
        tmp_path,
        """
        def broken(:  # malformed: not parseable
            return 1
        """,
    )
    found = list(scan_module_numeric_literals(str(p), trivial=set(), small_int_max=2))
    assert found == []


def test_has_nearby_citation_false_for_malformed_syntax(tmp_path):
    """A file that fails to ``ast.parse`` (and has no citation comment) -> False."""
    p = _write_module(
        tmp_path,
        """
        def broken(:  # malformed: not parseable
            coeff = 7.23
        """,
    )
    assert has_nearby_citation(str(p), 1) is False


# ======================================================================================
# Parametrization: custom cite_re / custom assert_helpers override the baked-in defaults
# ======================================================================================


def test_has_nearby_citation_honors_custom_cite_re(tmp_path):
    """A comment matching a CUSTOM pattern but NOT the default is recognized only when the
    custom ``cite_re`` is passed (proves the parameter is wired through)."""
    p = _write_module(
        tmp_path,
        """
        # see CUSTOMREF-marker
        ALPHA = 2.35
        """,
    )
    lineno = _literal_line(p, "2.35")
    custom = re.compile(r"CUSTOMREF-marker")
    assert has_nearby_citation(str(p), lineno, cite_re=custom) is True
    assert has_nearby_citation(str(p), lineno) is False  # default does not match


def test_body_has_assert_honors_custom_assert_helpers(tmp_path):
    """A test body calling ``mylib.check_(...)`` counts as asserting only when that prefix is
    supplied via ``assert_helpers`` (proves the parameter is wired through)."""
    p = _write_test_file(
        tmp_path,
        """
        import mylib

        def test_custom():
            mylib.check_(1.0, 1.0)
        """,
    )
    node_id = f"{p}::test_custom"
    assert (
        ratchet.test_body_has_assert(node_id, assert_helpers=("mylib.check_",)) is True
    )
    assert ratchet.test_body_has_assert(node_id) is False  # default does not recognize
