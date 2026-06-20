"""Generic ratchet primitives for jaxstro-ecosystem testing registries.

Content-free mechanisms every per-package testing registry (api_coverage, grad_audit,
physics_registry, provenance_registry) reuses: partition/staleness assertions, pytest
node-id resolution, an AST asserts-behavior check, and numeric-literal/citation scanning
for provenance tripwires. They bake in NO package-specific symbol names, paths, or
citations — only generic shapes — so each package layers its own manifests/policy on top.

TEST INFRASTRUCTURE, not core differentiable code: ``ast``/``tokenize``/``subprocess`` and
the rest of the stdlib are appropriate here, and nothing on a JAX path imports it. Ships in
the base wheel (dependency-light, no pytest at import); consumers bring their own pytest.

See the jaxstro-ecosystem ratchet-harness hoist design (progenax docs/plans, 2026-06-19).

Public surface:

- ``assert_partition(all_symbols, *buckets, label)`` — buckets exactly partition a universe.
- ``assert_no_stale(mapping, universe, label)`` — every key still exists in the universe.
- ``resolve_node_ids(node_ids, *, rootdir)`` — subset pytest ``--collect-only`` resolves.
- ``test_body_has_assert(node_id, *, assert_helpers=ASSERT_HELPERS)`` — the cited test body
  contains an assert (or a recognized assert-helper call).
- ``ASSERT_HELPERS`` — the default recognized assert-helper call-prefix allowlist.
- ``DEFAULT_CITE_RE`` — the default content-free citation regex.
- ``scan_module_numeric_literals(path, *, trivial, small_int_max)`` — citable literals.
- ``has_nearby_citation(path, lineno, *, window=4, cite_re=DEFAULT_CITE_RE)`` — a citation
  sits near a literal.

Both baked-in conventions (``ASSERT_HELPERS``, ``DEFAULT_CITE_RE``) are exposed as defaults a
downstream ecosystem package may override (pass its own ``assert_helpers`` / ``cite_re``)
without editing this shared module.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

# Recognized assert-helper call prefixes. A cited test whose body contains a call whose
# dotted name starts with one of these counts as "asserts behavior" even without a bare
# ``assert`` statement (the SoTA anti-theater check). Generic across numpy/chex/jax/pytest.
ASSERT_HELPERS: tuple[str, ...] = (
    "np.testing.assert_",
    "npt.assert_",
    "chex.assert_",
    "pytest.approx",
    "jnp.allclose",
    "jnp.isclose",
    "np.allclose",
    "np.isclose",
)

# A comment / docstring counts as a citation if it names a paper year, a table/equation
# reference, a recognized standards authority, or an explicit ``provenance:`` marker. This
# is intentionally content-free (no specific author names) — registries that want a tighter
# author-name regex layer it on top in their own test module. Public as the default; callers
# may override it by passing their own ``cite_re`` to ``has_nearby_citation``.
DEFAULT_CITE_RE = re.compile(
    r"(provenance:|\b(18|19|20)\d{2}\b|\bTable\b|\bEq\.?\b|\bSection\b|§|"
    r"\bCODATA\b|\bIAU\b|\bp(?:p|g|age)?\.?\s*\d)",
    re.IGNORECASE,
)


# ======================================================================================
# Partition / staleness primitives
# ======================================================================================


def _as_key_set(bucket: dict | set | frozenset) -> set:
    """Coerce a bucket (dict -> its keys, set -> itself) to a plain set of symbols."""
    if isinstance(bucket, dict):
        return set(bucket.keys())
    return set(bucket)


def assert_partition(
    all_symbols: set[str], *buckets: dict | set | frozenset, label: str
) -> None:
    """Assert the buckets EXACTLY partition ``all_symbols``.

    Three conditions, each with an actionable message:
      1. Coverage — the union of all bucket keys covers ``all_symbols`` (no uncategorized
         symbol).
      2. Disjointness — no symbol appears in two buckets (an ambiguous status).
      3. No stale — no bucket key is absent from ``all_symbols`` (a deleted/renamed symbol
         still referenced).
    """
    key_sets = [_as_key_set(b) for b in buckets]
    union: set[str] = set().union(*key_sets) if key_sets else set()

    missing = sorted(all_symbols - union)
    assert not missing, f"[{label}] symbols not categorized in any bucket: {missing}"

    for i in range(len(key_sets)):
        for j in range(i + 1, len(key_sets)):
            overlap = sorted(key_sets[i] & key_sets[j])
            assert not overlap, (
                f"[{label}] symbols in BOTH bucket #{i} and bucket #{j} "
                f"(ambiguous status): {overlap}"
            )

    stale = sorted(union - all_symbols)
    assert not stale, (
        f"[{label}] bucket entries no longer in the universe (stale — remove them): {stale}"
    )


def assert_no_stale(
    mapping: Mapping[str, object], universe: set[str], label: str
) -> None:
    """Assert every key in ``mapping`` still exists in ``universe`` (catches renames)."""
    stale = sorted(set(mapping) - universe)
    assert not stale, (
        f"[{label}] mapping keys no longer in the universe (stale — remove them): {stale}"
    )


# ======================================================================================
# Node-id resolution (anti-theater: the cited test actually exists)
# ======================================================================================


def resolve_node_ids(node_ids: Iterable[str], *, rootdir: str) -> set[str]:
    """Return the subset of ``node_ids`` that ``pytest --collect-only -q`` resolves.

    Runs pytest ONCE in a subprocess from ``rootdir`` collecting exactly the given ids.
    Resolution is computed as ``requested - unresolved``, where an id is UNRESOLVED if
    pytest explains it as either:

    - ``ERROR: not found: <id>`` — the node id itself does not exist in an importable
      file, OR
    - ``found no collectors for <id>`` — the id's FILE failed to import/collect (e.g. a
      bad import or a collection-time exception), so the cited test cannot be reached.

    Both modes must count as unresolved; otherwise a cited test in an import-broken file
    would be silently reported RESOLVED, defeating the anti-theater guarantee. As a final
    safety net, if pytest exits non-zero (a collection error occurred) but NONE of the
    requested ids were explained by either line — an unexpected collection-error mode — we
    do not trust the run: every unexplained requested id is treated as unresolved (the
    safe, fail-loud default).

    Matching is on the rootdir-relative form, so callers may pass either relative or
    absolute paths. ``-o addopts=`` clears any project ``-v`` (which would otherwise
    switch pytest to the indented-tree format).
    """
    ids = list(node_ids)
    if not ids:
        return set()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-header",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            *ids,
        ],
        cwd=rootdir,
        capture_output=True,
        text=True,
    )
    root = Path(rootdir)

    def _rel(nid: str) -> str:
        file_part, sep, rest = nid.partition("::")
        p = Path(file_part)
        if p.is_absolute():
            try:
                file_part = str(p.relative_to(root))
            except ValueError:
                file_part = str(p)
        return f"{file_part}{sep}{rest}"

    # Collect ids pytest explained as unresolved (on stdout or stderr, version-dependent):
    # ``not found: <id>`` (node id missing) and ``found no collectors for <id>`` (the file
    # failed to import/collect). Both count as unresolved.
    not_found_re = re.compile(r"not found:\s*(\S+)")
    no_collectors_re = re.compile(r"found no collectors for\s*(\S+)")
    unresolved_emitted: set[str] = set()
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        for rx in (not_found_re, no_collectors_re):
            m = rx.search(line)
            if m:
                unresolved_emitted.add(_rel(m.group(1).strip()))

    requested_rel = {_rel(nid) for nid in ids}
    # Fail-loud safety net: a collection error (non-zero exit) that pytest did NOT explain
    # via a not-found / no-collectors line for any requested id is an unexpected mode — do
    # not silently return all-resolved. Treat the unexplained requested ids as unresolved.
    if proc.returncode != 0 and not (requested_rel & unresolved_emitted):
        unresolved_emitted |= requested_rel

    return {nid for nid in ids if _rel(nid) not in unresolved_emitted}


# ======================================================================================
# Cited-test asserts-behavior check (AST)
# ======================================================================================


def _split_node_id(node_id: str) -> tuple[str, list[str]]:
    """Split ``file.py::A::b`` into ``("file.py", ["A", "b"])``."""
    parts = node_id.split("::")
    return parts[0], parts[1:]


def _find_test_function(tree: ast.AST, qualpath: list[str]) -> ast.FunctionDef | None:
    """Resolve ``qualpath`` (e.g. ``["TestClass", "test_x"]`` or ``["test_x"]``) to its
    FunctionDef node, walking nested ClassDef scopes."""
    scope: list[ast.stmt] = list(getattr(tree, "body", []))
    node: ast.AST | None = None
    for name in qualpath:
        node = None
        for stmt in scope:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and stmt.name == name
            ):
                node = stmt
                break
        if node is None:
            return None
        scope = list(getattr(node, "body", []))
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node  # type: ignore[return-value]
    return None


def _call_dotted_name(call: ast.Call) -> str:
    """Best-effort dotted name of a call target (e.g. ``np.testing.assert_allclose``)."""
    parts: list[str] = []
    node: ast.AST = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_body_has_assert(
    node_id: str, *, assert_helpers: tuple[str, ...] = ASSERT_HELPERS
) -> bool:
    """True iff the test fn named by ``node_id`` contains an assert (statement or helper).

    Handles both ``file.py::test_fn`` and ``file.py::Class::test_method``. "Has assert"
    means the function body contains an ``ast.Assert`` statement OR a call whose dotted
    name starts with one of ``assert_helpers`` (default ``ASSERT_HELPERS``; pass a different
    tuple to recognize another ecosystem's assert helpers). Returns False if the file or
    function can't be resolved (the caller's resolve_node_ids check guards real existence
    separately).
    """
    file_part, qualpath = _split_node_id(node_id)
    if not qualpath:
        return False
    path = Path(file_part)
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return False
    fn = _find_test_function(tree, qualpath)
    if fn is None:
        return False
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Assert):
            return True
        if isinstance(sub, ast.Call):
            dotted = _call_dotted_name(sub)
            if any(dotted.startswith(h) for h in assert_helpers):
                return True
    return False


# ======================================================================================
# Numeric-literal scanning (provenance tripwire support)
# ======================================================================================


def _is_citable_shaped(
    value: int | float, *, trivial: set[float], small_int_max: int
) -> bool:
    """A literal is 'citable-shaped' (a candidate coefficient) unless it is trivial:
    a bool, a member of ``trivial``, effectively zero (|v| < 1e-9), or a small integer
    (|v| <= ``small_int_max``) — small ints are indices / shapes / loop bounds."""
    if isinstance(value, bool):
        return False
    v = float(value)
    if v in trivial or abs(v) < 1e-9:
        return False
    if isinstance(value, int) and abs(value) <= small_int_max:
        return False
    return True


def scan_module_numeric_literals(
    path: str | Path, *, trivial: set[float], small_int_max: int
) -> Iterator[tuple[float, int]]:
    """Yield ``(value, lineno)`` for each citable-shaped numeric literal in a source file.

    ``ast.walk`` finds the numeric ``ast.Constant`` nodes (carrying accurate line numbers in
    modern Python). A negative literal such as ``-2.35`` parses as ``UnaryOp(USub,
    Constant(2.35))``, so unary +/- over a numeric constant is folded to a SIGNED value
    yielded at the unary-op's lineno (Task 3 matches provenance literals by value-text, so
    the sign must survive). Skips: bools, members of ``trivial``, |v| < 1e-9, and ints with
    |v| <= ``small_int_max``.
    """
    src = Path(path).read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Robust fallback: a non-parseable file yields nothing rather than exploding.
        return
    # Constant nodes folded into a signed UnaryOp, tracked by identity so the inner
    # Constant is not also yielded unsigned by the plain-Constant pass below.
    consumed: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            value: int | float = node.operand.value
            if isinstance(node.op, ast.USub):
                value = -value
            consumed.add(id(node.operand))
            if _is_citable_shaped(value, trivial=trivial, small_int_max=small_int_max):
                yield float(value), node.lineno
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        ):
            continue
        if isinstance(node.value, bool):
            continue
        if id(node) in consumed:
            continue
        if not _is_citable_shaped(
            node.value, trivial=trivial, small_int_max=small_int_max
        ):
            continue
        yield float(node.value), node.lineno


def _safe_tokenize(src: str) -> Iterator[tokenize.TokenInfo]:
    """Tokenize ``src``, swallowing the trailing TokenError on truncated input."""
    try:
        yield from tokenize.generate_tokens(io.StringIO(src).readline)
    except tokenize.TokenError:
        return


# ======================================================================================
# Nearby-citation detection (provenance tripwire support)
# ======================================================================================


def _cited_comment_lines(src: str, cite_re: re.Pattern[str]) -> set[int]:
    """Line numbers carrying a ``#`` comment whose text matches a citation token."""
    out: set[int] = set()
    for tok in _safe_tokenize(src):
        if tok.type == tokenize.COMMENT and cite_re.search(tok.string):
            out.add(tok.start[0])
    return out


def _cited_docstring_spans(
    tree: ast.AST, cite_re: re.Pattern[str]
) -> list[tuple[int, int]]:
    """(start, end) line spans of every class/function whose docstring carries a citation
    token — a literal inside such a block is provenanced by the docstring.

    ``ast.Module`` is DELIBERATELY EXCLUDED: a module-level docstring spans the ENTIRE
    file, so a single citation token in it would whitelist every module-level literal and
    neuter the tripwire (a new uncited module-level coefficient would stay green). Only the
    scoped class/function docstring (the legitimate "coefficient inside a cited function/
    class docstring" case) provenances a literal."""
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc and cite_re.search(doc):
                start = getattr(node, "lineno", 1)
                ends = [getattr(c, "end_lineno", None) for c in ast.walk(node)]
                end = max([e for e in ends if e is not None], default=start)
                spans.append((start, end))
    return spans


def has_nearby_citation(
    path: str | Path,
    lineno: int,
    *,
    window: int = 4,
    cite_re: re.Pattern[str] = DEFAULT_CITE_RE,
) -> bool:
    """True iff a citation-shaped comment sits within ``window`` lines ABOVE ``lineno`` (or
    on the literal's own line), OR the literal sits inside a citation-bearing docstring.

    ``cite_re`` (default ``DEFAULT_CITE_RE``) is the citation pattern; pass a custom compiled
    pattern to recognize a different ecosystem's citation convention."""
    src = Path(path).read_text()
    cited_lines = _cited_comment_lines(src, cite_re)
    if any(
        line in cited_lines for line in range(lineno, max(0, lineno - window - 1), -1)
    ):
        return True
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    return any(
        start <= lineno <= end for start, end in _cited_docstring_spans(tree, cite_re)
    )
