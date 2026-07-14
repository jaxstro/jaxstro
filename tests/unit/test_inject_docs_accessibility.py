"""Tests for the deterministic docs accessibility post-build hook."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "inject_docs_accessibility.py"

spec = importlib.util.spec_from_file_location("inject_docs_accessibility", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_injected_hook_is_scoped_contextual_and_idempotent() -> None:
    source = "<html><body><main>Page</main></body></html>"

    patched = module.inject_accessibility_hook(source)

    assert module.HOOK_MARKER in patched
    assert '<script id="jaxstro-docs-disclosure-labels">' in patched
    assert ".myst-primary-sidebar .myst-toc-item > button[aria-controls]" in patched
    assert ":scope > [title]" in patched
    assert 'button.getAttribute("aria-expanded") === "true" ? "Close" : "Open"' in (
        patched
    )
    assert "new MutationObserver(schedule)" in patched
    assert module.inject_accessibility_hook(patched) == patched

    stale = patched.replace(
        'window.addEventListener("pageshow", schedule);', "stale-hook"
    )
    refreshed = module.inject_accessibility_hook(stale)
    assert "stale-hook" not in refreshed
    assert 'window.addEventListener("pageshow", schedule);' in refreshed


def test_inject_tree_patches_nested_html_and_leaves_other_files_unchanged(
    tmp_path: Path,
) -> None:
    root = tmp_path / "html"
    nested = root / "method"
    nested.mkdir(parents=True)
    index = root / "index.html"
    page = nested / "index.html"
    asset = root / "asset.json"
    index.write_text("<html><body>Home</body></html>", encoding="utf-8")
    page.write_text("<html><body>Method</body></html>", encoding="utf-8")
    asset.write_text('{"kept": true}\n', encoding="utf-8")

    changed, unchanged = module.inject_tree(root)

    assert (changed, unchanged) == (2, 0)
    assert module.HOOK_MARKER in index.read_text(encoding="utf-8")
    assert module.HOOK_MARKER in page.read_text(encoding="utf-8")
    assert asset.read_text(encoding="utf-8") == '{"kept": true}\n'
    assert module.inject_tree(root) == (0, 2)


def test_verify_tree_accepts_exactly_one_current_hook_per_html_page(
    tmp_path: Path,
) -> None:
    root = tmp_path / "html"
    nested = root / "method"
    nested.mkdir(parents=True)
    (root / "index.html").write_text(
        module.inject_accessibility_hook("<html><body>Home</body></html>"),
        encoding="utf-8",
    )
    (nested / "index.html").write_text(
        module.inject_accessibility_hook("<html><body>Method</body></html>"),
        encoding="utf-8",
    )

    assert module.verify_tree(root) == 2


def test_verify_hook_identity_is_scoped_to_script_elements() -> None:
    source = module.inject_accessibility_hook("<html><body>Page</body></html>")
    source = source.replace("<body>", f"<body><div id='{module.HOOK_ID}'></div>")

    module.verify_accessibility_hook(source)


@pytest.mark.parametrize(
    ("broken_source", "expected"),
    [
        ("<html><body>Missing</body></html>", "missing"),
        (
            module.inject_accessibility_hook(
                "<html><body>Duplicate</body></html>"
            ).replace("</body>", f"{module.ACCESSIBILITY_HOOK}\n</body>"),
            "duplicate",
        ),
        (
            module.inject_accessibility_hook(
                "<html><body>Duplicate</body></html>"
            ).replace(
                "<body>",
                f"<body><script id='{module.HOOK_ID}'></script>",
            ),
            "duplicate",
        ),
        (
            module.inject_accessibility_hook(
                "<html><body>Duplicate</body></html>"
            ).replace(
                "<body>",
                f"<body><script data-test=\"duplicate\" id = '{module.HOOK_ID}'></script>",
            ),
            "duplicate",
        ),
        (
            module.inject_accessibility_hook("<html><body>Stale</body></html>").replace(
                'window.addEventListener("pageshow", schedule);', "stale-hook"
            ),
            "stale",
        ),
        (
            module.inject_accessibility_hook(
                "<html><body>Malformed</body></html>"
            ).replace(f"<!-- {module.HOOK_END_MARKER} -->", ""),
            "malformed",
        ),
        (
            module.inject_accessibility_hook("<html><body>Misplaced</body></html>")
            .replace(module.ACCESSIBILITY_HOOK, "")
            .replace("</html>", f"{module.ACCESSIBILITY_HOOK}</html>"),
            "placement",
        ),
    ],
)
def test_verify_tree_rejects_noncurrent_hooks(
    tmp_path: Path, broken_source: str, expected: str
) -> None:
    root = tmp_path / "html"
    root.mkdir()
    (root / "index.html").write_text(broken_source, encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        module.verify_tree(root)
