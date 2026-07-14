#!/usr/bin/env python3
"""Inject the site-level navigation accessibility hook into built MyST HTML.

The inline hook is a fixed, repository-owned string: it contains no user or
document interpolation, changes only ``aria-label`` attributes, and uses no URL,
so it is base-path independent. The current Pages workflow defines no CSP that
would block this inline script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HOOK_MARKER = "jaxstro-docs-disclosure-labels-start"
HOOK_END_MARKER = "jaxstro-docs-disclosure-labels-end"
HOOK_ID = "jaxstro-docs-disclosure-labels"
ACCESSIBILITY_HOOK = f"""<!-- {HOOK_MARKER} -->
<script id="{HOOK_ID}">
(() => {{
  const disclosureSelector =
    ".myst-primary-sidebar .myst-toc-item > button[aria-controls]";

  const labelDisclosures = (root = document) => {{
    root.querySelectorAll(disclosureSelector).forEach((button) => {{
      const item = button.closest(".myst-toc-item");
      const section = item?.querySelector(":scope > [title]")?.title?.trim();
      if (!section) return;

      const action =
        button.getAttribute("aria-expanded") === "true" ? "Close" : "Open";
      const label = `${{action}} ${{section}}`;
      if (button.getAttribute("aria-label") !== label) {{
        button.setAttribute("aria-label", label);
      }}
    }});
  }};

  let queued = false;
  const schedule = () => {{
    if (queued) return;
    queued = true;
    queueMicrotask(() => {{
      queued = false;
      labelDisclosures();
    }});
  }};

  new MutationObserver(schedule).observe(document.documentElement, {{
    attributes: true,
    attributeFilter: ["aria-expanded", "aria-label"],
    childList: true,
    subtree: true,
  }});
  window.addEventListener("pageshow", schedule);
  schedule();
}})();
</script>
<!-- {HOOK_END_MARKER} -->"""

HOOK_START = f"<!-- {HOOK_MARKER} -->"
HOOK_END = f"<!-- {HOOK_END_MARKER} -->"
HOOK_SCRIPT = f'<script id="{HOOK_ID}">'


def inject_accessibility_hook(source: str) -> str:
    """Return HTML with one accessibility hook immediately before ``</body>``."""
    if HOOK_MARKER in source or HOOK_END_MARKER in source:
        if source.count(HOOK_MARKER) != 1 or source.count(HOOK_END_MARKER) != 1:
            raise ValueError("built HTML has malformed accessibility hook markers")
        start = source.index(f"<!-- {HOOK_MARKER} -->")
        end = source.index(f"<!-- {HOOK_END_MARKER} -->", start)
        end += len(f"<!-- {HOOK_END_MARKER} -->")
        return f"{source[:start]}{ACCESSIBILITY_HOOK}{source[end:]}"
    if "</body>" not in source:
        raise ValueError("built HTML has no closing body tag")
    return source.replace("</body>", f"{ACCESSIBILITY_HOOK}\n</body>", 1)


def verify_accessibility_hook(source: str) -> None:
    """Require exactly one current hook immediately before ``</body>``."""
    signal_count = sum(
        source.count(signal) for signal in (HOOK_MARKER, HOOK_END_MARKER, HOOK_ID)
    )
    if signal_count == 0:
        raise ValueError("missing accessibility hook")

    start_count = source.count(HOOK_START)
    end_count = source.count(HOOK_END)
    script_count = source.count(HOOK_SCRIPT)
    if max(start_count, end_count, script_count) > 1:
        raise ValueError("duplicate accessibility hook")
    if (start_count, end_count, script_count) != (1, 1, 1):
        raise ValueError("malformed accessibility hook")

    start = source.index(HOOK_START)
    end = source.index(HOOK_END, start) + len(HOOK_END)
    if source[start:end] != ACCESSIBILITY_HOOK:
        raise ValueError("stale accessibility hook")

    body = source.find("</body>", end)
    if body < 0 or source[end:body].strip():
        raise ValueError("malformed accessibility hook placement")


def inject_tree(root: Path) -> tuple[int, int]:
    """Patch every HTML file below *root* and return changed/unchanged counts."""
    if not root.is_dir():
        raise FileNotFoundError(f"built HTML directory does not exist: {root}")

    changed = 0
    unchanged = 0
    for path in sorted(root.rglob("*.html")):
        source = path.read_text(encoding="utf-8")
        patched = inject_accessibility_hook(source)
        if patched == source:
            unchanged += 1
            continue
        path.write_text(patched, encoding="utf-8")
        changed += 1
    return changed, unchanged


def verify_tree(root: Path) -> int:
    """Verify every HTML file below *root* and return the checked page count."""
    if not root.is_dir():
        raise FileNotFoundError(f"built HTML directory does not exist: {root}")

    paths = sorted(root.rglob("*.html"))
    if not paths:
        raise ValueError(f"built HTML directory contains no HTML files: {root}")

    errors = []
    for path in paths:
        try:
            verify_accessibility_hook(path.read_text(encoding="utf-8"))
        except ValueError as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"docs accessibility hook verification failed:\n{details}")
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="built MyST HTML directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that every HTML page has exactly one current hook",
    )
    args = parser.parse_args()
    if args.check:
        checked = verify_tree(args.root)
        print(f"docs accessibility hook: {checked} pages verified current")
        return 0
    changed, unchanged = inject_tree(args.root)
    print(f"docs accessibility hook: {changed} patched, {unchanged} already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
