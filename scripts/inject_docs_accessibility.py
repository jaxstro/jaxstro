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
ACCESSIBILITY_HOOK = f"""<!-- {HOOK_MARKER} -->
<script id="jaxstro-docs-disclosure-labels">
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="built MyST HTML directory")
    args = parser.parse_args()
    changed, unchanged = inject_tree(args.root)
    print(f"docs accessibility hook: {changed} patched, {unchanged} already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
