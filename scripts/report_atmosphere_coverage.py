#!/usr/bin/env python3
"""Report local atmosphere library coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

from jaxstro.atmospheres.acquisition import (
    acquisition_rows_to_markdown,
    plan_targeted_acquisition,
)
from jaxstro.atmospheres.coverage import (
    coverage_rows_to_json,
    coverage_rows_to_markdown,
)
from jaxstro.atmospheres.library import AtmosphereCatalogCoverage, AtmosphereLibrary


def format_report(
    rows: list[AtmosphereCatalogCoverage] | tuple[AtmosphereCatalogCoverage, ...],
    *,
    include_acquisition: bool = False,
) -> str:
    """Format coverage rows as a human-readable Markdown report."""
    text = "## Coverage\n\n" + coverage_rows_to_markdown(rows)
    if include_acquisition:
        text += "\n## Targeted Acquisition Decision Table\n\n"
        text += acquisition_rows_to_markdown(plan_targeted_acquisition(rows))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--acquisition", action="store_true")
    args = parser.parse_args()

    library = AtmosphereLibrary.from_local(args.data_dir)
    rows = library.coverage()
    if args.format == "json":
        print(coverage_rows_to_json(rows), end="")
        return
    print(format_report(rows, include_acquisition=args.acquisition), end="")


if __name__ == "__main__":
    main()
