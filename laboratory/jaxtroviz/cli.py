"""List, render, or freshness-check registered JaxtroViz figures."""

from __future__ import annotations

import argparse
import shutil

from .registry import FIGURES
from .style import render_webp_bytes, save_figure_formats


def _selected_names(only: list[str] | None) -> list[str]:
    names = list(FIGURES) if only is None else only
    unknown = sorted(set(names) - FIGURES.keys())
    if unknown:
        raise ValueError(f"unknown figure(s): {', '.join(unknown)}")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list registered figures")
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="select figures by registry name"
    )
    parser.add_argument(
        "--all", action="store_true", help="render every registered figure"
    )
    parser.add_argument(
        "--check", action="store_true", help="check committed WebP freshness"
    )
    args = parser.parse_args(argv)

    if args.list or not (args.only or args.all or args.check):
        for name, spec in FIGURES.items():
            print(f"  {name:<28} -> {spec.stem}.{{pdf,png,webp}}  [{spec.page}]")
            if spec.caption:
                print(f"    {spec.caption}")
        return 0

    try:
        names = _selected_names(args.only)
    except ValueError as error:
        parser.error(str(error))

    for name in names:
        spec = FIGURES[name]
        if args.check:
            expected = render_webp_bytes(spec.builder(), spec=spec.export)
            if not spec.site_webp.is_file() or spec.site_webp.read_bytes() != expected:
                print(f"stale {spec.site_webp}")
                return 1
            print(f"fresh {spec.site_webp}")
            continue
        for path in save_figure_formats(
            spec.builder(), spec.output_stem, spec=spec.export
        ):
            print(f"wrote {path}")
        spec.site_webp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.output_stem.with_suffix(".webp"), spec.site_webp)
        print(f"wrote {spec.site_webp}  (site embed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
