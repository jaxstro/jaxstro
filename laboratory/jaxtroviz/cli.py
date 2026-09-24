"""List, render, or freshness-check registered JaxtroViz figures."""

from __future__ import annotations

import argparse
import io
import platform

from PIL import Image

from .registry import FIGURES
from .style import render_webp_bytes, save_figure_formats


def _selected_names(only: list[str] | None) -> list[str]:
    names = list(FIGURES) if only is None else only
    unknown = sorted(set(names) - FIGURES.keys())
    if unknown:
        raise ValueError(f"unknown figure(s): {', '.join(unknown)}")
    return names


# The committed WebP files are rendered on macOS arm64. Fonts, FreeType, and
# libwebp differ elsewhere, so bytes are compared only on that platform; other
# platforms check that each figure renders to an image of the same size.
# Approved 2026-09-24.
RENDER_ORIGIN = ("macOS", "arm64")


def _on_render_origin() -> bool:
    return (platform.platform().split("-")[0], platform.machine()) == RENDER_ORIGIN


def _webp_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def _is_fresh(committed: bytes, rendered: bytes) -> bool:
    if _on_render_origin():
        return committed == rendered
    return _webp_size(committed) == _webp_size(rendered)


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
            if not spec.site_webp.is_file():
                print(f"missing {spec.site_webp}")
                return 1
            committed = spec.site_webp.read_bytes()
            if not _is_fresh(committed, expected):
                print(
                    f"stale {spec.site_webp}: committed {_webp_size(committed)}, "
                    f"rendered {_webp_size(expected)}"
                )
                return 1
            print(f"fresh {spec.site_webp}")
            continue
        for path in save_figure_formats(
            spec.builder(), spec.output_stem, spec=spec.export
        ):
            print(f"wrote {path}")
        spec.site_webp.parent.mkdir(parents=True, exist_ok=True)
        spec.site_webp.write_bytes(render_webp_bytes(spec.builder(), spec=spec.export))
        print(f"wrote {spec.site_webp}  (site embed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
