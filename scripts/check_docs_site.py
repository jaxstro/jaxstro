#!/usr/bin/env python3
"""Validate jaxstro's generated routes and rendered documentation DOM."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen


class DocsGateError(RuntimeError):
    """Raised when generated documentation violates a release contract."""


def _canonical_route(url: str, base_path: str = "") -> str:
    path = urlsplit(url).path or "/"
    base = "/" + base_path.strip("/") if base_path.strip("/") else ""
    if base:
        if path == base:
            path = "/"
        elif path.startswith(base + "/"):
            path = path[len(base) :]
    if path != "/":
        path = path.rstrip("/")
    return path


def extract_page_routes(
    xref: dict[str, Any],
    content_dir: Path,
    *,
    base_path: str = "",
) -> dict[str, str]:
    """Map authored source locations to unique canonical generated routes."""
    mapping: dict[str, str] = {}
    owners: dict[str, str] = {}
    for reference in xref.get("references", []):
        if reference.get("kind") != "page":
            continue
        data_name = Path(urlsplit(str(reference["data"])).path).name
        content_path = content_dir / data_name
        if not content_path.is_file():
            raise DocsGateError(f"missing generated page data: {content_path}")
        payload = json.loads(content_path.read_text(encoding="utf-8"))
        location = str(payload["location"]).lstrip("/")
        route = _canonical_route(str(reference["url"]), base_path)
        if route in owners:
            raise DocsGateError(
                f"duplicate page route {route!r}: {owners[route]} and {location}"
            )
        if location in mapping:
            raise DocsGateError(f"duplicate authored page location: {location}")
        owners[route] = location
        mapping[location] = route
    if not mapping:
        raise DocsGateError("generated xref contains no page routes")
    return dict(sorted(mapping.items()))


def validate_route_manifest(actual: dict[str, str], expected: dict[str, str]) -> None:
    """Reject silent root-flat slug reassignment or page inventory drift."""
    if actual == expected:
        return
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(
        key for key in set(actual) & set(expected) if actual[key] != expected[key]
    )
    details = []
    if missing:
        details.append(f"missing={missing}")
    if added:
        details.append(f"added={added}")
    if changed:
        details.append(
            "changed=" + repr({key: (expected[key], actual[key]) for key in changed})
        )
    raise DocsGateError("route manifest drift: " + "; ".join(details))


class _DOMCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[dict[str, str]] = []
        self.images: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if attributes.get("id"):
            self.ids.append(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes)
        if tag == "img":
            self.images.append(attributes)


def audit_html(
    route: str,
    html: str,
    valid_routes: set[str],
    *,
    base_path: str = "",
) -> None:
    """Validate IDs, internal navigation, and image alt text in rendered HTML."""
    collector = _DOMCollector()
    collector.feed(html)

    duplicate_ids = sorted(
        identifier for identifier, count in Counter(collector.ids).items() if count > 1
    )
    if duplicate_ids:
        raise DocsGateError(f"{route}: duplicate HTML id values: {duplicate_ids}")

    for attributes in collector.links:
        href = attributes["href"]
        parsed = urlsplit(href)
        if parsed.scheme in {"http", "https", "mailto", "tel"} or href.startswith("#"):
            continue
        absolute = urljoin(route.rstrip("/") + "/", href)
        target_route = _canonical_route(absolute, base_path)
        if target_route not in valid_routes:
            raise DocsGateError(
                f"{route}: unresolved internal link {href!r} -> {target_route!r}"
            )
        if attributes.get("target", "").lower() == "_blank":
            raise DocsGateError(f"{route}: internal link opens a new tab: {href!r}")

    for attributes in collector.images:
        if not attributes.get("alt", "").strip():
            raise DocsGateError(
                f"{route}: image missing nonempty alt text: "
                f"{attributes.get('src', '<unknown>')!r}"
            )


def development_server_path(route: str, base_path: str = "") -> str:
    """Return the unprefixed route exposed by ``myst start``.

    ``BASE_URL`` changes generated links and static deployment paths, but the
    development server continues to expose authored routes at its origin root.
    """
    del base_path
    return route if route.startswith("/") else "/" + route


def _fetch(url: str, *, attempts: int = 120, delay: float = 0.25) -> str:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with urlopen(url, timeout=10) as response:  # noqa: S310 - localhost gate
                if response.status != 200:
                    raise DocsGateError(f"{url}: HTTP {response.status}")
                return response.read().decode("utf-8")
        except (URLError, ConnectionError) as exc:
            last_error = exc
            time.sleep(delay)
    raise DocsGateError(f"rendered site did not become ready at {url}: {last_error}")


def audit_site(
    base_url: str,
    routes: set[str],
    *,
    base_path: str = "",
) -> None:
    """Fetch and validate every rendered page route."""
    for index, route in enumerate(sorted(routes)):
        page_path = development_server_path(route, base_path)
        html = _fetch(
            base_url.rstrip("/") + page_path,
            attempts=120 if index == 0 else 1,
        )
        audit_html(route, html, routes, base_path=base_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-path", default="")
    args = parser.parse_args(argv)

    xref_path = args.site / "myst.xref.json"
    content_dir = args.site / "content"
    xref = json.loads(xref_path.read_text(encoding="utf-8"))
    actual = extract_page_routes(xref, content_dir, base_path=args.base_path)
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_route_manifest(actual, expected)
    audit_site(args.base_url, set(actual.values()), base_path=args.base_path)
    print(
        f"docs gate passed: {len(actual)} unique routes, stable manifest, "
        "rendered IDs/links/alt text valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
