"""Contracts for future method background and delegated ecosystem guides."""

from __future__ import annotations

import cmath
import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
METHODS = DOCS / "20-methods"

GUIDES = {
    "change-constraints-evolution/nonlinear-systems.md": "Ecosystem guide",
    "change-constraints-evolution/adaptive-differential-equations.md": (
        "Ecosystem guide"
    ),
    "approximation-integration/adaptive-quadrature.md": "Ecosystem guide",
    "linear-structure/iterative-linear-solvers.md": "Ecosystem guide",
    "probability-sampling/quasi-monte-carlo.md": "Planned Jaxstro capability",
    "signals/signal-axes.md": "Planned Jaxstro capability",
    "signals/windows-spectral-leakage.md": "Planned Jaxstro capability",
    "signals/spectral-estimation.md": "Planned Jaxstro capability",
    "signals/phase-and-delay.md": "Planned Jaxstro capability",
}

ROUTES = {relative: f"/{Path(relative).stem}" for relative in GUIDES}

COMMON_SECTIONS = (
    "## The scientific question",
    "## Mathematical objects",
    "## Core derivation",
    "## What the ecosystem already owns",
    "## What Jaxstro may add",
    "## Evidence required before implementation",
    "## Claim boundary",
    "## Connected foundations and methods",
)

DELEGATED_OWNERS = {
    "change-constraints-evolution/nonlinear-systems.md": (
        "Optimistix",
        "https://docs.kidger.site/optimistix/",
    ),
    "change-constraints-evolution/adaptive-differential-equations.md": (
        "Diffrax",
        "https://docs.kidger.site/diffrax/",
    ),
    "approximation-integration/adaptive-quadrature.md": (
        "Quadax",
        "https://quadax.readthedocs.io/en/",
    ),
    "linear-structure/iterative-linear-solvers.md": (
        "Lineax",
        "https://docs.kidger.site/lineax/",
    ),
}

CURRENT_METHOD_ROUTES = {
    "20-methods/change-constraints-evolution/autodiff.md": "/autodiff",
    "20-methods/change-constraints-evolution/rootfinding.md": "/rootfinding",
    "20-methods/change-constraints-evolution/optimization.md": "/optimization",
    "20-methods/change-constraints-evolution/ode.md": "/ode",
    "20-methods/approximation-integration/interpolation.md": "/interpolation",
    "20-methods/approximation-integration/regular-grid.md": "/regular-grid",
    "20-methods/approximation-integration/bsplines.md": "/bsplines",
    "20-methods/approximation-integration/cumulative-trapz.md": ("/cumulative-trapz"),
    "20-methods/approximation-integration/quadrature.md": "/quadrature",
    "20-methods/linear-structure/linear-algebra.md": "/linear-algebra",
    "20-methods/linear-structure/operators.md": "/operators",
    "20-methods/linear-structure/special-functions.md": "/special-functions",
    "20-methods/probability-sampling/distributions.md": "/distributions",
    "20-methods/probability-sampling/random.md": "/random",
    "20-methods/probability-sampling/sampling.md": "/sampling",
    "20-methods/discrete-space/grids.md": "/grids",
    "20-methods/discrete-space/meshes.md": "/meshes",
    "20-methods/discrete-space/spatial.md": "/spatial",
}


def _read(relative: str) -> str:
    return (METHODS / relative).read_text(encoding="utf-8")


def _dft(values: list[complex]) -> list[complex]:
    """Evaluate the negative-exponent, unnormalized DFT documented by the guide."""
    size = len(values)
    return [
        sum(
            value * cmath.exp(-2j * math.pi * frequency * sample / size)
            for sample, value in enumerate(values)
        )
        for frequency in range(size)
    ]


def _inverse_dft(values: list[complex]) -> list[complex]:
    """Evaluate the positive-exponent inverse DFT with its documented 1/N."""
    size = len(values)
    return [
        sum(
            value * cmath.exp(2j * math.pi * frequency * sample / size)
            for frequency, value in enumerate(values)
        )
        / size
        for sample in range(size)
    ]


def _assert_complex_sequences_close(
    actual: list[complex], expected: list[complex]
) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.real == pytest.approx(expected_value.real, abs=1e-12)
        assert actual_value.imag == pytest.approx(expected_value.imag, abs=1e-12)


def _compact_math(source: str) -> str:
    return re.sub(r"\s+", "", source)


def test_all_guides_exist_once_in_toc_and_manifest_with_native_routes() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))

    assert len(GUIDES) == 9
    for relative in GUIDES:
        source = f"20-methods/{relative}"
        assert (METHODS / relative).is_file(), source
        assert myst.count(f"file: {source}") == 1, source
        assert list(manifest).count(source) == 1, source
        assert manifest[source] == ROUTES[relative]


def test_exact_status_admonitions_and_common_section_order() -> None:
    for relative, status in GUIDES.items():
        text = _read(relative)
        markers = (
            "Use this page when",
            f":::{'{'}important{'}'} {status}",
            *COMMON_SECTIONS,
        )
        positions = [text.index(marker) for marker in markers]
        assert positions == sorted(positions), relative
        assert text.count(f":::{'{'}important{'}'} {status}") == 1, relative


def test_each_guide_has_labeled_math_and_matching_cross_reference() -> None:
    label_pattern = re.compile(
        r"```\{math\}\s*\n:label:\s*(eq-[a-z0-9-]+)\s*\n", re.MULTILINE
    )
    for relative in GUIDES:
        text = _read(relative)
        labels = label_pattern.findall(text)
        assert labels, relative
        assert any(f"[]({f'#{label}'})" in text for label in labels), relative


def test_delegated_guides_name_and_link_official_owners() -> None:
    for relative, (owner, url) in DELEGATED_OWNERS.items():
        text = _read(relative)
        assert f"[{owner}]({url})" in text, relative

    iterative = _read("linear-structure/iterative-linear-solvers.md")
    assert "[JAX](https://docs.jax.dev/" in iterative


def test_qmc_distinguishes_three_point_constructions_and_error_claims() -> None:
    text = _read("probability-sampling/quasi-monte-carlo.md").lower()
    for phrase in (
        "deterministic low-discrepancy points",
        "independent random samples",
        "replicated randomized scrambles",
        "does not provide an uncertainty estimate",
    ):
        assert phrase in text
    assert "for $r \\geq 2$ independent scrambles" in text


def test_signal_pages_pin_runtime_ownership_and_planned_status() -> None:
    for relative in (
        "signals/signal-axes.md",
        "signals/windows-spectral-leakage.md",
        "signals/spectral-estimation.md",
        "signals/phase-and-delay.md",
    ):
        text = _read(relative)
        assert "[JAX FFT](https://docs.jax.dev/" in text, relative
        assert "`jaxstro.signal` does not exist" in text, relative

    windows = _read("signals/windows-spectral-leakage.md")
    assert "coherent gain" in windows
    assert "equivalent noise bandwidth" in windows

    estimation = _read("signals/spectral-estimation.md")
    assert "one-sided" in estimation
    assert "two-sided" in estimation

    phase = _read("signals/phase-and-delay.md")
    assert "cross spectrum" in phase
    assert "Phase wrapping" in phase


def test_documented_dft_roundtrip_and_window_product_theorem() -> None:
    signal = [1.0 + 0.5j, -2.0j, 3.0 - 1.0j, -0.5 + 2.0j]
    window = [1.0, 0.25, 0.5, 0.75]
    signal_spectrum = _dft(signal)
    window_spectrum = _dft([complex(value) for value in window])

    _assert_complex_sequences_close(_inverse_dft(signal_spectrum), signal)

    windowed_spectrum = _dft(
        [value * weight for value, weight in zip(signal, window, strict=True)]
    )
    size = len(signal)
    expected_product_spectrum = [
        sum(
            window_spectrum[index] * signal_spectrum[(frequency - index) % size]
            for index in range(size)
        )
        / size
        for frequency in range(size)
    ]
    _assert_complex_sequences_close(windowed_spectrum, expected_product_spectrum)

    axes = _compact_math(_read("signals/signal-axes.md"))
    assert r"X_k=\sum_{n=0}^{N-1}x_n" in axes
    assert r"x_n=\frac{1}{N}\sum_{k=0}^{N-1}X_k" in axes

    windows = _read("signals/windows-spectral-leakage.md")
    compact_windows = _compact_math(windows)
    assert ":label:eq-window-product-theorem" in compact_windows
    assert r"Y_k=\frac{1}{N}\sum_{m=0}^{N-1}W_mX_{(k-m)\bmodN}" in compact_windows
    assert "[](#eq-window-product-theorem)" in windows


def test_even_and_odd_real_fft_axes_pin_the_nyquist_limit() -> None:
    sample_frequency = 8.0

    def real_fft_axis(size: int) -> list[float]:
        return [index * sample_frequency / size for index in range(size // 2 + 1)]

    assert real_fft_axis(8) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert real_fft_axis(7) == pytest.approx([0.0, 8.0 / 7.0, 16.0 / 7.0, 24.0 / 7.0])
    assert real_fft_axis(8)[-1] == sample_frequency / 2
    assert real_fft_axis(7)[-1] == 3 * sample_frequency / 7
    assert real_fft_axis(7)[-1] < sample_frequency / 2

    axes = _read("signals/signal-axes.md")
    for phrase in (
        "Nyquist limit",
        "$k=N/2$",
        "$k=(N-1)/2$",
        "no Nyquist bin",
        "one-sided endpoint",
    ):
        assert phrase in axes


@pytest.mark.parametrize(
    "signal",
    (
        [2.0, -1.0, 0.5, 3.0],
        [2.0, -1.0, 0.5, 3.0, -0.25],
    ),
)
def test_parseval_and_one_sided_endpoint_accounting(signal: list[float]) -> None:
    cadence = 0.25
    size = len(signal)
    spectrum = _dft([complex(value) for value in signal])
    delta_frequency = 1.0 / (size * cadence)
    two_sided_density = [cadence * abs(value) ** 2 / size for value in spectrum]
    time_mean_square = sum(value**2 for value in signal) / size

    assert sum(two_sided_density) * delta_frequency == pytest.approx(time_mean_square)

    one_sided_density = two_sided_density[: size // 2 + 1]
    if size % 2 == 0:
        one_sided_density[1:-1] = [2 * value for value in one_sided_density[1:-1]]
    else:
        one_sided_density[1:] = [2 * value for value in one_sided_density[1:]]
    assert sum(one_sided_density) * delta_frequency == pytest.approx(time_mean_square)

    estimation = _read("signals/spectral-estimation.md")
    compact_estimation = _compact_math(estimation)
    assert r"P_k^{(2)}=\frac{\Deltat}{N}|X_k|^2" in compact_estimation
    assert r"\sum_{k=0}^{N-1}P_k^{(2)}\Deltaf" in compact_estimation
    assert "even-length Nyquist bin" in estimation
    assert "odd-length one-sided spectrum doubles every strictly positive bin" in (
        estimation
    )


def test_known_delayed_pair_pins_cross_spectrum_and_delay_sign() -> None:
    size = 8
    delay_samples = 1
    first = [1.0 + 0.0j, *([0.0j] * (size - 1))]
    second = first[-delay_samples:] + first[:-delay_samples]
    first_spectrum = _dft(first)
    second_spectrum = _dft(second)
    frequency_bin = 1
    frequency = frequency_bin / size

    cross_spectrum = (
        first_spectrum[frequency_bin].conjugate() * second_spectrum[frequency_bin]
    )
    phase = cmath.phase(cross_spectrum)
    inferred_delay = -phase / (2 * math.pi * frequency)

    assert phase == pytest.approx(-2 * math.pi * frequency * delay_samples)
    assert inferred_delay == pytest.approx(delay_samples)

    phase_page = _compact_math(_read("signals/phase-and-delay.md"))
    assert r"C_{xy}(f)=X^*(f)Y(f)" in phase_page
    assert r"\tau(f)=-\frac{\phi(f)}{2\pif}" in phase_page


def test_planned_pages_do_not_claim_unimplemented_runtime_modules() -> None:
    planned = [
        relative for relative, status in GUIDES.items() if status.startswith("Planned")
    ]
    forbidden = (
        "from jaxstro.signal import",
        "from jaxstro.numerics.qmc import",
        "`jaxstro.signal` provides",
        "`jaxstro.numerics.qmc` provides",
        "implemented in `jaxstro.signal`",
        "implemented in `jaxstro.numerics.qmc`",
    )
    for relative in planned:
        text = _read(relative)
        assert "does not exist" in text, relative
        assert not any(phrase in text for phrase in forbidden), relative


def test_new_sources_are_ascii_and_use_latex_for_math() -> None:
    for relative in GUIDES:
        text = _read(relative)
        assert text.isascii(), relative
        assert "```{math}" in text, relative


def test_new_routes_preserve_all_eighteen_current_method_routes() -> None:
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    assert len(CURRENT_METHOD_ROUTES) == 18
    for source, route in CURRENT_METHOD_ROUTES.items():
        assert manifest[source] == route


def test_methods_landing_links_the_signal_family() -> None:
    text = (METHODS / "methods.md").read_text(encoding="utf-8")
    assert ":link: ./signals/signal-axes.md" in text
