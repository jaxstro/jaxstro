"""Tests for deterministic spectral holdout metrics and policy selection."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxstro.testing.spectral_validation import (
    evaluate_spectral_metrics,
    longest_common_positive_slice,
    select_interpolation_policy,
)


def test_spectral_metrics_match_analytic_fixture() -> None:
    wavelength = jnp.array([100.0, 200.0, 400.0])
    truth = jnp.array([1.0, 2.0, 4.0])
    prediction = jnp.array([1.1, 1.8, 4.4])

    metrics = evaluate_spectral_metrics(wavelength, truth, prediction)

    np.testing.assert_allclose(metrics.median_relative_error, 0.1)
    np.testing.assert_allclose(metrics.p95_relative_error, 0.1)
    np.testing.assert_allclose(
        metrics.maximum_log_flux_error,
        max(abs(np.log(1.1)), abs(np.log(0.9))),
    )
    expected_integrated = abs(
        np.trapezoid(prediction, wavelength) / np.trapezoid(truth, wavelength) - 1.0
    )
    np.testing.assert_allclose(
        metrics.integrated_flux_relative_error, expected_integrated
    )


def test_policy_selection_requires_primary_win_and_secondary_nonregression() -> None:
    wavelength = jnp.array([100.0, 200.0, 400.0])
    truth = jnp.array([1.0, 2.0, 4.0])
    predictions = {
        "linear": jnp.array([1.05, 1.9, 4.2]),
        "positive_log": jnp.array([1.2, 1.7, 4.8]),
    }

    selection = select_interpolation_policy(wavelength, truth, predictions)

    assert selection.accepted_policy == "linear"
    assert selection.primary_metric == "p95_relative_error"
    assert set(selection.metrics) == {"linear", "positive_log"}


def test_policy_selection_returns_unvalidated_when_metrics_trade_off() -> None:
    wavelength = jnp.array([100.0, 200.0, 400.0])
    truth = jnp.array([1.0, 10.0, 1.0])
    predictions = {
        "linear": jnp.array([1.0, 8.0, 1.0]),
        "positive_log": jnp.array([1.2, 10.0, 1.2]),
    }

    selection = select_interpolation_policy(wavelength, truth, predictions)

    assert selection.accepted_policy is None
    assert selection.status == "POLICY_NOT_VALIDATED"


def test_policy_selection_requires_a_strict_primary_win() -> None:
    wavelength = jnp.array([100.0, 200.0, 400.0])
    truth = jnp.array([1.0, 2.0, 4.0])
    prediction = jnp.array([1.05, 1.9, 4.2])

    selection = select_interpolation_policy(
        wavelength,
        truth,
        {"linear": prediction, "positive_log": prediction},
    )

    assert selection.accepted_policy is None
    assert selection.status == "POLICY_NOT_VALIDATED"


def test_relative_metrics_do_not_hide_nonpositive_predictions() -> None:
    metrics = evaluate_spectral_metrics(
        jnp.array([100.0, 200.0, 400.0]),
        jnp.array([1.0, 2.0, 4.0]),
        jnp.array([0.0, 2.0, 4.0]),
    )

    assert metrics.p95_relative_error > 0.0


def test_longest_common_positive_slice_is_contiguous_and_shared() -> None:
    values = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0],
            [0.0, 2.0, 3.0, 4.0, 0.0, 5.0, 0.0],
        ]
    )

    support = longest_common_positive_slice(values)

    assert support == slice(1, 4)


def test_metrics_reject_nonfinite_or_nonpositive_support() -> None:
    import pytest

    with pytest.raises(ValueError, match="finite"):
        evaluate_spectral_metrics(
            jnp.array([1.0, 2.0]),
            jnp.array([1.0, jnp.nan]),
            jnp.array([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="positive support"):
        evaluate_spectral_metrics(
            jnp.array([1.0, 2.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([1.0, 2.0]),
        )
