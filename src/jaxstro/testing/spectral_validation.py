"""Deterministic metrics and evidence-only spectral policy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SpectralMetrics:
    """Four bounded holdout metrics for one predicted spectrum."""

    median_relative_error: float
    p95_relative_error: float
    maximum_log_flux_error: float
    integrated_flux_relative_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PolicySelection:
    """Accepted policy only when one candidate dominates the alternative."""

    status: str
    accepted_policy: str | None
    primary_metric: str
    metrics: dict[str, SpectralMetrics]


def evaluate_spectral_metrics(
    wavelength: object,
    truth: object,
    prediction: object,
) -> SpectralMetrics:
    """Measure one fixed-axis spectral prediction against positive truth."""
    wavelength_array = np.asarray(wavelength, dtype=float)
    truth_array = np.asarray(truth, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if (
        wavelength_array.ndim != 1
        or truth_array.shape != wavelength_array.shape
        or prediction_array.shape != wavelength_array.shape
    ):
        raise ValueError("spectral metrics require matching one-dimensional arrays")
    if not (
        np.all(np.isfinite(wavelength_array))
        and np.all(np.isfinite(truth_array))
        and np.all(np.isfinite(prediction_array))
    ):
        raise ValueError("spectral metrics require finite arrays")
    relative_support = truth_array != 0.0
    positive_support = (truth_array > 0.0) & (prediction_array > 0.0)
    if not np.any(positive_support):
        raise ValueError("spectral metrics require positive support")
    if not np.any(relative_support):  # protected by positive support
        raise ValueError("spectral metrics require nonzero truth support")
    relative = np.abs(
        prediction_array[relative_support] / truth_array[relative_support] - 1.0
    )
    log_error = np.abs(
        np.log(prediction_array[positive_support])
        - np.log(truth_array[positive_support])
    )
    truth_integral = float(np.trapezoid(truth_array, wavelength_array))
    predicted_integral = float(np.trapezoid(prediction_array, wavelength_array))
    if truth_integral == 0.0:
        raise ValueError("spectral metrics require nonzero integrated truth")
    return SpectralMetrics(
        median_relative_error=float(np.median(relative)),
        p95_relative_error=float(np.percentile(relative, 95.0)),
        maximum_log_flux_error=float(np.max(log_error)),
        integrated_flux_relative_error=abs(predicted_integral / truth_integral - 1.0),
    )


def select_interpolation_policy(
    wavelength: object,
    truth: object,
    predictions: Mapping[str, object],
    *,
    primary_metric: str = "p95_relative_error",
) -> PolicySelection:
    """Accept the primary winner only if no secondary metric regresses."""
    if set(predictions) != {"linear", "positive_log"}:
        raise ValueError("policy comparison requires linear and positive_log")
    metrics = {
        name: evaluate_spectral_metrics(wavelength, truth, prediction)
        for name, prediction in predictions.items()
    }
    ordered = sorted(
        metrics,
        key=lambda name: (getattr(metrics[name], primary_metric), name),
    )
    winner, alternative = ordered
    secondary = (
        "median_relative_error",
        "maximum_log_flux_error",
        "integrated_flux_relative_error",
    )
    primary_win = getattr(metrics[winner], primary_metric) < getattr(
        metrics[alternative], primary_metric
    )
    dominates = primary_win and all(
        getattr(metrics[winner], name) <= getattr(metrics[alternative], name)
        for name in secondary
    )
    accepted = winner if dominates else None
    return PolicySelection(
        status="accepted" if accepted is not None else "POLICY_NOT_VALIDATED",
        accepted_policy=accepted,
        primary_metric=primary_metric,
        metrics=metrics,
    )


def longest_common_positive_slice(vertex_values: object) -> slice:
    """Return the longest contiguous spectral run positive at every vertex."""
    values = np.asarray(vertex_values, dtype=float)
    if values.ndim < 2 or values.shape[-1] < 2:
        raise ValueError("positive support requires vertices and spectral bins")
    shared = np.all(values > 0.0, axis=tuple(range(values.ndim - 1)))
    padded = np.pad(shared.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    if starts.size == 0:
        raise ValueError("no common positive spectral support")
    lengths = stops - starts
    longest = int(np.argmax(lengths))
    if lengths[longest] < 2:
        raise ValueError("common positive spectral support has fewer than two bins")
    return slice(int(starts[longest]), int(stops[longest]))


__all__ = [
    "PolicySelection",
    "SpectralMetrics",
    "evaluate_spectral_metrics",
    "longest_common_positive_slice",
    "select_interpolation_policy",
]
