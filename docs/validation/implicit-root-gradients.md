# rootfinding.implicit-gradients

Artifact version: `1`

## Metrics

| Metric identity | Symbol | Value | Units | Status |
| --- | --- | ---: | --- | --- |
| exponential.absolute_residual | `abs(f(x_star))` | 0.0 | function units | info |
| exponential.ad_derivative | `dx_star/dtheta|AD` | 0.5 | coordinate units per parameter unit | info |
| exponential.analytic_derivative | `dx_star/dtheta|analytic` | 0.5 | coordinate units per parameter unit | info |
| exponential.bracket_width | `Delta_x` | 0.0 | coordinate units | info |
| exponential.certificate | `I_certified` | 1 | dimensionless | info |
| exponential.fd_derivative | `dx_star/dtheta|FD` | 0.5000000000088267 | coordinate units per parameter unit | info |
| exponential.relative_ad_analytic_error | `R_AD,analytic` | 0.0 | dimensionless | info |
| exponential.relative_ad_fd_error | `R_AD,FD` | 1.7653434269648046e-11 | dimensionless | info |
| exponential.root | `x_star` | 0.6931471805599453 | coordinate units | info |
| exponential.slope_magnitude | `abs(df/dx)` | 2.0 | function units per coordinate unit | info |
| linear.absolute_residual | `abs(f(x_star))` | 0.0 | function units | info |
| linear.ad_derivative | `dx_star/dtheta|AD` | 1.0 | coordinate units per parameter unit | info |
| linear.analytic_derivative | `dx_star/dtheta|analytic` | 1.0 | coordinate units per parameter unit | info |
| linear.bracket_width | `Delta_x` | 0.0 | coordinate units | info |
| linear.certificate | `I_certified` | 1 | dimensionless | info |
| linear.fd_derivative | `dx_star/dtheta|FD` | 1.0000000000065512 | coordinate units per parameter unit | info |
| linear.relative_ad_analytic_error | `R_AD,analytic` | 0.0 | dimensionless | info |
| linear.relative_ad_fd_error | `R_AD,FD` | 6.551204023665206e-12 | dimensionless | info |
| linear.root | `x_star` | 2.0 | coordinate units | info |
| linear.slope_magnitude | `abs(df/dx)` | 1.0 | function units per coordinate unit | info |
| quadratic.absolute_residual | `abs(f(x_star))` | 4.440892098500626e-16 | function units | info |
| quadratic.ad_derivative | `dx_star/dtheta|AD` | 0.3535533905932738 | coordinate units per parameter unit | info |
| quadratic.analytic_derivative | `dx_star/dtheta|analytic` | 0.35355339059327373 | coordinate units per parameter unit | info |
| quadratic.bracket_width | `Delta_x` | 1.7763568394002505e-14 | coordinate units | info |
| quadratic.certificate | `I_certified` | 1 | dimensionless | info |
| quadratic.fd_derivative | `dx_star/dtheta|FD` | 0.35355339059739416 | coordinate units per parameter unit | info |
| quadratic.relative_ad_analytic_error | `R_AD,analytic` | 1.5700924586837752e-16 | dimensionless | info |
| quadratic.relative_ad_fd_error | `R_AD,FD` | 1.1654168283690369e-11 | dimensionless | info |
| quadratic.root | `x_star` | 1.414213562373095 | coordinate units | info |
| quadratic.slope_magnitude | `abs(df/dx)` | 2.82842712474619 | function units per coordinate unit | info |

## Comparisons

| Comparison | Metric | Relation | Reference | Units | Absolute tolerance | Relative tolerance | Status | Note |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| exponential.absolute_residual.gate | `exponential.absolute_residual` | less_equal | 1e-12 | function units | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| exponential.bracket_width.gate | `exponential.bracket_width` | less_equal | 1e-12 | coordinate units | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| exponential.certificate.gate | `exponential.certificate` | equal | 1 | dimensionless | 0.0 | 0.0 | pass | The complete runtime certificate accepted the case. |
| exponential.relative_ad_analytic_error.gate | `exponential.relative_ad_analytic_error` | less_equal | 1e-09 | dimensionless | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| exponential.relative_ad_fd_error.gate | `exponential.relative_ad_fd_error` | less_equal | 1e-06 | dimensionless | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| exponential.slope_magnitude.gate | `exponential.slope_magnitude` | greater_equal | 1e-08 | function units per coordinate unit | 0.0 | 0.0 | pass | Implicit derivative requires adequate root-slope conditioning. |
| linear.absolute_residual.gate | `linear.absolute_residual` | less_equal | 1e-12 | function units | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| linear.bracket_width.gate | `linear.bracket_width` | less_equal | 1e-12 | coordinate units | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| linear.certificate.gate | `linear.certificate` | equal | 1 | dimensionless | 0.0 | 0.0 | pass | The complete runtime certificate accepted the case. |
| linear.relative_ad_analytic_error.gate | `linear.relative_ad_analytic_error` | less_equal | 1e-09 | dimensionless | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| linear.relative_ad_fd_error.gate | `linear.relative_ad_fd_error` | less_equal | 1e-06 | dimensionless | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| linear.slope_magnitude.gate | `linear.slope_magnitude` | greater_equal | 1e-08 | function units per coordinate unit | 0.0 | 0.0 | pass | Implicit derivative requires adequate root-slope conditioning. |
| quadratic.absolute_residual.gate | `quadratic.absolute_residual` | less_equal | 1e-12 | function units | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| quadratic.bracket_width.gate | `quadratic.bracket_width` | less_equal | 1e-12 | coordinate units | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| quadratic.certificate.gate | `quadratic.certificate` | equal | 1 | dimensionless | 0.0 | 0.0 | pass | The complete runtime certificate accepted the case. |
| quadratic.relative_ad_analytic_error.gate | `quadratic.relative_ad_analytic_error` | less_equal | 1e-09 | dimensionless | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| quadratic.relative_ad_fd_error.gate | `quadratic.relative_ad_fd_error` | less_equal | 1e-06 | dimensionless | 0.0 | 0.0 | pass | Existing implicit-root validation threshold. |
| quadratic.slope_magnitude.gate | `quadratic.slope_magnitude` | greater_equal | 1e-08 | function units per coordinate unit | 0.0 | 0.0 | pass | Implicit derivative requires adequate root-slope conditioning. |

## Environment policy

environment is an emission snapshot; --check gates deterministic controls, schema, units, and algorithmic metrics, not current revision

## Limitations

- Certification relies on caller assertions of uniqueness and smoothness.
- Flat-slope rejection is validated in executable tests, not represented as a certified case.

## Method payload

```json
{
  "cases": [
    {
      "absolute_residual": {
        "unit": "function units",
        "value": 0.0
      },
      "ad_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 1.0
      },
      "analytic_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 1.0
      },
      "bracket_width": {
        "unit": "coordinate units",
        "value": 0.0
      },
      "certified": true,
      "fd_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 1.0000000000065512
      },
      "name": "linear",
      "relative_ad_analytic_error": {
        "unit": "dimensionless",
        "value": 0.0
      },
      "relative_ad_fd_error": {
        "unit": "dimensionless",
        "value": 6.551204023665206e-12
      },
      "root": {
        "unit": "coordinate units",
        "value": 2.0
      },
      "slope_magnitude": {
        "unit": "function units per coordinate unit",
        "value": 1.0
      },
      "status": 0
    },
    {
      "absolute_residual": {
        "unit": "function units",
        "value": 4.440892098500626e-16
      },
      "ad_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 0.3535533905932738
      },
      "analytic_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 0.35355339059327373
      },
      "bracket_width": {
        "unit": "coordinate units",
        "value": 1.7763568394002505e-14
      },
      "certified": true,
      "fd_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 0.35355339059739416
      },
      "name": "quadratic",
      "relative_ad_analytic_error": {
        "unit": "dimensionless",
        "value": 1.5700924586837752e-16
      },
      "relative_ad_fd_error": {
        "unit": "dimensionless",
        "value": 1.1654168283690369e-11
      },
      "root": {
        "unit": "coordinate units",
        "value": 1.414213562373095
      },
      "slope_magnitude": {
        "unit": "function units per coordinate unit",
        "value": 2.82842712474619
      },
      "status": 0
    },
    {
      "absolute_residual": {
        "unit": "function units",
        "value": 0.0
      },
      "ad_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 0.5
      },
      "analytic_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 0.5
      },
      "bracket_width": {
        "unit": "coordinate units",
        "value": 0.0
      },
      "certified": true,
      "fd_derivative": {
        "unit": "coordinate units per parameter unit",
        "value": 0.5000000000088267
      },
      "name": "exponential",
      "relative_ad_analytic_error": {
        "unit": "dimensionless",
        "value": 0.0
      },
      "relative_ad_fd_error": {
        "unit": "dimensionless",
        "value": 1.7653434269648046e-11
      },
      "root": {
        "unit": "coordinate units",
        "value": 0.6931471805599453
      },
      "slope_magnitude": {
        "unit": "function units per coordinate unit",
        "value": 2.0
      },
      "status": 0
    }
  ],
  "controls": {
    "fd_step": {
      "unit": "parameter units",
      "value": 1e-05
    },
    "residual_limit": {
      "unit": "function units",
      "value": 1e-12
    },
    "slope_floor": {
      "unit": "function units per coordinate unit",
      "value": 1e-08
    },
    "width_limit": {
      "unit": "coordinate units",
      "value": 1e-12
    }
  },
  "environment": {
    "device": "cpu:0",
    "git_revision": "a3f0d4bd563555e67a3ec0ca8e1ed0e8be671f60",
    "jax_backend": "cpu",
    "jax_version": "0.10.1",
    "measured_at_utc": "2026-07-13T03:01:34.508984+00:00",
    "platform": "macOS-26.1-arm64-arm-64bit-Mach-O",
    "python_version": "3.13.7",
    "working_tree_dirty": true
  },
  "precision": "float64",
  "provenance_policy": "environment is an emission snapshot; --check gates deterministic controls, schema, units, and algorithmic metrics, not current revision",
  "schema_version": 2
}
```
