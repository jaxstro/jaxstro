# rootfinding.performance

Artifact version: `1`

## Metrics

| Metric identity | Symbol | Value | Units | Status |
| --- | --- | ---: | --- | --- |
| flat_slope.bisection.executed_iterations | `N_iter` | 48 | iterations | pass |
| flat_slope.bisection.final_absolute_residual | `abs(f(x_star))` | 5.605193857299268e-45 | function units | pass |
| flat_slope.bisection.final_relative_residual | `R_root` | 7.006492321624085e-46 | dimensionless | pass |
| flat_slope.bisection.function_evaluations | `N_eval` | 50 | evaluations | pass |
| flat_slope.bisection.warm_wall | `t_wall,warm` | 5.624955520033836e-06 | s | info |
| flat_slope.safeguarded_hybrid.executed_iterations | `N_iter` | 3 | iterations | pass |
| flat_slope.safeguarded_hybrid.final_absolute_residual | `abs(f(x_star))` | 0.0 | function units | pass |
| flat_slope.safeguarded_hybrid.final_relative_residual | `R_root` | 0.0 | dimensionless | pass |
| flat_slope.safeguarded_hybrid.function_evaluations | `N_eval` | 5 | evaluations | pass |
| flat_slope.safeguarded_hybrid.warm_wall | `t_wall,warm` | 0.0003239999059587717 | s | info |
| linear.bisection.executed_iterations | `N_iter` | 48 | iterations | pass |
| linear.bisection.final_absolute_residual | `abs(f(x_star))` | 7.105427357601002e-15 | function units | pass |
| linear.bisection.final_relative_residual | `R_root` | 3.552713678800501e-15 | dimensionless | pass |
| linear.bisection.function_evaluations | `N_eval` | 50 | evaluations | pass |
| linear.bisection.warm_wall | `t_wall,warm` | 5.209119990468025e-06 | s | info |
| linear.safeguarded_hybrid.executed_iterations | `N_iter` | 1 | iterations | pass |
| linear.safeguarded_hybrid.final_absolute_residual | `abs(f(x_star))` | 0.0 | function units | pass |
| linear.safeguarded_hybrid.final_relative_residual | `R_root` | 0.0 | dimensionless | pass |
| linear.safeguarded_hybrid.function_evaluations | `N_eval` | 3 | evaluations | pass |
| linear.safeguarded_hybrid.warm_wall | `t_wall,warm` | 0.0005396248307079077 | s | info |
| monotone_kink.bisection.executed_iterations | `N_iter` | 48 | iterations | pass |
| monotone_kink.bisection.final_absolute_residual | `abs(f(x_star))` | 2.1094237467877974e-15 | function units | pass |
| monotone_kink.bisection.final_relative_residual | `R_root` | 3.515706244646329e-15 | dimensionless | pass |
| monotone_kink.bisection.function_evaluations | `N_eval` | 50 | evaluations | pass |
| monotone_kink.bisection.warm_wall | `t_wall,warm` | 5.417037755250931e-06 | s | info |
| monotone_kink.safeguarded_hybrid.executed_iterations | `N_iter` | 48 | iterations | pass |
| monotone_kink.safeguarded_hybrid.final_absolute_residual | `abs(f(x_star))` | 1.6062151608764452e-13 | function units | pass |
| monotone_kink.safeguarded_hybrid.final_relative_residual | `R_root` | 2.6770252681274087e-13 | dimensionless | pass |
| monotone_kink.safeguarded_hybrid.function_evaluations | `N_eval` | 50 | evaluations | pass |
| monotone_kink.safeguarded_hybrid.warm_wall | `t_wall,warm` | 0.0006698749493807554 | s | info |
| oscillatory_fixed_point_residual.bisection.executed_iterations | `N_iter` | 48 | iterations | pass |
| oscillatory_fixed_point_residual.bisection.final_absolute_residual | `abs(f(x_star))` | 6.661338147750939e-16 | function units | pass |
| oscillatory_fixed_point_residual.bisection.final_relative_residual | `R_root` | 5.124106267500722e-16 | dimensionless | pass |
| oscillatory_fixed_point_residual.bisection.function_evaluations | `N_eval` | 50 | evaluations | pass |
| oscillatory_fixed_point_residual.bisection.warm_wall | `t_wall,warm` | 5.208887159824371e-06 | s | info |
| oscillatory_fixed_point_residual.safeguarded_hybrid.executed_iterations | `N_iter` | 1 | iterations | pass |
| oscillatory_fixed_point_residual.safeguarded_hybrid.final_absolute_residual | `abs(f(x_star))` | 0.0 | function units | pass |
| oscillatory_fixed_point_residual.safeguarded_hybrid.final_relative_residual | `R_root` | 0.0 | dimensionless | pass |
| oscillatory_fixed_point_residual.safeguarded_hybrid.function_evaluations | `N_eval` | 3 | evaluations | pass |
| oscillatory_fixed_point_residual.safeguarded_hybrid.warm_wall | `t_wall,warm` | 0.0002910420298576355 | s | info |
| quadratic.bisection.executed_iterations | `N_iter` | 48 | iterations | pass |
| quadratic.bisection.final_absolute_residual | `abs(f(x_star))` | 2.220446049250313e-15 | function units | pass |
| quadratic.bisection.final_relative_residual | `R_root` | 1.1102230246251565e-15 | dimensionless | pass |
| quadratic.bisection.function_evaluations | `N_eval` | 50 | evaluations | pass |
| quadratic.bisection.warm_wall | `t_wall,warm` | 5.208887159824371e-06 | s | info |
| quadratic.safeguarded_hybrid.executed_iterations | `N_iter` | 40 | iterations | pass |
| quadratic.safeguarded_hybrid.final_absolute_residual | `abs(f(x_star))` | 1.6106786041140528e-15 | function units | pass |
| quadratic.safeguarded_hybrid.final_relative_residual | `R_root` | 8.053393020570264e-16 | dimensionless | pass |
| quadratic.safeguarded_hybrid.function_evaluations | `N_eval` | 42 | evaluations | pass |
| quadratic.safeguarded_hybrid.warm_wall | `t_wall,warm` | 0.0005489580798894167 | s | info |

## Comparisons

| Comparison | Metric | Relation | Reference | Units | Absolute tolerance | Relative tolerance | Status | Note |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| flat_slope.hybrid-no-more-evaluations | `flat_slope.safeguarded_hybrid.function_evaluations` | less_equal | 50 | evaluations | 0.0 | 0.0 | pass | Hybrid evaluation count must not exceed fixed-step bisection. |
| linear.hybrid-no-more-evaluations | `linear.safeguarded_hybrid.function_evaluations` | less_equal | 50 | evaluations | 0.0 | 0.0 | pass | Hybrid evaluation count must not exceed fixed-step bisection. |
| monotone_kink.hybrid-no-more-evaluations | `monotone_kink.safeguarded_hybrid.function_evaluations` | less_equal | 50 | evaluations | 0.0 | 0.0 | pass | Hybrid evaluation count must not exceed fixed-step bisection. |
| oscillatory_fixed_point_residual.hybrid-no-more-evaluations | `oscillatory_fixed_point_residual.safeguarded_hybrid.function_evaluations` | less_equal | 50 | evaluations | 0.0 | 0.0 | pass | Hybrid evaluation count must not exceed fixed-step bisection. |
| quadratic.hybrid-no-more-evaluations | `quadratic.safeguarded_hybrid.function_evaluations` | less_equal | 50 | evaluations | 0.0 | 0.0 | pass | Hybrid evaluation count must not exceed fixed-step bisection. |

## Environment policy

Recorded execution environment; wall metrics are informational.

## Limitations

- Warm wall time is hardware- and load-dependent.

## Method payload

```json
{
  "cases": [
    {
      "bracket": [
        0.0,
        4.0
      ],
      "methods": {
        "bisection": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 48
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 7.105427357601002e-15
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 3.552713678800501e-15
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 50
          },
          "status": "fixed_steps",
          "warm_wall": {
            "unit": "s",
            "value": 5.209119990468025e-06
          }
        },
        "safeguarded_hybrid": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 1
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 0.0
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 0.0
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 3
          },
          "status": 3,
          "warm_wall": {
            "unit": "s",
            "value": 0.0005396248307079077
          }
        }
      },
      "name": "linear"
    },
    {
      "bracket": [
        0.0,
        2.0
      ],
      "methods": {
        "bisection": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 48
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 2.220446049250313e-15
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 1.1102230246251565e-15
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 50
          },
          "status": "fixed_steps",
          "warm_wall": {
            "unit": "s",
            "value": 5.208887159824371e-06
          }
        },
        "safeguarded_hybrid": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 40
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 1.6106786041140528e-15
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 8.053393020570264e-16
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 42
          },
          "status": 4,
          "warm_wall": {
            "unit": "s",
            "value": 0.0005489580798894167
          }
        }
      },
      "name": "quadratic"
    },
    {
      "bracket": [
        0.0,
        3.0
      ],
      "methods": {
        "bisection": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 48
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 5.605193857299268e-45
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 7.006492321624085e-46
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 50
          },
          "status": "fixed_steps",
          "warm_wall": {
            "unit": "s",
            "value": 5.624955520033836e-06
          }
        },
        "safeguarded_hybrid": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 3
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 0.0
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 0.0
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 5
          },
          "status": 3,
          "warm_wall": {
            "unit": "s",
            "value": 0.0003239999059587717
          }
        }
      },
      "name": "flat_slope"
    },
    {
      "bracket": [
        0.0,
        1.0
      ],
      "methods": {
        "bisection": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 48
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 2.1094237467877974e-15
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 3.515706244646329e-15
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 50
          },
          "status": "fixed_steps",
          "warm_wall": {
            "unit": "s",
            "value": 5.417037755250931e-06
          }
        },
        "safeguarded_hybrid": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 48
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 1.6062151608764452e-13
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 2.6770252681274087e-13
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 50
          },
          "status": 4,
          "warm_wall": {
            "unit": "s",
            "value": 0.0006698749493807554
          }
        }
      },
      "name": "monotone_kink"
    },
    {
      "bracket": [
        0.0,
        1.0
      ],
      "methods": {
        "bisection": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 48
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 6.661338147750939e-16
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 5.124106267500722e-16
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 50
          },
          "status": "fixed_steps",
          "warm_wall": {
            "unit": "s",
            "value": 5.208887159824371e-06
          }
        },
        "safeguarded_hybrid": {
          "converged": true,
          "executed_iterations": {
            "unit": "iterations",
            "value": 1
          },
          "final_absolute_residual": {
            "unit": "function units",
            "value": 0.0
          },
          "final_relative_residual": {
            "unit": "dimensionless",
            "value": 0.0
          },
          "function_evaluations": {
            "unit": "evaluations",
            "value": 3
          },
          "status": 3,
          "warm_wall": {
            "unit": "s",
            "value": 0.0002910420298576355
          }
        }
      },
      "name": "oscillatory_fixed_point_residual"
    }
  ],
  "controls": {
    "atol": 1e-12,
    "bisection_steps": 48,
    "hybrid_max_steps": 96,
    "matched_coordinate_tolerance": {
      "unit": "coordinate units",
      "value": 5e-12
    },
    "rtol": 1e-12,
    "safeguard_fraction": 0.1
  },
  "environment": {
    "device": "cpu:0",
    "git_revision": "fd28c3a592d9feff5145f4f6d02263af22f2e228",
    "jax_backend": "cpu",
    "jax_version": "0.10.1",
    "measured_at_utc": "2026-07-13T02:34:53.941963+00:00",
    "platform": "macOS-26.1-arm64-arm-64bit-Mach-O",
    "python_version": "3.13.7",
    "working_tree_dirty": true
  },
  "precision": "float64",
  "relative_residual_definition": "abs(f(root)) / max(abs(f(lo)), abs(f(hi)))",
  "schema_version": 1,
  "warm_repeats": 21
}
```
