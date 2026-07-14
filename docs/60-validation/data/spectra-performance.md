# spectra.performance

Artifact version: `1`

## Metrics

| Metric identity | Symbol | Value | Units | Status |
| --- | --- | ---: | --- | --- |
| spectra.batched_evaluation | `t_wall,batch` | 0.08381337486207485 | s | info |
| spectra.cached_evaluation_median | `t_wall,warm` | 6.875023245811462e-06 | s | info |
| spectra.first_jit_evaluation | `t_wall,first-jit` | 0.07296716584824026 | s | info |
| spectra.host_peak_memory | `M_host,peak` | 13877111 | bytes | info |
| spectra.host_preparation | `t_host,prepare` | 0.943114958005026 | s | info |

## Comparisons

- none

## Environment policy

Backend, dtype, and host-memory scope recorded; performance metrics are informational.

## Limitations

- Wall times depend on hardware and system load.
- Host peak memory covers Python allocations measured by tracemalloc.

## Method payload

```json
{
  "batched_output_shape": [
    8,
    64
  ],
  "case": {
    "batch_size": 8,
    "cached_repeats": 7,
    "product_id": "newera-v3-lowres",
    "spectral_bins": 64
  },
  "dtype": "float64",
  "host_peak_memory_bytes": 13877111,
  "jax_backend": "cpu",
  "memory_scope": "Python host allocations measured by tracemalloc",
  "output_shape": [
    64
  ],
  "schema_version": 1,
  "timings_seconds": {
    "batched_evaluation": 0.08381337486207485,
    "cached_evaluation_median": 6.875023245811462e-06,
    "first_jit_evaluation": 0.07296716584824026,
    "host_preparation": 0.943114958005026
  }
}
```
