"""Focused contracts for the observed Phase B replay-memory harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[2]
RUNNER_PATH = ROOT / "scripts/measure_quad_multidim_memory.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("quad_multidim_memory", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_campaign_grid_and_materiality_threshold():
    runner = _load_runner()
    assert runner.DIMENSIONS == (2, 4, 8, 16)
    assert runner.LEVELS == (8, 12, 16)
    assert runner.MATERIAL_INCREMENT_BYTES == 10 * 1024**3
    assert len(runner._cases()) == 96


def test_randomized_array_payload_is_explicitly_contract_unsupported():
    runner = _load_runner()
    reason = runner._unsupported_case("scrambled_sobol", "array")
    assert reason is not None
    assert runner._unsupported_case("sobol", "array") is None


def test_peak_rss_parser_and_paired_materiality():
    runner = _load_runner()
    assert runner._parse_rss("  1234  maximum resident set size") == 1234
    base = {
        "dimension": 16,
        "formula": "sobol",
        "level": 16,
        "payload": "scalar",
        "outcome": "measured",
    }
    pairs = runner._paired_materiality(
        [
            {**base, "mode": "primal", "peak_rss_bytes": 1_000},
            {
                **base,
                "mode": "replay_gradient",
                "peak_rss_bytes": 1_000 + runner.MATERIAL_INCREMENT_BYTES,
            },
        ]
    )
    assert pairs[0]["material"]
