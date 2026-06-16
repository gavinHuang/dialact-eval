"""
Unit tests for ivr/synthesizer.py.
"""

import yaml
import pytest

from ivr.synthesizer import synthesize, PATTERNS, SynthesisResult
from ivr.config import parse_config


# ── Seed reproducibility ───────────────────────────────────────────────────


def test_same_seed_produces_identical_output():
    for pattern in PATTERNS:
        r1 = synthesize([pattern], seed=42)
        r2 = synthesize([pattern], seed=42)
        assert r1[0].flow_yaml == r2[0].flow_yaml, f"flow_yaml differs for pattern {pattern!r}"
        assert r1[0].scenario_yaml == r2[0].scenario_yaml, f"scenario_yaml differs for pattern {pattern!r}"


def test_different_seeds_produce_different_hold_queue_output():
    r1 = synthesize(["hold-queue"], seed=1)
    r2 = synthesize(["hold-queue"], seed=2)
    # At minimum the flow or scenario should differ (parameters are randomized)
    assert r1[0].flow_yaml != r2[0].flow_yaml or r1[0].scenario_yaml != r2[0].scenario_yaml


# ── Flow YAML validity ─────────────────────────────────────────────────────


@pytest.mark.parametrize("pattern", list(PATTERNS.keys()))
def test_generated_flow_loads_via_parse_config(pattern):
    results = synthesize([pattern], seed=0)
    assert len(results) == 1
    flow_data = yaml.safe_load(results[0].flow_yaml)
    config = parse_config(flow_data)  # must not raise
    assert config.start in config.nodes


# ── Scenario YAML validity ─────────────────────────────────────────────────

_REQUIRED_SCENARIO_KEYS = {"id", "description", "agent", "timeout", "success_criteria"}


@pytest.mark.parametrize("pattern", list(PATTERNS.keys()))
def test_generated_scenario_has_required_keys(pattern):
    results = synthesize([pattern], seed=0)
    scenario_data = yaml.safe_load(results[0].scenario_yaml)
    assert "scenarios" in scenario_data
    for scenario in scenario_data["scenarios"]:
        missing = _REQUIRED_SCENARIO_KEYS - scenario.keys()
        assert not missing, f"Scenario missing keys {missing} for pattern {pattern!r}"


@pytest.mark.parametrize("pattern", list(PATTERNS.keys()))
def test_generated_scenario_timeout_positive(pattern):
    results = synthesize([pattern], seed=0)
    scenario_data = yaml.safe_load(results[0].scenario_yaml)
    for scenario in scenario_data["scenarios"]:
        assert scenario["timeout"] > 0


# ── synthesize() API ───────────────────────────────────────────────────────


def test_synthesize_no_patterns_returns_all():
    results = synthesize(seed=0)
    assert len(results) == len(PATTERNS)
    returned_patterns = {r.pattern for r in results}
    assert returned_patterns == set(PATTERNS.keys())


def test_synthesize_unknown_pattern_raises():
    with pytest.raises(ValueError, match="Unknown pattern"):
        synthesize(["nonexistent-pattern"])


def test_synthesize_returns_synthesis_results():
    results = synthesize(["out-of-hours"], seed=7)
    assert len(results) == 1
    assert isinstance(results[0], SynthesisResult)
    assert results[0].pattern == "out-of-hours"
    assert results[0].flow_yaml
    assert results[0].scenario_yaml


# ── Hold queue: scenario timeout >= total hold duration ────────────────────


def test_hold_queue_scenario_timeout_covers_hold_duration():
    # Run multiple seeds to check the invariant holds across randomizations
    for seed in range(10):
        results = synthesize(["hold-queue"], seed=seed)
        flow_data = yaml.safe_load(results[0].flow_yaml)
        scenario_data = yaml.safe_load(results[0].scenario_yaml)
        # Find hold node
        hold_nodes = [n for n in flow_data["nodes"].values() if n.get("type") == "hold"]
        assert hold_nodes, "hold-queue flow should have a hold node"
        hold_node = hold_nodes[0]
        total_hold = hold_node["repeat"] * hold_node["interval"]
        timeout = scenario_data["scenarios"][0]["timeout"]
        assert timeout >= total_hold, (
            f"seed={seed}: timeout {timeout} < total_hold {total_hold}"
        )
