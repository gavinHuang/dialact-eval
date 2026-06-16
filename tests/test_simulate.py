"""
Tests for ivr/simulate.py — local IVR walk engine.

These tests run with no LLM (the LLM is unavailable so the fallback "press 1"
heuristic is used) which keeps them fast and dependency-free.
"""

from __future__ import annotations

import pytest
import yaml

from ivr.config import parse_config
from ivr.simulate import IVRSimulator, flow_to_graph, _extract_digit


# ---------------------------------------------------------------------------
# _extract_digit
# ---------------------------------------------------------------------------

def test_extract_digit_bare():
    assert _extract_digit("2") == "2"

def test_extract_digit_dtmf_token():
    assert _extract_digit("[DTMF:5]") == "5"

def test_extract_digit_in_sentence():
    assert _extract_digit("I would press 3 to reach billing.") == "3"

def test_extract_digit_star():
    assert _extract_digit("*") == "*"

def test_extract_digit_fallback():
    assert _extract_digit("no digits here") == "1"


# ---------------------------------------------------------------------------
# flow_to_graph
# ---------------------------------------------------------------------------

SIMPLE_YAML = """
name: Simple
start: welcome
nodes:
  welcome:
    type: say
    say: "Welcome."
    next: menu
  menu:
    type: menu
    say: "Press 1 or 2."
    gather:
      timeout: 5
      num_digits: 1
    routes:
      "1": end
      "2": end
    default: menu
  end:
    type: hangup
"""

def test_flow_to_graph_nodes():
    config = parse_config(yaml.safe_load(SIMPLE_YAML))
    graph = flow_to_graph(config)
    node_ids = {n["id"] for n in graph["nodes"]}
    assert node_ids == {"welcome", "menu", "end"}
    assert graph["start"] == "welcome"
    assert graph["name"] == "Simple"

def test_flow_to_graph_edges():
    config = parse_config(yaml.safe_load(SIMPLE_YAML))
    graph = flow_to_graph(config)
    edge_pairs = [(e["from"], e["to"]) for e in graph["edges"]]
    assert ("welcome", "menu") in edge_pairs
    assert ("menu", "end") in edge_pairs


# ---------------------------------------------------------------------------
# IVRSimulator — offline (LLM falls back to "press 1")
# ---------------------------------------------------------------------------

MENU_YAML = """
name: Menu Flow
start: root
nodes:
  root:
    type: menu
    say: "Press 1 for A. Press 2 for B."
    gather:
      timeout: 5
      num_digits: 1
    routes:
      "1": branch_a
      "2": branch_b
    default: root
  branch_a:
    type: say
    say: "You chose A."
    next: done
  branch_b:
    type: say
    say: "You chose B."
    next: done
  done:
    type: hangup
"""

@pytest.mark.asyncio
async def test_simulator_navigates_to_hangup():
    """Simulator should reach hangup with fallback digit '1'."""
    config = parse_config(yaml.safe_load(MENU_YAML))
    # Use a bogus URL so LLM is unavailable → fallback digit "1"
    sim = IVRSimulator(config, goal="reach branch A", voice_agent_url="http://localhost:0")
    steps = []
    async for step in sim.run():
        steps.append(step)

    node_ids = [s.node_id for s in steps]
    assert "root" in node_ids
    assert "branch_a" in node_ids
    assert "done" in node_ids
    assert steps[-1].node_type == "hangup"


@pytest.mark.asyncio
async def test_simulator_records_digits():
    """Menu steps should record the digit pressed."""
    config = parse_config(yaml.safe_load(MENU_YAML))
    sim = IVRSimulator(config, goal="reach branch A", voice_agent_url="http://localhost:0")
    steps = []
    async for step in sim.run():
        steps.append(step)

    menu_steps = [s for s in steps if s.node_type == "menu"]
    assert len(menu_steps) == 1
    assert menu_steps[0].digit == "1"


@pytest.mark.asyncio
async def test_simulator_say_nodes_have_speech():
    config = parse_config(yaml.safe_load(MENU_YAML))
    sim = IVRSimulator(config, goal="go anywhere", voice_agent_url="http://localhost:0")
    steps = []
    async for step in sim.run():
        steps.append(step)

    say_steps = [s for s in steps if s.node_type == "say"]
    assert all(s.speech for s in say_steps)


SAY_CHAIN_YAML = """
name: Say Chain
start: a
nodes:
  a:
    type: say
    say: "First."
    next: b
  b:
    type: say
    say: "Second."
    next: c
  c:
    type: hangup
"""

@pytest.mark.asyncio
async def test_simulator_walks_say_chain():
    """All say nodes should be visited in order."""
    config = parse_config(yaml.safe_load(SAY_CHAIN_YAML))
    sim = IVRSimulator(config, goal="listen", voice_agent_url="http://localhost:0")
    steps = []
    async for step in sim.run():
        steps.append(step)

    assert [s.node_id for s in steps] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_simulator_max_steps_safety():
    """max_steps should prevent infinite loops."""
    # self-loop menu (default points back to itself, no digit "1" route elsewhere)
    loop_yaml = """
name: Loop
start: loop
nodes:
  loop:
    type: menu
    say: "Press 2 to exit."
    gather:
      timeout: 5
      num_digits: 1
    routes:
      "2": done
    default: loop
  done:
    type: hangup
"""
    config = parse_config(yaml.safe_load(loop_yaml))
    # LLM offline → fallback digit "1" which isn't in routes → default → loop
    sim = IVRSimulator(config, goal="exit", voice_agent_url="http://localhost:0", max_steps=5)
    steps = []
    async for step in sim.run():
        steps.append(step)

    assert len(steps) <= 5
