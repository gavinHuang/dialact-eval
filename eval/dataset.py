"""
dataset.py — Load eval scenarios and produce deepeval test cases.

Supports two scenario formats (both from voice-agent's eval/scenarios/):

1. Single-agent scenarios (two_agent_*.yaml):
   Each scenario has caller + answerer goals. We simulate a multi-turn conversation
   and produce a ConversationalTestCase.

2. Edge case scenarios (synthesized_edge_cases.yaml):
   Each scenario has a single agent + a sequence of scripted callee turns.
   Produces individual LLMTestCase per turn.

Usage:
    from dialact_eval.eval.dataset import load_scenario_dataset
    test_cases = load_scenario_dataset("path/to/scenarios.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class AgentConfig:
    goal: str
    identity: Optional[str] = None
    context: Optional[str] = None
    opening_line: Optional[str] = None


@dataclass
class TurnScript:
    """A scripted turn for edge case evaluation."""
    callee_says: str
    expected_phrases: List[str] = field(default_factory=list)
    disallowed_phrases: List[str] = field(default_factory=list)
    turn_type: str = ""          # "opening", "hold", "closing", or "" for general


@dataclass
class EvalScenario:
    """Unified scenario representation for the eval runner."""
    id: str
    description: str
    difficulty: str = "medium"
    goal: str = ""
    agent: Optional[AgentConfig] = None
    answerer: Optional[AgentConfig] = None
    script: List[TurnScript] = field(default_factory=list)
    required_phrases: List[str] = field(default_factory=list)
    max_turns: int = 50
    timeout: int = 120


# =============================================================================
# LOADERS
# =============================================================================

def load_scenario_dataset(path: str | Path) -> List[EvalScenario]:
    """
    Load scenarios from a YAML file.

    Supports:
    - Two-agent format (caller/answerer)
    - Edge case format (agent + script)
    - Legacy IVR format (agent + success_criteria)
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    scenarios = []
    for raw in data.get("scenarios", []):
        scenario = _parse_scenario(raw)
        if scenario:
            scenarios.append(scenario)

    return scenarios


def _parse_scenario(raw: dict) -> Optional[EvalScenario]:
    scenario_id = raw.get("id", "unknown")
    description = raw.get("description", "")
    difficulty = raw.get("difficulty", "medium")

    # Two-agent format
    if "caller" in raw and "answerer" in raw:
        caller_raw = raw["caller"]
        answerer_raw = raw["answerer"]
        criteria = raw.get("success_criteria", {}) or {}

        caller = AgentConfig(
            goal=caller_raw.get("goal", ""),
            identity=caller_raw.get("identity"),
            context=caller_raw.get("context"),
        )
        answerer = AgentConfig(
            goal=answerer_raw.get("goal", ""),
            opening_line=answerer_raw.get("opening_line"),
        )
        return EvalScenario(
            id=scenario_id,
            description=description,
            difficulty=difficulty,
            agent=caller,
            answerer=answerer,
            required_phrases=criteria.get("goal_phrases", []),
            max_turns=criteria.get("max_turns", 50),
            timeout=raw.get("timeout", 120),
        )

    # Edge case / scripted format
    if "script" in raw:
        agent_raw = raw.get("agent", {}) or {}
        agent = AgentConfig(
            goal=agent_raw.get("goal", raw.get("goal", "")),
            identity=agent_raw.get("identity"),
        )
        script = [
            TurnScript(
                callee_says=t.get("callee_says", ""),
                expected_phrases=t.get("expected_phrases", []),
                disallowed_phrases=t.get("disallowed_phrases", []),
                turn_type=t.get("turn_type", ""),
            )
            for t in raw.get("script", [])
        ]
        criteria = raw.get("success_criteria", {}) or {}
        return EvalScenario(
            id=scenario_id,
            description=description,
            difficulty=difficulty,
            goal=agent.goal,
            agent=agent,
            script=script,
            required_phrases=criteria.get("transcript_contains", []),
            timeout=raw.get("timeout", 30),
        )

    # Legacy IVR / single-agent format
    if "agent" in raw:
        agent_raw = raw.get("agent", {}) or {}
        goal = agent_raw.get("goal", "")
        if agent_raw.get("identity"):
            goal = f"You are {agent_raw['identity']}. {goal}"
        criteria = raw.get("success_criteria", {}) or {}
        return EvalScenario(
            id=scenario_id,
            description=description,
            difficulty=difficulty,
            goal=goal,
            agent=AgentConfig(goal=goal),
            required_phrases=criteria.get("transcript_contains", []),
            timeout=raw.get("timeout", 30),
        )

    return None


# =============================================================================
# TEST CASE BUILDERS
# =============================================================================

def scenario_to_test_cases(
    scenario: EvalScenario,
    conversation: List[dict],
) -> List[LLMTestCase]:
    """
    Convert a completed conversation (list of {"role": str, "text": str} dicts)
    into deepeval LLMTestCase objects — one per agent turn.

    Args:
        scenario: The source EvalScenario.
        conversation: Bilateral transcript from the eval run.

    Returns:
        One LLMTestCase per agent ("caller" role) turn.
    """
    test_cases = []
    agent_turns = [t for t in conversation if t["role"] == "caller"]
    other_turns = [t for t in conversation if t["role"] != "caller"]

    for i, agent_turn in enumerate(agent_turns):
        # The "input" is what the other party said just before this agent turn
        input_text = other_turns[i]["text"] if i < len(other_turns) else "[CALL_STARTED]"

        test_cases.append(LLMTestCase(
            input=input_text,
            actual_output=agent_turn["text"],
            metadata={
                "goal": scenario.goal or (scenario.agent.goal if scenario.agent else ""),
                "scenario_id": scenario.id,
                "turn_index": i,
                "required_phrases": scenario.required_phrases if i == len(agent_turns) - 1 else [],
            },
        ))

    return test_cases


def scenario_to_conversational_test_case(
    scenario: EvalScenario,
    conversation: List[dict],
) -> ConversationalTestCase:
    """
    Convert a completed conversation into a deepeval ConversationalTestCase.
    """
    turns = []
    for turn in conversation:
        role = "user" if turn["role"] != "caller" else "assistant"
        turns.append(Turn(role=role, content=turn["text"]))

    goal = scenario.goal or (scenario.agent.goal if scenario.agent else "")

    return ConversationalTestCase(
        turns=turns,
        metadata={
            "goal": goal,
            "scenario_id": scenario.id,
            "required_phrases": scenario.required_phrases,
            "difficulty": scenario.difficulty,
        },
    )
