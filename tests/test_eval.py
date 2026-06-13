"""
Tests for dialact_eval.eval — metrics, dataset loading, and runner.
"""

import pytest
from deepeval.test_case import LLMTestCase

from eval.metrics import (
    ConversationalToneMetric,
    CallProtocolMetric,
    SuccessPhrasesMetric,
    ScopeAdherenceMetric,
)
from eval.dataset import (
    load_scenario_dataset,
    EvalScenario,
    AgentConfig,
    TurnScript,
    scenario_to_test_cases,
)


# =============================================================================
# ConversationalToneMetric
# =============================================================================

def test_tone_passes_clean_response():
    metric = ConversationalToneMetric(threshold=0.7)
    tc = LLMTestCase(
        input="Hello?",
        actual_output="Hi, I'm calling about my appointment scheduled for next Tuesday.",
    )
    metric.measure(tc)
    assert metric.is_successful()


def test_tone_fails_markdown():
    metric = ConversationalToneMetric(threshold=0.7)
    tc = LLMTestCase(
        input="What can I do for you?",
        actual_output="Here are the options:\n- Option 1: Cancel\n- Option 2: Reschedule",
    )
    metric.measure(tc)
    assert not metric.is_successful()


def test_tone_penalizes_filler():
    metric = ConversationalToneMetric(threshold=0.9)
    tc = LLMTestCase(
        input="Hi",
        actual_output="Certainly! Of course, I'd be happy to help you today.",
    )
    metric.measure(tc)
    assert metric.score < 1.0


def test_tone_verbose_response():
    metric = ConversationalToneMetric(threshold=0.7)
    words = " ".join(["word"] * 150)
    tc = LLMTestCase(input="Hi", actual_output=words)
    metric.measure(tc)
    assert metric.score < 1.0


# =============================================================================
# CallProtocolMetric
# =============================================================================

def test_protocol_opening_has_content():
    metric = CallProtocolMetric(threshold=0.7)
    tc = LLMTestCase(
        input="[CALL_STARTED]",
        actual_output="Hi, this is Alex calling about your reservation.",
        metadata={"turn_type": "opening"},
    )
    metric.measure(tc)
    assert metric.is_successful()


def test_protocol_opening_empty_fails():
    metric = CallProtocolMetric(threshold=0.7)
    tc = LLMTestCase(
        input="[CALL_STARTED]",
        actual_output="",
        metadata={"turn_type": "opening"},
    )
    metric.measure(tc)
    assert not metric.is_successful()


def test_protocol_hold_no_speech():
    metric = CallProtocolMetric(threshold=0.7)
    tc = LLMTestCase(
        input="[HOLD_CHECK]",
        actual_output="signal_hold_continue()",
        metadata={"turn_type": "hold"},
    )
    metric.measure(tc)
    assert metric.is_successful()


def test_protocol_hold_with_speech_fails():
    metric = CallProtocolMetric(threshold=0.7)
    tc = LLMTestCase(
        input="[HOLD_CHECK]",
        actual_output="Please wait, I'm still on hold.",
        metadata={"turn_type": "hold"},
    )
    metric.measure(tc)
    assert not metric.is_successful()


def test_protocol_closing_requires_goodbye_and_hangup():
    metric = CallProtocolMetric(threshold=0.7)
    tc = LLMTestCase(
        input="Sure, no problem.",
        actual_output="Great, goodbye! signal_hangup()",
        metadata={"turn_type": "closing"},
    )
    metric.measure(tc)
    assert metric.is_successful()


# =============================================================================
# SuccessPhrasesMetric
# =============================================================================

def test_success_phrases_all_present():
    metric = SuccessPhrasesMetric(threshold=1.0)
    tc = LLMTestCase(
        input="Can you cancel my reservation?",
        actual_output="Your reservation has been successfully cancelled. That's all done and taken care of for you.",
        metadata={"required_phrases": ["successfully cancelled", "reservation"]},
    )
    metric.measure(tc)
    assert metric.is_successful()
    assert metric.score == 1.0


def test_success_phrases_partial_fails():
    metric = SuccessPhrasesMetric(threshold=1.0)
    # Response has "reservation" but not "successfully cancelled" — 1 of 2
    tc = LLMTestCase(
        input="Cancel please",
        actual_output="Your reservation has been cancelled.",
        metadata={"required_phrases": ["successfully cancelled", "reservation"]},
    )
    metric.measure(tc)
    assert not metric.is_successful()
    assert 0.0 < metric.score < 1.0


def test_success_phrases_vacuous():
    metric = SuccessPhrasesMetric()
    tc = LLMTestCase(
        input="Hi",
        actual_output="Hello there.",
        metadata={},
    )
    metric.measure(tc)
    assert metric.is_successful()


# =============================================================================
# ScopeAdherenceMetric
# =============================================================================

def test_scope_no_over_asking():
    metric = ScopeAdherenceMetric(threshold=0.8)
    tc = LLMTestCase(
        input="Hi, I'd like to cancel my appointment.",
        actual_output="Of course, I can help with that. Which appointment would you like to cancel?",
        metadata={"goal": "Cancel the appointment"},
    )
    metric.measure(tc)
    assert metric.is_successful()


def test_scope_disallowed_asks():
    metric = ScopeAdherenceMetric(threshold=0.8)
    tc = LLMTestCase(
        input="Hi",
        actual_output="Can you provide your account number and date of birth?",
        metadata={
            "goal": "Cancel appointment",
            "disallowed_asks": ["account number", "date of birth"],
        },
    )
    metric.measure(tc)
    assert not metric.is_successful()


# =============================================================================
# Dataset loading
# =============================================================================

def test_load_scenario_dataset(tmp_path):
    yaml_content = """
scenarios:
  - id: test-001
    description: Simple cancellation
    difficulty: easy
    caller:
      goal: Cancel my flight reservation
      identity: John Smith
    answerer:
      goal: You are an airline agent. Help callers with their requests.
      opening_line: Thank you for calling AirCo. How can I help?
    success_criteria:
      goal_phrases:
        - successfully cancelled
        - reservation
      max_turns: 10
"""
    f = tmp_path / "scenarios.yaml"
    f.write_text(yaml_content)

    scenarios = load_scenario_dataset(f)
    assert len(scenarios) == 1
    assert scenarios[0].id == "test-001"
    assert scenarios[0].difficulty == "easy"
    assert scenarios[0].answerer is not None
    assert "Cancel my flight" in scenarios[0].agent.goal
    assert "successfully cancelled" in scenarios[0].required_phrases


def test_load_scripted_scenario(tmp_path):
    yaml_content = """
scenarios:
  - id: edge-001
    description: Test opening line
    agent:
      goal: Check if the pharmacy is open on weekends
    script:
      - callee_says: "Thank you for calling City Pharmacy."
        expected_phrases: []
        turn_type: opening
      - callee_says: "We are open Saturday 9am-5pm, closed Sunday."
        expected_phrases: ["Saturday", "Sunday"]
    success_criteria:
      transcript_contains:
        - weekend
"""
    f = tmp_path / "edge.yaml"
    f.write_text(yaml_content)

    scenarios = load_scenario_dataset(f)
    assert len(scenarios) == 1
    assert len(scenarios[0].script) == 2
    assert scenarios[0].script[0].turn_type == "opening"


def test_scenario_to_test_cases():
    scenario = EvalScenario(
        id="test",
        description="test",
        goal="Cancel flight",
        required_phrases=["successfully cancelled"],
    )
    conversation = [
        {"role": "answerer", "text": "Thank you for calling."},
        {"role": "caller", "text": "Hi, I'd like to cancel my flight."},
        {"role": "answerer", "text": "Sure, which booking?"},
        {"role": "caller", "text": "Your reservation has been successfully cancelled."},
    ]
    test_cases = scenario_to_test_cases(scenario, conversation)
    assert len(test_cases) == 2
    assert test_cases[0].input == "Thank you for calling."
    assert test_cases[0].actual_output == "Hi, I'd like to cancel my flight."
    # Last turn should have required phrases
    assert "successfully cancelled" in test_cases[-1].metadata["required_phrases"]
