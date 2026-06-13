"""
Tests for dialact_eval.core — context, language, translation.

These tests are pure (no I/O, no API calls) where possible.
LLM integration tests are marked with @pytest.mark.integration.
"""

import pytest
from core.context import CallContext, build_system_prompt
from core.language import (
    EvalLanguageModel,
    _strip_control_tokens,
    is_suppressed_token,
    is_farewell,
)
from core.translation import extract_speech_text


# =============================================================================
# CallContext
# =============================================================================

def test_call_context_requires_goal():
    with pytest.raises(Exception):
        CallContext(goal="")


def test_call_context_defaults():
    ctx = CallContext(goal="Book a flight")
    assert ctx.agent_role == "a professional assistant"
    assert ctx.agent_tone == "friendly and concise"
    assert ctx.constraints == []


def test_call_context_from_yaml(tmp_path):
    yaml_content = "goal: Cancel my reservation\nagent_name: Alex\n"
    f = tmp_path / "ctx.yaml"
    f.write_text(yaml_content)
    ctx = CallContext.from_yaml(f)
    assert ctx.goal == "Cancel my reservation"
    assert ctx.agent_name == "Alex"


def test_build_system_prompt_contains_goal():
    ctx = CallContext(goal="Find out the flight status")
    prompt = build_system_prompt(ctx, tools=True)
    assert "Find out the flight status" in prompt


def test_build_system_prompt_constraints():
    ctx = CallContext(
        goal="Update address",
        constraints=["Do not ask for SSN", "Always confirm changes"],
    )
    prompt = build_system_prompt(ctx)
    assert "Do not ask for SSN" in prompt
    assert "Always confirm changes" in prompt


def test_build_system_prompt_no_tools_uses_tags():
    ctx = CallContext(goal="Check balance")
    prompt = build_system_prompt(ctx, tools=False)
    assert "[HANGUP]" in prompt


# =============================================================================
# Token suppression
# =============================================================================

def test_suppressed_token_dtmf_tag():
    assert is_suppressed_token("[DTMF:2]")


def test_suppressed_token_hold():
    assert is_suppressed_token("[HOLD_CONTINUE]")


def test_suppressed_token_function_call():
    assert is_suppressed_token("press_dtmf")
    assert is_suppressed_token("signal_hold_continue")


def test_non_suppressed_token():
    assert not is_suppressed_token("Hello, how can I help?")
    assert not is_suppressed_token("Great,")


def test_farewell_detection():
    assert is_farewell("Great, goodbye!")
    assert is_farewell("Farewell and thank you.")
    assert not is_farewell("I'll hold for a moment.")


# =============================================================================
# Control token stripping
# =============================================================================

def test_strip_control_tokens_dtmf():
    raw = "I'll press [DTMF:2] for customer service."
    result = _strip_control_tokens(raw)
    assert "[DTMF:2]" not in result
    assert "customer service" in result


def test_strip_control_tokens_hangup():
    raw = "Great, goodbye! [HANGUP]"
    result = _strip_control_tokens(raw)
    assert "[HANGUP]" not in result
    assert "goodbye" in result


def test_strip_control_tokens_hold():
    raw = "signal_hold_continue()"
    result = _strip_control_tokens(raw)
    assert result == ""


# =============================================================================
# Translation extract_speech_text
# =============================================================================

def test_extract_speech_text_removes_dtmf():
    text = "[DTMF:1]"
    assert extract_speech_text(text) == ""


def test_extract_speech_text_keeps_speech():
    text = "Hello, I'm calling about your reservation."
    assert extract_speech_text(text) == text


def test_extract_speech_text_mixed():
    text = "signal_hold_continue(), Please wait."
    result = extract_speech_text(text)
    assert "Please wait" in result
    assert "signal_hold_continue" not in result


# =============================================================================
# EvalLanguageModel (integration — requires GROQ_API_KEY)
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_eval_language_model_generates_response():
    """Integration test: requires GROQ_API_KEY in environment."""
    model = EvalLanguageModel(goal="Find out if the restaurant is open on Sundays")
    result = await model.generate("[CALL_STARTED]")
    assert result.text.strip(), "Expected non-empty response"
    assert not result.hangup, "Should not hang up on opening"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_eval_language_model_multi_turn():
    """Integration test: multi-turn conversation history is maintained."""
    model = EvalLanguageModel(goal="Cancel my dentist appointment")
    result1 = await model.generate("[CALL_STARTED]")
    assert result1.text.strip()
    assert len(model.history) > 0

    result2 = await model.generate("Hi, this is Dr Smith's office. How can I help?")
    assert result2.text.strip()
    assert len(model.history) > len([]) + 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_eval_language_model_stream():
    """Integration test: token streaming via async generator."""
    model = EvalLanguageModel(goal="Ask about the hotel check-in time")
    tokens = []
    async for token in model.token_stream("[CALL_STARTED]"):
        tokens.append(token)
    assert tokens, "Expected at least one token"
    assert "".join(tokens).strip()
