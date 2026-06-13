"""
metrics.py — Custom deepeval metrics for voice agent response evaluation.

These metrics evaluate the text generation quality of the voice agent,
not the overall voice experience. They focus on:

  1. GoalAdherenceMetric      — Does the agent pursue and accomplish the stated goal?
  2. ConversationalToneMetric — Is the response natural, concise, and non-robotic?
  3. CallProtocolMetric       — Does the agent follow call protocol (greeting, confirmation, hangup)?
  4. SuccessPhrasesMetric     — Does the response contain required success phrases?
  5. ScopeAdherenceMetric     — Does the agent avoid asking for out-of-scope information?

Usage:
    from dialact_eval.eval.metrics import GoalAdherenceMetric, CallProtocolMetric
    metric = GoalAdherenceMetric(threshold=0.7)
    metric.measure(test_case)
"""

from __future__ import annotations

import re
from typing import Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ConversationalTestCase


class GoalAdherenceMetric(BaseMetric):
    """
    Evaluates whether the agent's response actively pursues the stated call goal.

    Uses an LLM judge to score how well the agent's output advances its goal
    given the conversational context.
    """

    def __init__(self, threshold: float = 0.7, model: Optional[str] = None):
        self.threshold = threshold
        self.model = model or "gpt-4o-mini"
        self.score = 0.0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Goal Adherence"

    def measure(self, test_case: LLMTestCase) -> float:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        from deepeval.models import GPTModel
        from deepeval.scorer import Scorer

        goal = (test_case.metadata or {}).get("goal", "")
        if not goal:
            self.score = 1.0
            self.reason = "No goal specified — vacuously passes."
            self.success = True
            return self.score

        prompt = f"""You are evaluating an AI phone agent's response quality.

Goal: {goal}
Input (what the other party said): {test_case.input}
Agent response: {test_case.actual_output}

Score the agent's response from 0.0 to 1.0 based on:
- Does the response actively work toward the stated goal? (weight: 0.5)
- Is the response appropriate given what the other party said? (weight: 0.3)
- Does the response avoid straying off-topic or adding unnecessary steps? (weight: 0.2)

Respond with JSON: {{"score": <float 0-1>, "reason": "<one sentence>"}}"""

        model = GPTModel(model=self.model)
        resp = await model.a_generate(prompt)

        import json
        try:
            data = json.loads(resp)
            self.score = float(data.get("score", 0.5))
            self.reason = data.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract score with regex
            m = re.search(r'"score"\s*:\s*([0-9.]+)', resp)
            self.score = float(m.group(1)) if m else 0.5
            self.reason = "Could not parse LLM judge response"

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success


class ConversationalToneMetric(BaseMetric):
    """
    Evaluates whether the response is conversational (not robotic or overly formal).

    Checks for:
    - No markdown formatting (bullets, headers)
    - Appropriate length (not too verbose for spoken audio)
    - Natural phrasing
    - No filler padding ("Certainly!", "Of course!", etc.)
    """

    # Patterns that indicate non-conversational responses
    _MARKDOWN_RE = re.compile(r'[*#`]|\n[-*]\s|\n\d+\.\s')
    _FILLER_RE = re.compile(
        r'\b(certainly|of course|absolutely|great question|sure thing|no problem)\b',
        re.IGNORECASE,
    )
    _MAX_WORDS = 120  # spoken responses should be concise

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Conversational Tone"

    def measure(self, test_case: LLMTestCase) -> float:
        response = test_case.actual_output or ""
        score = 1.0
        issues = []

        if self._MARKDOWN_RE.search(response):
            score -= 0.4
            issues.append("contains markdown formatting")

        word_count = len(response.split())
        if word_count > self._MAX_WORDS:
            penalty = min(0.3, (word_count - self._MAX_WORDS) / self._MAX_WORDS * 0.3)
            score -= penalty
            issues.append(f"too verbose ({word_count} words, max {self._MAX_WORDS})")

        filler_matches = self._FILLER_RE.findall(response)
        if filler_matches:
            score -= 0.15
            issues.append(f"contains filler phrases: {filler_matches}")

        self.score = max(0.0, score)
        self.reason = "Passed" if not issues else "; ".join(issues)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class CallProtocolMetric(BaseMetric):
    """
    Evaluates whether the agent follows the expected call protocol for its turn type.

    Protocol rules checked:
    - Opening turn ([CALL_STARTED]): must include introduction + purpose
    - Confirmation turn: must include appropriate confirmation phrase
    - Closing turn: must include goodbye and hangup signal (or signal_hangup tool)
    - Hold turns: must NOT include speech (only tool calls or [HOLD_CONTINUE])

    Turn type is inferred from metadata["turn_type"] or from the input.
    """

    _HANGUP_RE = re.compile(r'signal_hangup|\[HANGUP\]', re.IGNORECASE)
    _HOLD_CONTINUE_RE = re.compile(r'signal_hold_continue|\[HOLD_CONTINUE\]', re.IGNORECASE)
    _GOODBYE_RE = re.compile(r'\b(goodbye|bye|farewell|take care)\b', re.IGNORECASE)

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Call Protocol"

    def measure(self, test_case: LLMTestCase) -> float:
        response = test_case.actual_output or ""
        input_text = test_case.input or ""
        meta = test_case.metadata or {}
        turn_type = meta.get("turn_type", "")

        # Infer turn type from input if not provided
        if not turn_type:
            if "[CALL_STARTED]" in input_text:
                turn_type = "opening"
            elif "[HOLD_CHECK]" in input_text or "[IVR]" in input_text:
                turn_type = "hold"
            elif meta.get("is_closing"):
                turn_type = "closing"

        if turn_type == "opening":
            # Must introduce and state purpose — check for a greeting
            has_content = len(response.strip()) > 10
            self.score = 1.0 if has_content else 0.0
            self.reason = "Opening response has content" if has_content else "Opening response is empty"

        elif turn_type == "hold":
            # Hold response should NOT have speech text
            speech_only = re.sub(r'signal_hold[a-z_]*\([^)]*\)|\[HOLD[_A-Z]*\]', '', response).strip()
            if not speech_only:
                self.score = 1.0
                self.reason = "Correct: no speech during hold"
            else:
                self.score = 0.0
                self.reason = f"Spoke during hold: {speech_only[:50]!r}"

        elif turn_type == "closing":
            # Closing must have goodbye + hangup signal
            has_goodbye = bool(self._GOODBYE_RE.search(response))
            has_hangup = bool(self._HANGUP_RE.search(response))
            self.score = (0.5 if has_goodbye else 0.0) + (0.5 if has_hangup else 0.0)
            parts = []
            if not has_goodbye:
                parts.append("missing goodbye phrase")
            if not has_hangup:
                parts.append("missing hangup signal")
            self.reason = "Correct closing" if not parts else "; ".join(parts)

        else:
            # General turn — just check it has content
            has_content = len(response.strip()) > 5
            self.score = 1.0 if has_content else 0.3
            self.reason = "Response has content" if has_content else "Response is too short"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class SuccessPhrasesMetric(BaseMetric):
    """
    Checks whether required success phrases appear in the agent's response.

    Expected phrases are provided via metadata["required_phrases"].
    All phrases must appear (case-insensitive) for the metric to pass.

    Example:
        test_case.metadata = {
            "required_phrases": ["successfully cancelled", "reservation"]
        }
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Success Phrases"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.metadata or {}
        required = meta.get("required_phrases", [])

        if not required:
            self.score = 1.0
            self.reason = "No required phrases specified — vacuously passes."
            self.success = True
            return self.score

        response_lower = (test_case.actual_output or "").lower()
        missing = [p for p in required if p.lower() not in response_lower]
        found = [p for p in required if p.lower() in response_lower]

        self.score = len(found) / len(required)
        if missing:
            self.reason = f"Missing phrases: {missing}"
        else:
            self.reason = f"All {len(required)} required phrases present"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class ScopeAdherenceMetric(BaseMetric):
    """
    Checks whether the agent asks for information NOT mentioned in the goal.

    The agent should only ask for information explicitly required by the goal.
    This catches cases where the agent over-asks (e.g., requesting account numbers
    when the goal doesn't mention verification).

    Disallowed questions are detected heuristically; override with
    metadata["disallowed_asks"] for scenario-specific rules.
    """

    # Generic over-asking patterns
    _OVER_ASK_RE = re.compile(
        r'\b(can you (provide|give|confirm|verify|share)|'
        r'what is your (account|id|number|password|pin|dob|date of birth)|'
        r'could you (verify|confirm) your|'
        r'may i (have|get) your (name|account|id|number))\b',
        re.IGNORECASE,
    )

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Scope Adherence"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.metadata or {}
        response = test_case.actual_output or ""
        goal = meta.get("goal", "")

        # User-specified disallowed asks take priority
        disallowed = meta.get("disallowed_asks", [])
        if disallowed:
            violations = [d for d in disallowed if d.lower() in response.lower()]
            self.score = 1.0 - (len(violations) / len(disallowed))
            self.reason = f"Violations: {violations}" if violations else "No disallowed asks found"
            self.success = self.score >= self.threshold
            return self.score

        # Heuristic check: flag generic over-asking patterns
        matches = self._OVER_ASK_RE.findall(response)
        if not matches:
            self.score = 1.0
            self.reason = "No over-asking detected"
        else:
            # Check if the matched ask is justified by the goal
            goal_lower = goal.lower()
            unjustified = []
            for m in matches:
                m_str = m if isinstance(m, str) else " ".join(m)
                # Simple heuristic: if key words from the ask appear in the goal, it's ok
                words = re.findall(r'\w+', m_str.lower())
                if not any(w in goal_lower for w in words if len(w) > 4):
                    unjustified.append(m_str)

            if unjustified:
                self.score = max(0.0, 1.0 - 0.3 * len(unjustified))
                self.reason = f"Potentially out-of-scope asks: {unjustified}"
            else:
                self.score = 0.9
                self.reason = "Asks appear justified by goal"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success
