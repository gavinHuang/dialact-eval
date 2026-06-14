"""
runner.py — Batch evaluation runner using deepeval.

Runs scenarios through EvalLanguageModel to produce conversations,
then evaluates them with deepeval metrics.

Usage:
    # CLI
    dialact-eval eval run eval/scenarios/two_agent_medium.yaml

    # Programmatic
    from dialact_eval.eval.runner import run_eval
    import asyncio
    results = asyncio.run(run_eval("path/to/scenarios.yaml"))
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from core.context import CallContext
from core.language import LLMClient
from eval.dataset import EvalScenario, load_scenario_dataset, scenario_to_test_cases
from eval.metrics import (
    GoalAdherenceMetric,
    ConversationalToneMetric,
    CallProtocolMetric,
    SuccessPhrasesMetric,
    ScopeAdherenceMetric,
)


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass
class TurnEvalResult:
    """Evaluation result for a single agent turn."""
    turn_index: int
    input: str
    output: str
    metrics: dict  # metric_name → score


@dataclass
class ScenarioEvalResult:
    """Full evaluation result for one scenario."""
    scenario_id: str
    description: str
    difficulty: str
    passed: bool
    turns: int
    wall_clock_s: float
    conversation: List[dict]   # [{"role": str, "text": str}]
    turn_results: List[TurnEvalResult]
    error: Optional[str] = None

    @property
    def avg_goal_adherence(self) -> float:
        scores = [t.metrics.get("Goal Adherence", 0.0) for t in self.turn_results]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_tone(self) -> float:
        scores = [t.metrics.get("Conversational Tone", 0.0) for t in self.turn_results]
        return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# CONVERSATION SIMULATION
# =============================================================================

async def _run_two_agent_conversation(
    scenario: EvalScenario,
    max_turns: int = 50,
) -> List[dict]:
    """
    Simulate a two-agent conversation between caller and answerer.

    Returns bilateral transcript: [{"role": "caller"|"answerer", "text": str}]
    """
    if not scenario.agent or not scenario.answerer:
        return []

    # Build caller goal string
    caller_goal = scenario.agent.goal
    if scenario.agent.identity:
        caller_goal = f"You are {scenario.agent.identity}. {caller_goal}"
    if scenario.agent.context:
        caller_goal = f"{caller_goal}\n\nContext: {scenario.agent.context}"

    caller_model = LLMClient(goal=caller_goal)
    answerer_model = LLMClient(goal=scenario.answerer.goal)

    conversation: List[dict] = []
    turn_count = 0

    # Determine who speaks first
    opening = scenario.answerer.opening_line or ""
    if opening:
        conversation.append({"role": "answerer", "text": opening})
        turn_count += 1
        current_input = opening
        current_speaker = "caller"
    else:
        current_input = "[CALL_STARTED]"
        current_speaker = "caller"

    try:
        while turn_count < max_turns:
            try:
                if current_speaker == "caller":
                    result = await caller_model.generate(current_input)
                    text = result.text
                    if not text.strip():
                        break
                    conversation.append({"role": "caller", "text": text})
                    turn_count += 1
                    if result.hangup:
                        break
                    current_input = text
                    current_speaker = "answerer"
                else:
                    result = await answerer_model.generate(current_input)
                    text = result.text
                    if not text.strip():
                        break
                    conversation.append({"role": "answerer", "text": text})
                    turn_count += 1
                    if result.hangup:
                        break
                    current_input = text
                    current_speaker = "caller"
            except Exception:
                break
    finally:
        await caller_model.aclose()
        await answerer_model.aclose()

    return conversation


async def _run_scripted_conversation(
    scenario: EvalScenario,
) -> List[dict]:
    """
    Run the agent against a scripted sequence of callee utterances.

    Returns bilateral transcript.
    """
    if not scenario.agent:
        return []

    goal = scenario.agent.goal
    if scenario.agent.identity:
        goal = f"You are {scenario.agent.identity}. {goal}"

    model = LLMClient(goal=goal)
    conversation: List[dict] = []

    try:
        # Opening turn
        result = await model.generate("[CALL_STARTED]")
        if result.text.strip():
            conversation.append({"role": "caller", "text": result.text})

        for turn in scenario.script:
            callee_text = turn.callee_says
            if callee_text:
                conversation.append({"role": "answerer", "text": callee_text})

            result = await model.generate(callee_text)
            if result.text.strip() or result.dtmf_digits:
                text = result.text or f"[DTMF:{result.dtmf_digits}]"
                conversation.append({"role": "caller", "text": text})

            if result.hangup:
                break
    finally:
        await model.aclose()

    return conversation


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

async def run_scenario(
    scenario: EvalScenario,
    use_deepeval_judge: bool = False,
) -> ScenarioEvalResult:
    """
    Run a single scenario and evaluate it.

    Args:
        scenario: The scenario to run.
        use_deepeval_judge: If True, use LLM-as-judge for GoalAdherenceMetric
                            (requires OpenAI API key). If False, skips that metric.
    """
    start = time.monotonic()
    error: Optional[str] = None
    conversation: List[dict] = []

    try:
        if scenario.script:
            conversation = await _run_scripted_conversation(scenario)
        elif scenario.answerer:
            max_turns = min(scenario.max_turns, 50)
            conversation = await asyncio.wait_for(
                _run_two_agent_conversation(scenario, max_turns=max_turns),
                timeout=scenario.timeout,
            )
        else:
            error = "No script or answerer defined"
    except asyncio.TimeoutError:
        error = f"Timeout after {scenario.timeout}s"
    except Exception as e:
        error = str(e)

    elapsed = time.monotonic() - start

    # Build test cases
    test_cases = scenario_to_test_cases(scenario, conversation)

    # Evaluate each turn
    turn_results = []
    for tc in test_cases:
        metrics_scores: dict = {}

        # Always run deterministic metrics
        tone = ConversationalToneMetric()
        tone.measure(tc)
        metrics_scores["Conversational Tone"] = tone.score

        protocol = CallProtocolMetric()
        protocol.measure(tc)
        metrics_scores["Call Protocol"] = protocol.score

        if tc.metadata.get("required_phrases"):
            phrases = SuccessPhrasesMetric()
            phrases.measure(tc)
            metrics_scores["Success Phrases"] = phrases.score

        scope = ScopeAdherenceMetric()
        scope.measure(tc)
        metrics_scores["Scope Adherence"] = scope.score

        # LLM-as-judge (optional — requires OpenAI API key)
        if use_deepeval_judge:
            try:
                goal_metric = GoalAdherenceMetric()
                await goal_metric.a_measure(tc)
                metrics_scores["Goal Adherence"] = goal_metric.score
            except Exception:
                pass  # Skip if no API key

        turn_results.append(TurnEvalResult(
            turn_index=tc.metadata.get("turn_index", 0),
            input=tc.input,
            output=tc.actual_output,
            metrics=metrics_scores,
        ))

    # Determine overall pass: check required phrases appear in full transcript
    full_text = " ".join(t["text"] for t in conversation if t["role"] == "caller").lower()
    phrases_pass = all(p.lower() in full_text for p in scenario.required_phrases)
    passed = phrases_pass and not error

    return ScenarioEvalResult(
        scenario_id=scenario.id,
        description=scenario.description,
        difficulty=scenario.difficulty,
        passed=passed,
        turns=len(conversation),
        wall_clock_s=elapsed,
        conversation=conversation,
        turn_results=turn_results,
        error=error,
    )


# =============================================================================
# BATCH RUNNER
# =============================================================================

async def run_eval(
    dataset_path: str,
    output_dir: Optional[str] = None,
    use_deepeval_judge: bool = False,
    scenario_filter: Optional[str] = None,
) -> List[ScenarioEvalResult]:
    """
    Load scenarios, run all of them, print a summary, and optionally save reports.

    Args:
        dataset_path: Path to YAML scenario file.
        output_dir: Directory to write JSON + Markdown reports (None = no output).
        use_deepeval_judge: Enable LLM-as-judge scoring (requires OpenAI API key).
        scenario_filter: If set, only run scenarios whose ID contains this string.

    Returns:
        List of ScenarioEvalResult objects.
    """
    scenarios = load_scenario_dataset(dataset_path)
    if scenario_filter:
        scenarios = [s for s in scenarios if scenario_filter.lower() in s.id.lower()]

    results: List[ScenarioEvalResult] = []

    print(f"\nRunning {len(scenarios)} scenarios from {dataset_path}\n")
    print(f"{'ID':<35} {'Difficulty':<10} {'Result':<8} {'Turns':>6} {'Time':>8}")
    print("-" * 72)

    for scenario in scenarios:
        result = await run_scenario(scenario, use_deepeval_judge=use_deepeval_judge)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        error_suffix = f"  [{result.error}]" if result.error else ""
        print(
            f"{result.scenario_id:<35} {result.difficulty:<10} {status:<8} "
            f"{result.turns:>6} {result.wall_clock_s:>7.1f}s{error_suffix}"
        )

    print("-" * 72)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_tone = sum(r.avg_tone for r in results) / total if total else 0.0
    print(f"Pass rate: {passed}/{total} ({passed / total * 100:.0f}%)")
    print(f"Avg conversational tone: {avg_tone:.2f}")

    if output_dir:
        _write_reports(results, dataset_path, output_dir)

    return results


def _write_reports(
    results: List[ScenarioEvalResult],
    dataset_path: str,
    output_dir: str,
) -> None:
    """Write JSON and Markdown reports to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = Path(dataset_path).stem
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    # JSON report
    json_path = out / f"{stem}_{ts}.json"
    serialized = []
    for r in results:
        serialized.append({
            "scenario_id": r.scenario_id,
            "description": r.description,
            "difficulty": r.difficulty,
            "passed": r.passed,
            "turns": r.turns,
            "wall_clock_s": r.wall_clock_s,
            "error": r.error,
            "avg_goal_adherence": r.avg_goal_adherence,
            "avg_tone": r.avg_tone,
            "conversation": r.conversation,
            "turn_metrics": [
                {
                    "turn_index": t.turn_index,
                    "input": t.input,
                    "output": t.output,
                    "metrics": t.metrics,
                }
                for t in r.turn_results
            ],
        })
    with open(json_path, "w") as f:
        json.dump(serialized, f, indent=2)

    # Markdown report
    md_path = out / f"{stem}_{ts}.md"
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)

    rows = "\n".join(
        f"| {r.scenario_id} | {r.difficulty} | {'PASS' if r.passed else 'FAIL'} "
        f"| {r.turns} | {r.wall_clock_s:.1f}s | {r.avg_tone:.2f} |"
        for r in results
    )

    md = (
        f"# Eval Report: {stem}\n\n"
        f"**Run:** {ts}  \n"
        f"**Dataset:** {dataset_path}  \n"
        f"**Pass rate:** {passed_count}/{total} ({passed_count / total * 100:.0f}%)\n\n"
        "## Results\n\n"
        "| Scenario ID | Difficulty | Result | Turns | Time | Avg Tone |\n"
        "|-------------|-----------|--------|-------|------|----------|\n"
        f"{rows}\n\n"
    )

    # Append conversation transcripts for each scenario
    for r in results:
        md += f"### {r.scenario_id}\n\n"
        md += f"*{r.description}*\n\n"
        for turn in r.conversation:
            role_label = "**Agent**" if turn["role"] == "caller" else "Caller"
            md += f"{role_label}: {turn['text']}\n\n"
        md += "---\n\n"

    with open(md_path, "w") as f:
        f.write(md)

    print(f"\nReports written to:\n  {json_path}\n  {md_path}")
