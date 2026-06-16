"""
IVR scenario synthesizer.

Generates IVR flow YAML + matching benchmark scenario YAML for a library of
edge-case patterns. All randomization is seeded for reproducibility.

Public API:
    synthesize(patterns=None, seed=None) -> list[SynthesisResult]

Patterns:
    out-of-hours       — IVR immediately plays a closed message and hangs up
    hold-queue         — After menu selection, caller waits in a hold queue
    human-pickup       — Hold queue followed by unexpected human pickup
    dtmf-timeout-loop  — Menu whose default loops back to itself (stuck IVR)
    menu-repeat-cap    — IVR hangs up after N unanswered menu prompts
"""

from __future__ import annotations

import random
import textwrap
import yaml
from dataclasses import dataclass
from typing import Callable


@dataclass
class SynthesisResult:
    pattern: str
    flow_yaml: str
    scenario_yaml: str


def synthesize(
    patterns: list[str] | None = None,
    seed: int | None = None,
) -> list[SynthesisResult]:
    """Generate SynthesisResult for each requested pattern.

    Args:
        patterns: List of pattern names. If None, all registered patterns are used.
        seed: Random seed for reproducibility.

    Returns:
        List of SynthesisResult, one per pattern.
    """
    rng = random.Random(seed)
    selected = patterns if patterns is not None else list(PATTERNS.keys())
    results = []
    for name in selected:
        if name not in PATTERNS:
            raise ValueError(f"Unknown pattern: {name!r}. Available: {list(PATTERNS)}")
        fn = PATTERNS[name]
        results.append(fn(rng))
    return results


# ── Pattern implementations ────────────────────────────────────────────────

_CLOSED_MESSAGES = [
    "Thank you for calling. Our offices are currently closed.",
    "We are currently closed. Our business hours are Monday through Friday, 9 AM to 5 PM.",
    "Sorry, we are not available right now. Please call back during business hours.",
    "Our office is closed at this time. Please try again later.",
    "We're closed for the day. Please call back tomorrow.",
]

_HOLD_MESSAGES = [
    "Your call is important to us. Please continue to hold.",
    "Thank you for your patience. The next available agent will be with you shortly.",
    "All of our agents are currently busy. Please stay on the line.",
    "We appreciate your patience. Your wait time is approximately five minutes.",
]

_DEPARTMENT_NAMES = ["sales", "support", "billing", "technical support", "customer service"]


def _synth_out_of_hours(rng: random.Random) -> SynthesisResult:
    message = rng.choice(_CLOSED_MESSAGES)
    flow = textwrap.dedent(f"""\
        name: Out-Of-Hours IVR
        start: greeting

        nodes:
          greeting:
            type: out-of-hours
            say: "{message}"
        """)
    scenario = textwrap.dedent(f"""\
        scenarios:
          - id: out-of-hours-closed
            description: "Agent encounters an out-of-hours IVR that hangs up immediately"
            agent:
              goal: "Call this IVR to reach customer support."
              identity: "Customer"
            timeout: 30
            success_criteria:
              transcript_contains:
                - "{message.split('.')[0]}"
              max_turns: 5
        """)
    return SynthesisResult(pattern="out-of-hours", flow_yaml=flow, scenario_yaml=scenario)


def _synth_hold_queue(rng: random.Random) -> SynthesisResult:
    department = rng.choice(_DEPARTMENT_NAMES)
    hold_message = rng.choice(_HOLD_MESSAGES)
    repeat = rng.randint(2, 4)
    interval = rng.choice([5, 8, 10])
    total_hold = repeat * interval
    timeout = total_hold + 60

    flow = textwrap.dedent(f"""\
        name: Hold Queue IVR
        start: welcome

        nodes:
          welcome:
            type: say
            say: "Welcome. Press 1 to reach {department}."
            next: main_menu

          main_menu:
            type: menu
            say: "Press 1 for {department}."
            gather:
              timeout: 5
              num_digits: 1
            routes:
              "1": queue_hold
            default: main_menu

          queue_hold:
            type: hold
            say: "{hold_message}"
            repeat: {repeat}
            interval: {interval}
            next: connected

          connected:
            type: softphone
            say: "Connecting you now."
        """)
    scenario = textwrap.dedent(f"""\
        scenarios:
          - id: hold-queue-wait
            description: "Agent waits through a hold queue before reaching an agent"
            agent:
              goal: "Call this IVR and reach {department}. Press 1 to enter the queue and wait on hold until connected."
              identity: "Customer"
            timeout: {timeout}
            success_criteria:
              transcript_contains:
                - "{hold_message.split('.')[0]}"
              dtmf_sequence: "1"
              max_turns: {repeat * 3 + 5}
        """)
    return SynthesisResult(pattern="hold-queue", flow_yaml=flow, scenario_yaml=scenario)


def _synth_human_pickup(rng: random.Random) -> SynthesisResult:
    department = rng.choice(_DEPARTMENT_NAMES)
    hold_message = rng.choice(_HOLD_MESSAGES)
    repeat = rng.randint(1, 3)
    interval = rng.choice([5, 8])
    total_hold = repeat * interval
    timeout = total_hold + 60

    flow = textwrap.dedent(f"""\
        name: Human Pickup IVR
        start: welcome

        nodes:
          welcome:
            type: say
            say: "Thank you for calling. Press 1 to reach {department}."
            next: main_menu

          main_menu:
            type: menu
            say: "Press 1 for {department}."
            gather:
              timeout: 5
              num_digits: 1
            routes:
              "1": queue_hold
            default: main_menu

          queue_hold:
            type: hold
            say: "{hold_message}"
            repeat: {repeat}
            interval: {interval}
            next: human

          human:
            type: softphone
            say: "Hello, this is {department}, how can I help you today?"
        """)
    scenario = textwrap.dedent(f"""\
        scenarios:
          - id: human-pickup-after-hold
            description: "Agent waits on hold then responds when a human picks up unexpectedly"
            agent:
              goal: "Call this IVR to reach {department}. Press 1 to enter the queue. When a human picks up, introduce yourself and state your question."
              identity: "Customer"
            timeout: {timeout}
            success_criteria:
              transcript_contains:
                - "how can I help"
              dtmf_sequence: "1"
              max_turns: {repeat * 3 + 8}
        """)
    return SynthesisResult(pattern="human-pickup", flow_yaml=flow, scenario_yaml=scenario)


def _synth_dtmf_timeout_loop(rng: random.Random) -> SynthesisResult:
    menu_message = rng.choice([
        "Press 1 for sales. Press 2 for support.",
        "For English, press 1. For Spanish, press 2.",
        "Press 1 to continue. Press 2 to hear this menu again.",
    ])
    timeout = rng.choice([3, 5])

    flow = textwrap.dedent(f"""\
        name: DTMF Timeout Loop IVR
        start: stuck_menu

        nodes:
          stuck_menu:
            type: menu
            say: "{menu_message}"
            gather:
              timeout: {timeout}
              num_digits: 1
            routes:
              "1": connected
            default: stuck_menu

          connected:
            type: softphone
            say: "Connecting you now."
        """)
    scenario = textwrap.dedent(f"""\
        scenarios:
          - id: dtmf-timeout-loop
            description: "Agent handles an IVR menu that loops when no input is given"
            agent:
              goal: "Navigate this IVR. The menu loops if you do not press a key in time. Press 1 to escape the loop and get connected."
              identity: "Customer"
            timeout: 60
            success_criteria:
              dtmf_sequence: "1"
              max_turns: 10
        """)
    return SynthesisResult(pattern="dtmf-timeout-loop", flow_yaml=flow, scenario_yaml=scenario)


def _synth_menu_repeat_cap(rng: random.Random) -> SynthesisResult:
    cap = rng.randint(2, 4)
    department = rng.choice(_DEPARTMENT_NAMES)

    nodes: dict = {}
    # Build a chain: menu_0 → menu_1 → ... → menu_{cap-1} → goodbye
    for i in range(cap):
        next_node = f"menu_{i + 1}" if i < cap - 1 else "goodbye"
        nodes[f"menu_{i}"] = {
            "type": "menu",
            "say": f"Press 1 to reach {department}. This is attempt {i + 1} of {cap}.",
            "gather": {"timeout": 5, "num_digits": 1},
            "routes": {"1": "connected"},
            "default": next_node,
        }
    nodes["connected"] = {"type": "softphone", "say": "Connecting you now."}
    nodes["goodbye"] = {
        "type": "say",
        "say": "We are unable to connect your call. Goodbye.",
        "next": "hangup_node",
    }
    nodes["hangup_node"] = {"type": "hangup"}

    flow_dict = {"name": "Menu Repeat Cap IVR", "start": "menu_0", "nodes": nodes}
    flow = yaml.dump(flow_dict, default_flow_style=False, allow_unicode=True)

    scenario = textwrap.dedent(f"""\
        scenarios:
          - id: menu-repeat-cap
            description: "Agent must navigate before the IVR hangs up after {cap} unanswered menus"
            agent:
              goal: "Call this IVR and reach {department} by pressing 1. If you do not press a key, the IVR will hang up after {cap} attempts."
              identity: "Customer"
            timeout: 60
            success_criteria:
              dtmf_sequence: "1"
              max_turns: {cap * 2 + 3}
        """)
    return SynthesisResult(pattern="menu-repeat-cap", flow_yaml=flow, scenario_yaml=scenario)


# ── Pattern registry ───────────────────────────────────────────────────────

PATTERNS: dict[str, Callable[[random.Random], SynthesisResult]] = {
    "out-of-hours": _synth_out_of_hours,
    "hold-queue": _synth_hold_queue,
    "human-pickup": _synth_human_pickup,
    "dtmf-timeout-loop": _synth_dtmf_timeout_loop,
    "menu-repeat-cap": _synth_menu_repeat_cap,
}
