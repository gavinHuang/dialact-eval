"""
simulate.py — Local IVR walk engine with LLM-driven navigation.

Runs an IVR flow entirely in-process (no Twilio, no real calls) by:
  1. Walking nodes from the start node.
  2. At `say` / `hold` / `pause` nodes: accumulate speech, follow `next`.
  3. At `menu` nodes: send accumulated speech + menu prompt to the LLM,
     receive a DTMF digit, follow the matching route.
  4. Stop at `hangup` or when max_steps is reached.

LLM calls go through core.language.LLMClient, which delegates to
voice-agent's /llm/sessions API — the same LanguageModel used in production.
If voice-agent is not available the simulator falls back to pressing "1" at
every menu (useful for unit tests / offline use).

Usage::

    from ivr.config import load_config
    from ivr.simulate import IVRSimulator

    config = load_config("flows/example.yaml")
    sim = IVRSimulator(config, goal="Navigate to the billing department")
    async for step in sim.run():
        print(step)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, List, Optional

from .config import IVRConfig, Node


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SimStep:
    """One step in the simulation trace."""
    step: int
    node_id: str
    node_type: str
    speech: str               # Text the IVR spoke
    digit: Optional[str]      # DTMF the agent pressed (None if no input needed)
    next_node: Optional[str]  # Where the flow went next
    llm_raw: str = ""         # Raw LLM response (debug)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "speech": self.speech,
            "digit": self.digit,
            "next_node": self.next_node,
            "llm_raw": self.llm_raw,
        }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class IVRSimulator:
    """
    Walk an IVR flow locally with an LLM agent making navigation decisions.

    Parameters
    ----------
    config : IVRConfig
        Loaded IVR config (from load_config / parse_config).
    goal : str
        The agent's goal for this call (e.g. "Reach the billing department").
    voice_agent_url : str
        Base URL of a running voice-agent server for /llm/sessions.
        Defaults to $VOICE_AGENT_URL or http://localhost:3040.
    max_steps : int
        Safety limit on the number of nodes visited (prevents infinite loops).
    """

    def __init__(
        self,
        config: IVRConfig,
        goal: str,
        voice_agent_url: Optional[str] = None,
        max_steps: int = 50,
        log_dir: Optional[str] = None,
    ):
        self._config = config
        self._goal = goal
        self._base_url = (
            voice_agent_url
            or os.getenv("VOICE_AGENT_URL", "http://localhost:3040")
        )
        self._max_steps = max_steps
        _default_log_dir = Path(__file__).parent.parent / "logs" / "ivr_sim"
        self._log_dir = Path(
            log_dir
            or os.getenv("IVR_SIM_LOG_DIR", str(_default_log_dir))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> AsyncIterator[SimStep]:
        """Async generator yielding one SimStep per IVR node visited."""
        from core.language import LLMClient

        steps: List[SimStep] = []
        async with LLMClient(
            goal=self._goal,
            base_url=self._base_url,
        ) as llm:
            async for step in self._walk(llm):
                steps.append(step)
                yield step

        self._save_log(steps)

    # ------------------------------------------------------------------
    # Internal walk
    # ------------------------------------------------------------------

    async def _walk(self, llm) -> AsyncIterator[SimStep]:
        node_id = self._config.start
        step_num = 0
        pending_speech: list[str] = []

        # Prime the LLM (mirrors shuo/agent.py call start)
        try:
            await llm.generate("[CALL_STARTED]")
        except Exception:
            pass  # non-fatal: LLM may be unavailable

        while step_num < self._max_steps:
            try:
                node = self._config.get(node_id)
            except KeyError:
                break

            step_num += 1

            if node.type == "hangup":
                yield SimStep(
                    step=step_num,
                    node_id=node_id,
                    node_type="hangup",
                    speech="",
                    digit=None,
                    next_node=None,
                )
                break

            elif node.type in ("say", "pause", "out-of-hours"):
                speech = node.speech
                pending_speech.append(speech)
                next_id = node.next
                yield SimStep(
                    step=step_num,
                    node_id=node_id,
                    node_type=node.type,
                    speech=speech,
                    digit=None,
                    next_node=next_id,
                )
                if next_id:
                    node_id = next_id
                else:
                    break

            elif node.type == "hold":
                speech = node.speech or ""
                pending_speech.append(speech)
                next_id = node.next
                yield SimStep(
                    step=step_num,
                    node_id=node_id,
                    node_type="hold",
                    speech=speech,
                    digit=None,
                    next_node=next_id,
                )
                if next_id:
                    node_id = next_id
                else:
                    break

            elif node.type == "softphone":
                speech = node.speech or ""
                yield SimStep(
                    step=step_num,
                    node_id=node_id,
                    node_type="softphone",
                    speech=speech,
                    digit=None,
                    next_node=None,
                )
                break  # call would be connected to a human here

            elif node.type == "menu":
                ivr_prompt = self._build_ivr_prompt(pending_speech, node)
                pending_speech = []

                digit, llm_raw = await self._ask_llm(llm, ivr_prompt)

                next_id = node.routes.get(digit) or node.default or node_id
                yield SimStep(
                    step=step_num,
                    node_id=node_id,
                    node_type="menu",
                    speech=node.speech,
                    digit=digit,
                    next_node=next_id,
                    llm_raw=llm_raw,
                )
                node_id = next_id

            else:
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_log(self, steps: List[SimStep]) -> None:
        """Write simulation trace to a JSON log file in self._log_dir."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            slug = re.sub(r"[^a-z0-9]+", "_", self._goal.lower())[:40].strip("_")
            filename = self._log_dir / f"ivr_sim_{ts}_{slug}.json"

            final_node = steps[-1].node_id if steps else None
            reached_goal = any(s.node_type == "softphone" for s in steps) or (
                steps[-1].node_type == "hangup" if steps else False
            )

            log = {
                "timestamp": datetime.now().isoformat(),
                "flow": self._config.name,
                "goal": self._goal,
                "steps_taken": len(steps),
                "final_node": final_node,
                "path": [s.node_id for s in steps],
                "trace": [s.to_dict() for s in steps],
            }

            with open(filename, "w") as f:
                json.dump(log, f, indent=2)
        except Exception:
            pass  # logging must never break the simulation

    def _build_ivr_prompt(self, preceding: list[str], menu_node: Node) -> str:
        parts: list[str] = []
        if preceding:
            parts.append("The IVR system said: " + " ".join(preceding))
        parts.append("The IVR menu said: " + menu_node.speech)
        parts.append(
            "Based on your goal, reply with ONLY the single digit (or * or #) "
            "you want to press. Do not include any other text."
        )
        return "\n".join(parts)

    async def _ask_llm(self, llm, prompt: str) -> tuple[str, str]:
        """Ask the LLM which digit to press. Returns (digit, raw_response)."""
        try:
            result = await llm.generate(prompt)
            raw = result.text or result.raw_text or ""
            digit = _extract_digit(raw)
            return digit, raw
        except Exception as exc:
            return "1", f"[LLM unavailable: {exc}]"


def _extract_digit(text: str) -> str:
    """Extract the first DTMF digit (* # 0-9) from an LLM response."""
    text = text.strip()
    m = re.search(r'\[DTMF:([0-9*#])\]', text)
    if m:
        return m.group(1)
    m = re.search(r'[0-9*#]', text)
    if m:
        return m.group(0)
    return "1"  # fallback


# ---------------------------------------------------------------------------
# Flow graph extraction (for UI rendering)
# ---------------------------------------------------------------------------

def flow_to_graph(config: IVRConfig) -> dict:
    """
    Convert an IVRConfig to a JSON-serialisable graph suitable for UI rendering.

    Returns::

        {
          "name": "My IVR",
          "start": "main_menu",
          "nodes": [{"id": "main_menu", "type": "menu", "speech": "Press 1 for..."}, ...],
          "edges": [{"from": "main_menu", "to": "sales", "label": "1"}, ...]
        }
    """
    nodes = []
    edges = []

    for nid, node in config.nodes.items():
        nodes.append({
            "id": nid,
            "type": node.type,
            "speech": node.speech,
        })

        for digit, dest in node.routes.items():
            edges.append({"from": nid, "to": dest, "label": digit})

        if node.next and node.next not in node.routes.values():
            edges.append({"from": nid, "to": node.next, "label": ""})

        if node.default and node.default != nid:
            edge_exists = any(
                e["from"] == nid and e["to"] == node.default for e in edges
            )
            if not edge_exists:
                edges.append({"from": nid, "to": node.default, "label": "default"})

    return {
        "name": config.name,
        "start": config.start,
        "nodes": nodes,
        "edges": edges,
    }
