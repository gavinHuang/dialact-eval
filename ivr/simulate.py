"""
simulate.py — Local IVR walk engine with LLM-driven navigation.

Runs an IVR flow entirely in-process (no Twilio, no real calls) by:
  1. Walking nodes from the start node.
  2. At `say` / `hold` / `pause` nodes: accumulate speech, follow `next`.
  3. At `menu` nodes: send accumulated speech + menu prompt to the LLM,
     receive a DTMF digit, follow the matching route.
  4. Stop at `hangup` or when max_steps is reached.

LLM calls delegate to voice-agent's /llm/sessions API (POST /llm/sessions,
POST /llm/sessions/{id}/generate) so the same LanguageModel used in production
drives the simulation.  If voice-agent is not available the simulator falls back
to a simple heuristic (press "1" at every menu — useful for unit tests).

Usage::

    from ivr.config import load_config
    from ivr.simulate import IVRSimulator

    config = load_config("flows/example.yaml")
    sim = IVRSimulator(config, goal="Navigate to the billing department")
    async for step in sim.run():
        print(step)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

import httpx

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
# LLM client (thin wrapper around voice-agent /llm/sessions)
# ---------------------------------------------------------------------------

class _LLMSession:
    """
    Minimal async HTTP client for voice-agent's stateful LLM session API.

    Raises _LLMUnavailableError if the server is unreachable or returns an error.
    """

    def __init__(self, base_url: str, goal: str):
        self._base_url = base_url.rstrip("/")
        self._goal = goal
        self._session_id: Optional[str] = None
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)

    async def _ensure_session(self) -> str:
        if self._session_id:
            return self._session_id
        resp = await self._client.post(
            "/llm/sessions",
            json={
                "goal": self._goal,
                "agent_role": "a caller navigating an IVR phone system",
                "agent_tone": "direct and efficient",
            },
        )
        resp.raise_for_status()
        self._session_id = resp.json()["session_id"]
        return self._session_id

    async def generate(self, message: str) -> dict:
        sid = await self._ensure_session()
        resp = await self._client.post(
            f"/llm/sessions/{sid}/generate",
            json={"message": message},
        )
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        if self._session_id:
            try:
                await self._client.delete(f"/llm/sessions/{self._session_id}")
            except Exception:
                pass
        await self._client.aclose()

    async def __aenter__(self) -> "_LLMSession":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()


class _LLMUnavailableError(Exception):
    pass


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
        Base URL of a running voice-agent server (for /llm/sessions).
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
    ):
        self._config = config
        self._goal = goal
        self._base_url = (
            voice_agent_url
            or os.getenv("VOICE_AGENT_URL", "http://localhost:3040")
        )
        self._max_steps = max_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> AsyncIterator[SimStep]:
        """Async generator yielding one SimStep per IVR node visited."""
        async with _LLMSession(self._base_url, self._goal) as llm:
            async for step in self._walk(llm):
                yield step

    # ------------------------------------------------------------------
    # Internal walk
    # ------------------------------------------------------------------

    async def _walk(self, llm: _LLMSession) -> AsyncIterator[SimStep]:
        node_id = self._config.start
        step_num = 0
        pending_speech: list[str] = []  # accumulated narration before a menu

        # Send a [CALL_STARTED] event to prime the LLM (mirrors shuo/agent.py)
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
                # Build the full IVR prompt for the LLM
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
                # Unknown node type — skip
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    async def _ask_llm(self, llm: _LLMSession, prompt: str) -> tuple[str, str]:
        """Ask the LLM which digit to press. Returns (digit, raw_response)."""
        try:
            result = await llm.generate(prompt)
            raw = result.get("text", "") or result.get("raw_text", "")
            digit = _extract_digit(raw)
            return digit, raw
        except Exception as exc:
            # Fallback: press "1" when LLM is unavailable
            return "1", f"[LLM unavailable: {exc}]"


def _extract_digit(text: str) -> str:
    """Extract the first DTMF digit (* # 0-9) from an LLM response."""
    import re
    text = text.strip()
    # Check for DTMF in TurnResult (voice-agent encodes DTMF as [DTMF:X])
    m = re.search(r'\[DTMF:([0-9*#])\]', text)
    if m:
        return m.group(1)
    # Bare digit response
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
          "nodes": [
            {"id": "main_menu", "type": "menu", "speech": "Press 1 for...", "routes": {"1": "sales"}},
            ...
          ],
          "edges": [
            {"from": "main_menu", "to": "sales", "label": "1"},
            ...
          ]
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

        # Edges from routes
        for digit, dest in node.routes.items():
            edges.append({"from": nid, "to": dest, "label": digit})

        # Edge from next
        if node.next and node.next not in node.routes.values():
            edges.append({"from": nid, "to": node.next, "label": ""})

        # Edge from default (if not already covered)
        if node.default and node.default != nid:
            edge_exists = any(
                e["from"] == nid and e["to"] == node.default
                for e in edges
            )
            if not edge_exists:
                edges.append({"from": nid, "to": node.default, "label": "default"})

    return {
        "name": config.name,
        "start": config.start,
        "nodes": nodes,
        "edges": edges,
    }
