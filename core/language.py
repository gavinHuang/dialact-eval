"""
language.py — HTTP client for voice-agent's LLM session API.

LLMClient delegates all LLM turns to voice-agent's /llm endpoints,
eliminating the need to duplicate pydantic-ai setup in dialact-eval.

Session lifecycle:
  - Session is created lazily on first use (POST /llm/sessions)
  - Session is deleted on aclose() or async context manager exit
  - Use `async with LLMClient(ctx) as model:` for automatic cleanup

Interface matches the old EvalLanguageModel so runner.py and ui/app.py
call sites are unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional, Awaitable

import httpx

from shuo.context import CallContext


# =============================================================================
# TURN RESULT  (mirrors TurnResult in shuo/llm_api.py — built from HTTP JSON)
# =============================================================================

@dataclass
class TurnResult:
    """Outcome of a single LLM turn."""
    text:          str           # Speech text (control tokens stripped)
    raw_text:      str           # Full token stream including control tokens
    dtmf_digits:   Optional[str] # DTMF digits pressed (None if none)
    hangup:        bool          # Agent signalled hangup
    hold_continue: bool          # Agent is still on hold
    has_speech:    bool          # Response contains spoken text


def _parse_turn_result(data: dict) -> TurnResult:
    return TurnResult(
        text=data.get("text", ""),
        raw_text=data.get("raw_text", ""),
        dtmf_digits=data.get("dtmf_digits"),
        hangup=data.get("hangup", False),
        hold_continue=data.get("hold_continue", False),
        has_speech=data.get("has_speech", False),
    )


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """
    Async HTTP client for voice-agent's /llm session API.

    Usage:
        async with LLMClient(ctx=CallContext(goal="Book a flight")) as model:
            result = await model.generate("[CALL_STARTED]")
            result = await model.generate("Hi, how can I help?")

    Or without context manager (manual cleanup):
        model = LLMClient(ctx=ctx)
        result = await model.generate("Hello")
        await model.aclose()
    """

    def __init__(
        self,
        ctx: Optional[CallContext] = None,
        goal: str = "",
        callee_lang: str = "English",
        base_url: Optional[str] = None,
    ):
        self._ctx = ctx
        self._goal = goal
        self._callee_lang = callee_lang
        self._base_url = (base_url or os.getenv("VOICE_AGENT_URL", "http://localhost:3040")).rstrip("/")
        self._session_id: Optional[str] = None
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=60.0)

    # ── Session lifecycle ────────────────────────────────────────────

    async def _ensure_session(self) -> str:
        if self._session_id is not None:
            return self._session_id

        body: dict = {"callee_lang": self._callee_lang}
        if self._ctx is not None:
            body.update(self._ctx.model_dump())
        else:
            body["goal"] = self._goal

        resp = await self._http.post("/llm/sessions", json=body)
        resp.raise_for_status()
        self._session_id = resp.json()["session_id"]
        return self._session_id

    async def aclose(self) -> None:
        if self._session_id is not None:
            try:
                await self._http.delete(f"/llm/sessions/{self._session_id}")
            except Exception:
                pass
            self._session_id = None
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    # ── History helpers (no-ops: history is managed server-side) ────

    def reset(self) -> None:
        """
        Reset conversation history.

        Creates a fresh session on next use; the old session is deleted.
        """
        if self._session_id is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._delete_session(self._session_id))
                else:
                    loop.run_until_complete(self._delete_session(self._session_id))
            except Exception:
                pass
            self._session_id = None

    async def _delete_session(self, session_id: str) -> None:
        try:
            await self._http.delete(f"/llm/sessions/{session_id}")
        except Exception:
            pass

    # ── Public API ──────────────────────────────────────────────────

    async def generate(self, message: str) -> TurnResult:
        """
        Generate a response to the given message (blocking).

        Maintains conversation history across calls (server-side).
        """
        session_id = await self._ensure_session()
        resp = await self._http.post(
            f"/llm/sessions/{session_id}/generate",
            json={"message": message},
        )
        resp.raise_for_status()
        return _parse_turn_result(resp.json())

    async def stream_generate(
        self,
        message: str,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> TurnResult:
        """
        Stream response tokens, calling on_token for each speech token.

        Returns TurnResult after all tokens are received.
        """
        session_id = await self._ensure_session()
        result: Optional[TurnResult] = None

        async with self._http.stream(
            "POST",
            f"/llm/sessions/{session_id}/stream",
            json={"message": message},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "token":
                    if on_token:
                        await on_token(event["text"])
                elif event.get("type") == "done":
                    result = _parse_turn_result(event)

        return result or TurnResult(
            text="", raw_text="", dtmf_digits=None,
            hangup=False, hold_continue=False, has_speech=False,
        )

    async def token_stream(self, message: str) -> AsyncIterator[str]:
        """
        Async generator that yields speech tokens one by one.

        Example:
            async for token in model.token_stream("Hello"):
                print(token, end="", flush=True)
        """
        session_id = await self._ensure_session()

        async with self._http.stream(
            "POST",
            f"/llm/sessions/{session_id}/stream",
            json={"message": message},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "token":
                    yield event["text"]
                elif event.get("type") == "done":
                    break
