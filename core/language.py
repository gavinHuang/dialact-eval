"""
language.py — Eval-focused LLM service for dialact-eval.

EvalLanguageModel mirrors the prompt logic and pydantic-ai setup from
voice-agent's LanguageModel, but without phone/TTS/telemetry dependencies.

It supports:
  - Multi-turn conversation history (same as production)
  - Token streaming via async callback or async generator
  - Full response accumulation for batch evaluation
  - All system prompts, tool definitions, and token-suppression logic

System prompts, token patterns, and output helpers are imported from
shuo.prompts (voice-agent) — the single source of truth. No duplication.
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, List, AsyncIterator

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, PartDeltaEvent, TextPartDelta
from pydantic_ai.settings import ModelSettings

from shuo.log import ServiceLogger
from shuo.context import CallContext, build_system_prompt
from shuo.translation import extract_speech_text
from shuo.prompts import (
    PROMPT_WITH_TOOLS,
    PROMPT_TEXT_TAGS,
    supports_tools,
    goal_suffix,
    is_suppressed_token,
    is_farewell,
)

log = ServiceLogger("LLM")

_LLM_TIMEOUT     = float(os.getenv("LLM_TIMEOUT",     "30.0"))
_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES",   "1"))


# =============================================================================
# TURN CONTEXT  (mutable per-turn tool side-effects)
# =============================================================================

@dataclass
class _TurnCtx:
    dtmf_queue:     List[str] = field(default_factory=list)
    hold_start:     bool = False
    hold_end:       bool = False
    hold_continue:  bool = False
    hangup_pending: bool = False


# =============================================================================
# TURN RESULT  (returned by EvalLanguageModel.generate())
# =============================================================================

@dataclass
class TurnResult:
    """Outcome of a single LLM turn for evaluation purposes."""
    text: str                   # Full response text (speech only, control tokens stripped)
    raw_text: str               # Raw token stream including control tokens
    dtmf_digits: Optional[str]  # DTMF digits pressed (None if none)
    hangup: bool                # Agent signalled hangup
    hold_continue: bool         # Agent is still on hold
    has_speech: bool            # Response has spoken text


# =============================================================================
# EVAL LANGUAGE MODEL
# =============================================================================

class EvalLanguageModel:
    """
    LLM service for evaluation — same prompt logic as voice-agent's LanguageModel
    but without phone/TTS/telemetry dependencies.

    Usage:
        model = EvalLanguageModel(ctx=CallContext(goal="Book a flight"))
        result = await model.generate("[CALL_STARTED]")
        result = await model.generate("Hi, how can I help?")
        print(model.history)  # full pydantic-ai message history
    """

    def __init__(
        self,
        goal: str = "",
        ctx: Optional[CallContext] = None,
        callee_lang: str = "English",
    ):
        model_name = os.getenv("LLM_MODEL", "groq:llama-3.3-70b-versatile")
        self._tools_enabled = supports_tools(model_name)

        if ctx is not None:
            context_suffix = "\n\n" + build_system_prompt(ctx, tools=self._tools_enabled)
        else:
            context_suffix = goal_suffix(goal, self._tools_enabled)

        lang_suffix = (
            f"\n\nIMPORTANT: Always respond in {callee_lang}, regardless of the language of incoming messages."
            if callee_lang.lower() != "english"
            else ""
        )
        prompt = (
            (PROMPT_WITH_TOOLS if self._tools_enabled else PROMPT_TEXT_TAGS)
            + context_suffix
            + lang_suffix
        )

        settings = ModelSettings(
            max_tokens=  int(os.getenv("LLM_MAX_TOKENS",   "500")),
            temperature= float(os.getenv("LLM_TEMPERATURE", "0.7")),
        )

        self._agent: Agent[_TurnCtx, str] = Agent(
            model=model_name,
            deps_type=_TurnCtx,
            model_settings=settings,
        )

        @self._agent.system_prompt
        def _sys(_ctx: RunContext[_TurnCtx]) -> str:
            return prompt

        if self._tools_enabled:
            @self._agent.tool
            async def press_dtmf(ctx: RunContext[_TurnCtx], digit: str) -> str:
                """Press a DTMF digit on the phone keypad."""
                ctx.deps.dtmf_queue.append(digit)
                return f"Sending digit {digit!r}"

            @self._agent.tool(retries=0)
            async def signal_hold(ctx: RunContext[_TurnCtx]) -> str:
                """Signal that hold music has been detected."""
                ctx.deps.hold_start = True
                return "Hold mode activated"

            @self._agent.tool(retries=0)
            async def signal_hold_continue(ctx: RunContext[_TurnCtx]) -> str:
                """Signal that hold music is still playing."""
                ctx.deps.hold_continue = True
                return "Still on hold"

            @self._agent.tool(retries=0)
            async def signal_hold_end(ctx: RunContext[_TurnCtx]) -> str:
                """Signal that a real person has returned from hold."""
                ctx.deps.hold_end = True
                return "Person returned"

            @self._agent.tool
            async def signal_hangup(ctx: RunContext[_TurnCtx]) -> str:
                """Signal that the call should be hung up after this response."""
                ctx.deps.hangup_pending = True
                return "Will hang up after audio"

        self._history: List[ModelMessage] = []
        self._ctx = ctx
        self._goal = goal

    # ── Public API ──────────────────────────────────────────────────

    @property
    def history(self) -> List[ModelMessage]:
        return self._history.copy()

    def set_history(self, messages: List[ModelMessage]) -> None:
        self._history = list(messages)

    def reset(self) -> None:
        """Clear conversation history (start fresh)."""
        self._history = []

    async def generate(self, message: str) -> TurnResult:
        """
        Generate a response to the given message.

        Maintains conversation history across calls.
        Returns TurnResult with full text, control signals, and metadata.
        """
        turn_ctx = _TurnCtx()
        raw_tokens: List[str] = []

        attempt = 0
        tokens_emitted = False

        while attempt <= _LLM_MAX_RETRIES:
            try:
                async with asyncio.timeout(_LLM_TIMEOUT):
                    async with self._agent.iter(
                        message,
                        deps=turn_ctx,
                        message_history=self._history,
                    ) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        if (
                                            isinstance(event, PartDeltaEvent)
                                            and isinstance(event.delta, TextPartDelta)
                                            and event.delta.content_delta
                                        ):
                                            tokens_emitted = True
                                            raw_tokens.append(event.delta.content_delta)
                            elif Agent.is_call_tools_node(node):
                                async with node.stream(run.ctx) as stream:
                                    async for _ in stream:
                                        pass

                    self._history = list(run.result.all_messages())
                    break

            except asyncio.TimeoutError:
                log.warning(f"LLM timed out after {_LLM_TIMEOUT}s (attempt {attempt + 1})")
                if attempt < _LLM_MAX_RETRIES and not tokens_emitted:
                    attempt += 1
                    raw_tokens.clear()
                    continue
                log.error("LLM timed out — returning empty turn")
                break
            except Exception as e:
                log.error(f"Generation failed (attempt {attempt + 1})", e)
                if attempt < _LLM_MAX_RETRIES and not tokens_emitted:
                    attempt += 1
                    raw_tokens.clear()
                    continue
                break

        raw_text = "".join(raw_tokens)
        speech_text = extract_speech_text(raw_text)

        dtmf_digits = "".join(turn_ctx.dtmf_queue) if turn_ctx.dtmf_queue else None
        hangup = turn_ctx.hangup_pending
        if not hangup and dtmf_digits is None and is_farewell(raw_text):
            hangup = True

        return TurnResult(
            text=speech_text,
            raw_text=raw_text,
            dtmf_digits=dtmf_digits,
            hangup=hangup,
            hold_continue=turn_ctx.hold_continue,
            has_speech=bool(speech_text.strip()) and not turn_ctx.hold_continue and dtmf_digits is None,
        )

    async def stream_generate(
        self,
        message: str,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> TurnResult:
        """
        Stream response tokens, calling on_token for each speech token.

        Control tokens (DTMF, HOLD, etc.) are suppressed from the callback
        but captured in TurnResult.raw_text.
        """
        turn_ctx = _TurnCtx()
        raw_tokens: List[str] = []
        response_started = False

        async with self._agent.iter(
            message,
            deps=turn_ctx,
            message_history=self._history,
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if (
                                isinstance(event, PartDeltaEvent)
                                and isinstance(event.delta, TextPartDelta)
                                and event.delta.content_delta
                            ):
                                token = event.delta.content_delta
                                raw_tokens.append(token)

                                if on_token and not is_suppressed_token(token):
                                    # Strip leading punctuation on first speech token
                                    if not response_started:
                                        token = token.lstrip(", \t\n")
                                        if token:
                                            response_started = True
                                            await on_token(token)
                                    else:
                                        await on_token(token)
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:
                            pass

        self._history = list(run.result.all_messages())

        raw_text = "".join(raw_tokens)
        speech_text = extract_speech_text(raw_text)
        dtmf_digits = "".join(turn_ctx.dtmf_queue) if turn_ctx.dtmf_queue else None
        hangup = turn_ctx.hangup_pending
        if not hangup and dtmf_digits is None and is_farewell(raw_text):
            hangup = True

        return TurnResult(
            text=speech_text,
            raw_text=raw_text,
            dtmf_digits=dtmf_digits,
            hangup=hangup,
            hold_continue=turn_ctx.hold_continue,
            has_speech=bool(speech_text.strip()) and not turn_ctx.hold_continue and dtmf_digits is None,
        )

    async def token_stream(self, message: str) -> AsyncIterator[str]:
        """
        Async generator that yields speech tokens one by one.

        Example:
            async for token in model.token_stream("Hello"):
                print(token, end="", flush=True)
        """
        turn_ctx = _TurnCtx()
        raw_tokens: List[str] = []
        response_started = False

        async with self._agent.iter(
            message,
            deps=turn_ctx,
            message_history=self._history,
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if (
                                isinstance(event, PartDeltaEvent)
                                and isinstance(event.delta, TextPartDelta)
                                and event.delta.content_delta
                            ):
                                token = event.delta.content_delta
                                raw_tokens.append(token)

                                if not is_suppressed_token(token):
                                    if not response_started:
                                        token = token.lstrip(", \t\n")
                                        if token:
                                            response_started = True
                                            yield token
                                    else:
                                        yield token
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:
                            pass

        self._history = list(run.result.all_messages())
