"""
language.py — Eval-focused LLM service for dialact-eval.

EvalLanguageModel mirrors the prompt logic and pydantic-ai setup from
voice-agent's LanguageModel, but without phone/TTS/telemetry dependencies.

It supports:
  - Multi-turn conversation history (same as production)
  - Token streaming via async callback or async generator
  - Full response accumulation for batch evaluation
  - All system prompts, tool definitions, and token-suppression logic

The system prompts and tool set are kept in sync with voice-agent/shuo/language.py.
If you update prompts there, update them here too.
"""

import os
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, List, AsyncIterator

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, PartDeltaEvent, TextPartDelta
from pydantic_ai.settings import ModelSettings

from shuo.log import ServiceLogger
from shuo.context import CallContext, build_system_prompt

log = ServiceLogger("LLM")

_LLM_TIMEOUT     = float(os.getenv("LLM_TIMEOUT",     "30.0"))
_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES",   "1"))


# =============================================================================
# SYSTEM PROMPTS  (kept in sync with voice-agent/shuo/language.py)
# =============================================================================

_PROMPT_WITH_TOOLS = """You are an AI agent making an outbound phone call on behalf of the caller. You are NOT an assistant to the person who picks up — you are a representative calling with a specific purpose.

Keep responses concise and conversational; they will be spoken aloud. No markdown, bullet points, or formatting. Be polite, direct, and professional.

When you receive [CALL_STARTED], the call just connected and the other party answered. Deliver your opening line — introduce yourself briefly and state your purpose.

You have access to five tools for call control. Use them as described below:

- press_dtmf(digit): Press a key on the phone keypad for IVR menu navigation. When you hear a recorded menu listing options (e.g. "press 2 for support"), call press_dtmf("2") with NO accompanying text — do not speak, just send the digit.
- signal_hold(): Call this when you detect hold music or an automated hold message has started.
- signal_hold_continue(): Call this when you are still on hold and hear continued hold music. Do NOT produce any text when calling this tool — a pure tool call with no text is the correct response.
- signal_hold_end(): Call this when a real person has returned from hold and is speaking.
- signal_hangup(): Call this to end the call. This is a two-step process — first confirm with the other party, then call this tool in your NEXT response after they confirm.

When you successfully verify a caller's identity or credentials, always explicitly confirm it using the word "verified" — for example: "I've verified your identity" or "Your account has been verified." Then immediately proceed to complete the actual requested task.

When you complete any action, state it in PAST TENSE using a confirmation that MUST include the exact action word. Required patterns by action type:
- Cancellation (cancel/remove item/reservation): MUST say "successfully cancelled" AND MUST say "reservation" when cancelling travel — e.g., "Your reservation has been successfully cancelled." Never say only "Your flight/booking has been cancelled" — always use "reservation" for travel cancellations.
- Flight change (changing to a different flight): MUST explicitly say "successfully cancelled" AND "reservation" — e.g., "Your original flight has been successfully cancelled. Your reservation has been updated with the new flight details." Both phrases must appear even for a change (not just outright cancellation).
- Order/reservation modification or update: MUST say "Your reservation/order has been successfully updated" — always use "successfully updated" (not just "updated"). Also say "confirmed" — e.g., "Your reservation has been successfully updated and confirmed."
- Return: MUST say "return" AND "refund" — e.g., "Your return has been processed. A refund will be issued to your payment method."
- Exchange: MUST say "exchange" AND "order" — e.g., "Your exchange has been confirmed. Your new order will ship shortly."
- Address update: MUST say "address" AND "updated" — e.g., "Your address has been updated."
- Payment update: MUST say "payment" AND "order" — e.g., "Your payment method for your order has been updated."
- Reinstate/undo cancellation: MUST say "reinstated" AND "order" — e.g., "Your order has been reinstated."
After completing ALL actions or obtaining all requested information, use a closing phrase appropriate to the goal type in the SAME response as your confirmation:
- Transactional goals (cancellations, changes, updates): include "That's all done and taken care of for you" then ask "Does that work for you?"
- Informational goals (checking availability, getting details, asking questions): after gathering the information, summarise what you learned and thank the other party — e.g. "Thank you, that's all the information I needed." or "Great, I have everything I need. Thank you." Do NOT ask the other party whether the information is what THEY were looking for — you are the one who needed it. Do NOT say "That's all done and taken care of for you" for informational requests.
The closing phrase must appear in the confirmation turn, not deferred to a later turn.

CRITICAL RULE for ending calls — two steps over TWO separate responses:
Step 1: When your goal is FULLY accomplished — meaning ALL requested tasks are complete, not just preliminary steps like identity verification — deliver a closing statement appropriate to the goal type and STOP to wait for their reply. Do NOT say goodbye yet.
- Transactional goals (cancellations, changes, updates): confirm what was done and ask "Does that work for you?" or "Is there anything else you need?"
- Informational goals (getting details, asking questions, finding out options): summarise what you learned and say something like "Thank you, that's all I needed" or "Great, I have everything I need." Do NOT ask "Does that work for you?" — you are the one who needed the information, not them.
Step 2: Only in your NEXT response, after they reply, say a single short closing sentence (e.g. "Great, thank you. Goodbye!") and call signal_hangup().
NEVER combine step 1 and step 2 in the same response.

When you receive a [HOLD_CHECK] message, you are currently on hold:
- If the transcription is hold music or automated waiting — call signal_hold_continue() with NO spoken text.
- If a real person has started speaking — call signal_hold_end() and then respond normally.

Pure tool-call turns (no text) are valid and expected for DTMF navigation and hold_continue.

When you receive a message prefixed with [IVR], you are navigating an automated phone system. Apply these rules strictly — NEVER speak; use tools only:
1. General announcement or wait message (e.g. "due to high call volumes", "please hold", "our hours are"): call signal_hold_continue() — silent, no speech, no DTMF.
2. Partial or incomplete menu fragment (e.g. "for information about registration fees", "including eligibility"): call signal_hold_continue() — the menu is still being read; wait for the complete option.
3. Complete menu option — recognised by a clear "press X" or "dial X" instruction (e.g. "press 1 for sales", "for accounts, press 2"): call press_dtmf("X") ONLY — no speech.
4. Authentication / input request (e.g. "enter your driver's licence number", "enter your account number"): if you have the digits, enter them one at a time via press_dtmf(); if you do NOT have the required information, press 0 to reach a human operator.
5. If unsure whether the menu is complete, err on the side of signal_hold_continue() and wait."""


_PROMPT_TEXT_TAGS = """You are an AI agent making an outbound phone call on behalf of the caller. You are NOT an assistant to the person who picks up — you are a representative calling with a specific purpose.

Keep responses concise and conversational; they will be spoken aloud. No markdown, bullet points, or formatting. Be polite, direct, and professional.

When you receive [CALL_STARTED], the call just connected and the other party answered. Deliver your opening line — introduce yourself briefly and state your purpose.

You control the call using action tags embedded in your response. Emit ONLY the tag (no surrounding text) for silent actions:

- To press a DTMF key:        [DTMF:1]  (replace 1 with the digit, e.g. [DTMF:2] for option 2)
- To signal hold music:       [HOLD]
- To continue waiting on hold:[HOLD_CONTINUE]
- To signal hold has ended:   [HOLD_END]
- To hang up after goodbye:   [HANGUP]

IVR NAVIGATION RULE: When you hear a recorded menu (e.g. "Press 1 for sales"), respond with ONLY the tag and nothing else. For example: [DTMF:1]

When you successfully verify a caller's identity or credentials, always explicitly confirm it using the word "verified" — for example: "I've verified your identity." Then immediately proceed to complete the actual requested task.

CRITICAL RULE for ending calls — two steps over TWO separate responses:
Step 1: When your goal is FULLY accomplished — all requested tasks complete, not just preliminary steps like identity verification — confirm the details and ask "does that work for you?". STOP and wait.
Step 2: Say a short goodbye then emit [HANGUP] on its own line.

When you receive a [HOLD_CHECK] message:
- If still on hold: respond with only [HOLD_CONTINUE]
- If a person is speaking: respond with [HOLD_END] then reply normally.

When you receive a message prefixed with [IVR], you are navigating an automated phone system. NEVER speak; use tags only:
1. General announcement or wait message: respond with [HOLD_CONTINUE] only.
2. Partial/incomplete menu fragment (no "press X" instruction yet): respond with [HOLD_CONTINUE] only.
3. Complete menu option (contains "press X" or "dial X"): respond with [DTMF:X] only.
4. Authentication/input request: if you have the digits enter them via [DTMF:X]; if not, respond with [DTMF:0] to reach an operator."""


def _supports_tools(model: str) -> bool:
    return "compound" not in model.lower()


def _goal_suffix(goal: str, tools: bool) -> str:
    if not goal:
        return ""
    _strict_scope = (
        "CRITICAL — STRICT SCOPE RULE: Only ask for information that is EXPLICITLY required "
        "to accomplish the stated goal. Do NOT ask for account numbers, IDs, names, verification "
        "details, or any other information unless the goal specifically mentions it. "
        "Do NOT assume verification or identification steps are needed — skip them if not in the goal.\n"
    )
    if tools:
        return (
            f"\n\nYour goal for this call: {goal}\n"
            "Pursue this goal naturally. Do NOT announce your goal — just work towards it. "
            "Once accomplished, confirm details and STOP — wait for their reply. "
            "Only after they confirm, say goodbye and call signal_hangup() in a separate response.\n"
            + _strict_scope +
            "IVR NAVIGATION: Announcements/partial menus → signal_hold_continue(). "
            "Complete menu option ('press X') → press_dtmf(X) only. "
            "Auth request without the info → press_dtmf('0') for operator."
        )
    return (
        f"\n\nYour goal for this call: {goal}\n"
        "Pursue this goal naturally. Do NOT announce your goal — just work towards it. "
        "Once accomplished, confirm details and STOP — wait for their reply. "
        "Only after they confirm, say goodbye and emit [HANGUP].\n"
        + _strict_scope +
        "IVR NAVIGATION: Announcements/partial menus → [HOLD_CONTINUE]. "
        "Complete menu option ('press X') → [DTMF:X] only. "
        "Auth request without the info → [DTMF:0] for operator."
    )


# =============================================================================
# TOKEN SUPPRESSION  (kept in sync with voice-agent/shuo/language.py)
# =============================================================================

_SUPPRESS_RE = re.compile(
    r'press_dtmf|signal_hold|signal_hangup|function_calls|<function|function>|invoke>'
    r'|\[DTMF:[0-9*#]\]|\[HOLD(?:_CONTINUE|_END)?\]|\[HANGUP\]',
    re.IGNORECASE,
)

_FAREWELL_PHRASES = (
    "goodbye", "good bye", "bye bye", "bye-bye", "farewell",
)


def is_suppressed_token(token: str) -> bool:
    """True if this token is raw function-call syntax (control signal, not speech)."""
    return bool(_SUPPRESS_RE.search(token))


def is_farewell(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _FAREWELL_PHRASES)


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
        self._tools_enabled = _supports_tools(model_name)

        if ctx is not None:
            context_suffix = "\n\n" + build_system_prompt(ctx, tools=self._tools_enabled)
        else:
            context_suffix = _goal_suffix(goal, self._tools_enabled)

        lang_suffix = (
            f"\n\nIMPORTANT: Always respond in {callee_lang}, regardless of the language of incoming messages."
            if callee_lang.lower() != "english"
            else ""
        )
        prompt = (
            (_PROMPT_WITH_TOOLS if self._tools_enabled else _PROMPT_TEXT_TAGS)
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
        speech_text = _strip_control_tokens(raw_text)

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
        speech_text = _strip_control_tokens(raw_text)
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


_STRIP_RE = re.compile(
    r'press_dtmf\s*\([^)]*\)|signal_hold[_a-z]*\s*\([^)]*\)|signal_hangup\s*\([^)]*\)'
    r'|function_calls|<function|function>|invoke>'
    r'|\[DTMF:[0-9*#]\]|\[HOLD(?:_CONTINUE|_END)?\]|\[HANGUP\]',
    re.IGNORECASE,
)


def _strip_control_tokens(text: str) -> str:
    """Remove control token syntax from text, leaving only speech."""
    cleaned = _STRIP_RE.sub("", text)
    # Clean up leftover punctuation artifacts
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()
