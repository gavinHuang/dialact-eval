"""
context.py — CallContext: typed call parameters and system prompt assembly.

Provides:
  CallContext           — Pydantic model of all agent/call fields
  load_identity_file()  — discover and parse ~/identity.md or <cwd>/identity.md
  build_system_prompt() — assemble the goal/persona portion of the system prompt
  confirm_context()     — interactive pre-call confirmation gate

Shared between voice-agent and dialact-eval.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# CALL CONTEXT
# =============================================================================

class CallContext(BaseModel):
    """
    Typed parameters for a single outbound call.

    Required: goal (must be a non-empty string — raises ValueError if empty/None)
    Optional: all others (agent_name defaults to None — agent will not state a name)
    """
    goal: str
    agent_name: Optional[str] = None
    agent_role: str = "a professional assistant"
    agent_tone: str = "friendly and concise"
    agent_background: Optional[str] = None
    caller_name: Optional[str] = None
    caller_context: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    success_criteria: Optional[str] = None

    @model_validator(mode="after")
    def _validate_goal_required(self) -> "CallContext":
        if not self.goal:
            raise ValueError("CallContext: 'goal' is required and cannot be empty")
        return self

    @classmethod
    def _partial(cls, **kwargs) -> "CallContext":
        """Construct a CallContext bypassing validation (used by CLI before goal is known)."""
        defaults = {
            "goal": "",
            "agent_name": None,
            "agent_role": "a professional assistant",
            "agent_tone": "friendly and concise",
            "agent_background": None,
            "caller_name": None,
            "caller_context": None,
            "constraints": [],
            "success_criteria": None,
        }
        defaults.update(kwargs)
        return cls.model_construct(**defaults)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CallContext":
        """Load a CallContext from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            goal=data.get("goal", ""),
            agent_name=data.get("agent_name") or None,
            agent_role=data.get("agent_role", "a professional assistant"),
            agent_tone=data.get("agent_tone", "friendly and concise"),
            agent_background=data.get("agent_background"),
            caller_name=data.get("caller_name"),
            caller_context=data.get("caller_context"),
            constraints=list(data.get("constraints") or []),
            success_criteria=data.get("success_criteria"),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialize this CallContext to a YAML file."""
        data: dict = {"goal": self.goal}
        if self.agent_name:
            data["agent_name"] = self.agent_name
        if self.agent_role != "a professional assistant":
            data["agent_role"] = self.agent_role
        if self.agent_tone != "friendly and concise":
            data["agent_tone"] = self.agent_tone
        if self.agent_background:
            data["agent_background"] = self.agent_background
        if self.caller_name:
            data["caller_name"] = self.caller_name
        if self.caller_context:
            data["caller_context"] = self.caller_context
        if self.constraints:
            data["constraints"] = list(self.constraints)
        if self.success_criteria:
            data["success_criteria"] = self.success_criteria
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# =============================================================================
# IDENTITY FILE LOADING
# =============================================================================

def load_identity_file(cwd: Path) -> tuple[dict, str]:
    """
    Discover and parse an identity.md file.

    Search order:
      1. <cwd>/identity.md   (project-local persona)
      2. ~/identity.md        (user-global persona)

    Returns (fields_dict, source_label).
    """
    candidates = [
        (cwd / "identity.md",        "identity.md"),
        (Path.home() / "identity.md", "~/identity.md"),
    ]
    for path, label in candidates:
        if path.exists():
            return _parse_identity_file(path), label
    return {}, ""


def _parse_identity_file(path: Path) -> dict:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}

    result: dict = {}
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if fm_match:
        front_matter_text = fm_match.group(1)
        body = fm_match.group(2).strip()
        try:
            fm = yaml.safe_load(front_matter_text) or {}
        except yaml.YAMLError:
            fm = {}
        if fm.get("name"):
            result["agent_name"] = str(fm["name"])
        if fm.get("role"):
            result["agent_role"] = str(fm["role"])
        if fm.get("tone"):
            result["agent_tone"] = str(fm["tone"])
        if body:
            result["agent_background"] = body
    else:
        result["agent_background"] = content

    return result


# =============================================================================
# SYSTEM PROMPT ASSEMBLY
# =============================================================================

def build_system_prompt(ctx: CallContext, tools: bool = True) -> str:
    """
    Assemble the goal/persona/context portion of the system prompt.

    Appended to the base operational prompt in language.py.
    """
    if not ctx.goal:
        return ""

    lines = [
        "Your identity on this call:",
        f"- Name: {ctx.agent_name}" if ctx.agent_name else "- Name: (not provided — do NOT state or invent a name when introducing yourself)",
        f"- Role: {ctx.agent_role}",
        f"- Tone: {ctx.agent_tone}",
    ]

    if ctx.agent_background:
        lines.append(f"\nBackground:\n{ctx.agent_background}")

    lines.append(f"\nYour goal for this call: {ctx.goal}")
    lines.append(
        "Pursue this goal naturally. Do NOT announce your goal — just work towards it. "
        "Once accomplished, confirm details and STOP — wait for their reply. "
        + (
            "Only after they confirm, say goodbye and call signal_hangup() in a separate response."
            if tools else
            "Only after they confirm, say goodbye and emit [HANGUP]."
        )
    )
    lines.append(
        "\nCRITICAL — STRICT SCOPE RULE: Only ask for information that is EXPLICITLY required "
        "to accomplish the stated goal. Do NOT ask for account numbers, IDs, names, verification "
        "details, or any other information unless the goal or context specifically mentions it. "
        "Do NOT assume that verification or identification steps are needed — if they are not "
        "part of the goal, skip them entirely."
    )

    lines.append(
        "\nDo NOT state or announce the name of the person you are calling during the call — "
        "phone calls do not require stating the other party's name. "
        "Never invent, assume, or guess any name that was not explicitly provided to you."
    )
    if ctx.caller_name:
        lines.append(
            f"The person you are calling is named {ctx.caller_name}. "
            "Use this for your own reference only — do not proactively say their name on the call unless they ask."
        )
    if ctx.caller_context:
        lines.append(f"Context about the person you are calling: {ctx.caller_context}")

    if ctx.constraints:
        lines.append("\nInstructions you must follow:")
        for c in ctx.constraints:
            lines.append(f"- {c}")

    if ctx.success_criteria:
        lines.append(f"\nThis call is successful when: {ctx.success_criteria}")

    ivr_rule = (
        "\nIVR NAVIGATION: When the other party is an automated phone system, "
        + (
            "respond with tools only — no speech. "
            "Announcements or partial menus → signal_hold_continue(). "
            "Complete menu option ('press X for Y') → press_dtmf(X). "
            "Auth/input request without the info → press_dtmf('0') for operator."
            if tools else
            "respond with tags only — no speech. "
            "Announcements or partial menus → [HOLD_CONTINUE]. "
            "Complete menu option ('press X for Y') → [DTMF:X]. "
            "Auth/input request without the info → [DTMF:0] for operator."
        )
    )
    lines.append(ivr_rule)

    return "\n".join(lines)


# =============================================================================
# PRE-CALL CONFIRMATION (CLI helper)
# =============================================================================

_EDITABLE_FIELDS = [
    ("Agent name",       "agent_name",       False),
    ("Agent role",       "agent_role",       False),
    ("Agent tone",       "agent_tone",       False),
    ("Goal",             "goal",             False),
    ("Caller name",      "caller_name",      False),
    ("Caller context",   "caller_context",   False),
    ("Constraints",      "constraints",      True),
    ("Success criteria", "success_criteria", False),
]

_ACTION_PROCEED = "Proceed with call"
_ACTION_CANCEL  = "Cancel"


def _render_context(ctx: CallContext, sources: dict) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False,
                  show_edge=False, padding=(0, 1))
    table.add_column("Field", style="dim", min_width=18)
    table.add_column("Value")
    table.add_column("Source", style="dim italic")

    for label, fname, is_list in _EDITABLE_FIELDS:
        value = getattr(ctx, fname)
        src   = sources.get(fname, "")

        if is_list:
            display = ", ".join(value) if value else "[dim](none)[/dim]"
        else:
            display = value if value else "[dim](not set)[/dim]"

        from rich.markup import escape as _escape
        table.add_row(label, display, _escape(f"[{src}]") if src else "")

    if ctx.agent_background:
        preview = ctx.agent_background[:80].replace("\n", " ")
        if len(ctx.agent_background) > 80:
            preview += "…"
        src = sources.get("agent_background", "")
        table.add_row("Agent background", preview, f"[{src}]" if src else "")

    console.print()
    console.rule("[bold]Call Context[/bold]", style="dim")
    console.print(table)
    console.rule(style="dim")
    console.print()


def _build_choices(ctx: CallContext) -> list:
    import questionary
    choices = [
        questionary.Choice(title=_ACTION_PROCEED, value=_ACTION_PROCEED),
        questionary.Choice(title=_ACTION_CANCEL,  value=_ACTION_CANCEL),
        questionary.Separator(),
    ]
    for label, fname, is_list in _EDITABLE_FIELDS:
        value = getattr(ctx, fname)
        if is_list:
            preview = ", ".join(value) if value else "(none)"
        else:
            preview = (value[:50] + "…") if value and len(value) > 50 else (value or "(not set)")
        title = f"Edit  {label:<20}  {preview}"
        choices.append(questionary.Choice(title=title, value=fname))
    return choices


def _edit_field(ctx: CallContext, fname: str, sources: dict) -> CallContext:
    import questionary
    label, _, is_list = next(f for f in _EDITABLE_FIELDS if f[1] == fname)
    current = getattr(ctx, fname)

    if is_list:
        current_str = ", ".join(current) if current else ""
        new_str = questionary.text(
            f"{label} (comma-separated, blank to clear):",
            default=current_str,
        ).ask()
        if new_str is None:
            return ctx
        new_val = [s.strip() for s in new_str.split(",") if s.strip()]
        sources.pop(fname, None)
        return ctx.model_copy(update={fname: new_val})
    else:
        new_val = questionary.text(f"{label}:", default=current or "").ask()
        if new_val is None:
            return ctx
        new_val = new_val.strip()
        if fname == "goal" and not new_val:
            print("  Goal cannot be empty.")
            return ctx
        sources.pop(fname, None)
        result = new_val if new_val else None
        return ctx.model_copy(update={fname: new_val if fname == "goal" else result})


def confirm_context(
    ctx: CallContext,
    yes: bool = False,
    sources: Optional[dict] = None,
) -> CallContext:
    """
    Display the assembled CallContext, prompt for any missing required fields,
    and ask the operator to confirm before dialing.
    """
    import click
    sources = sources or {}

    if not ctx.goal:
        if yes:
            click.echo("Error: 'goal' is required but was not provided.", err=True)
            sys.exit(1)
        goal_input = click.prompt("  Call goal (required)").strip()
        if not goal_input:
            click.echo("Error: goal cannot be empty.", err=True)
            sys.exit(1)
        ctx = ctx.model_copy(update={"goal": goal_input})

    if yes:
        _render_context(ctx, sources)
        return ctx

    import questionary
    while True:
        _render_context(ctx, sources)
        action = questionary.select(
            "What would you like to do?",
            choices=_build_choices(ctx),
            use_shortcuts=False,
        ).ask()

        if action is None or action == _ACTION_CANCEL:
            click.echo("Call cancelled.")
            sys.exit(0)
        elif action == _ACTION_PROCEED:
            return ctx
        else:
            ctx = _edit_field(ctx, action, sources)
