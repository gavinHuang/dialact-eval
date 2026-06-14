"""Re-exports from shuo.context (voice-agent). Source of truth: voice-agent/shuo/context.py."""
from shuo.context import (  # noqa: F401
    CallContext,
    load_identity_file,
    build_system_prompt,
    confirm_context,
    ACTION_CANCEL,
    ACTION_PROCEED,
    EDITABLE_FIELDS,
)
