"""Re-exports from shuo.log (voice-agent). Source of truth: voice-agent/shuo/log.py."""
from shuo.log import (  # noqa: F401
    C,
    colorize,
    quote,
    ServiceLogger,
    ColorFormatter,
    CorrelatedFileFormatter,
    setup_logging,
    get_logger,
    set_log_call_id,
    clear_log_call_id,
)
