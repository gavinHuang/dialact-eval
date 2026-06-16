"""
IVR TwiML engine.

Renders IVR nodes as TwiML responses.

URL scheme:
    POST /twiml          -- entry point (renders start node)
    POST /ivr/step?node=ID  -- render a node
    POST /ivr/gather?node=ID -- handle DTMF input for a menu node
"""

from __future__ import annotations

from urllib.parse import urlencode
from .config import IVRConfig, Node


class TwiMLEngine:
    """Generates TwiML strings for IVR nodes."""

    def __init__(self, config: IVRConfig, base_url: str = ""):
        self._config = config
        self._base = base_url.rstrip("/")

    def render_entry(self) -> str:
        """TwiML for the entry point — redirects to start node."""
        return self._redirect(self._config.start)

    def render_node(self, node_id: str) -> str:
        """TwiML for a specific node."""
        node = self._config.get(node_id)

        if node.type == "say":
            return self._render_say(node)
        elif node.type == "menu":
            return self._render_menu(node)
        elif node.type == "softphone":
            return self._render_softphone(node)
        elif node.type == "pause":
            return self._render_pause(node)
        elif node.type == "hangup":
            return self._render_hangup(node)
        elif node.type == "hold":
            return self._render_hold(node)
        elif node.type == "out-of-hours":
            return self._render_out_of_hours(node)
        else:
            raise ValueError(f"Unknown node type: {node.type!r}")

    def render_gather(self, node_id: str, digits: str) -> str:
        """TwiML response after gathering DTMF from a menu node."""
        node = self._config.get(node_id)
        if node.type != "menu":
            raise ValueError(f"Node {node_id!r} is not a menu node")

        dest = node.routes.get(digits) or node.default
        if dest:
            return self._redirect(dest)
        return self._redirect(node_id)  # re-prompt

    # ── Private helpers ────────────────────────────────────────────

    def _render_say(self, node: Node) -> str:
        speech = _esc(node.speech)
        if node.next:
            redirect = f"<Redirect>{self._step_url(node.next)}</Redirect>"
        else:
            redirect = "<Hangup/>"
        return f'<?xml version="1.0"?><Response><Say>{speech}</Say>{redirect}</Response>'

    def _render_menu(self, node: Node) -> str:
        speech = _esc(node.speech)
        timeout = node.gather.timeout
        num_digits = node.gather.num_digits
        gather_url = self._gather_url(node.id)
        return (
            f'<?xml version="1.0"?><Response>'
            f'<Gather action="{gather_url}" method="POST" '
            f'timeout="{timeout}" numDigits="{num_digits}">'
            f"<Say>{speech}</Say>"
            f"</Gather>"
            f"<Redirect>{self._step_url(node.id)}</Redirect>"
            f"</Response>"
        )

    def _render_softphone(self, node: Node) -> str:
        parts = []
        if node.speech:
            parts.append(f"<Say>{_esc(node.speech)}</Say>")
        parts.append("<Dial><Client>browser</Client></Dial>")
        body = "".join(parts)
        return f'<?xml version="1.0"?><Response>{body}</Response>'

    def _render_pause(self, node: Node) -> str:
        length = node.length
        if node.next:
            redirect = f"<Redirect>{self._step_url(node.next)}</Redirect>"
        else:
            redirect = "<Hangup/>"
        return f'<?xml version="1.0"?><Response><Pause length="{length}"/>{redirect}</Response>'

    def _render_hangup(self, _node: Node) -> str:
        return '<?xml version="1.0"?><Response><Hangup/></Response>'

    def _render_hold(self, node: Node) -> str:
        """Unroll repeat×(Pause+Say) then redirect to next node."""
        parts = []
        message = _esc(node.speech) if node.speech else ""
        for _ in range(node.repeat):
            parts.append(f'<Pause length="{node.interval}"/>')
            if message:
                parts.append(f"<Say>{message}</Say>")
        if node.next:
            parts.append(f"<Redirect>{self._step_url(node.next)}</Redirect>")
        else:
            parts.append("<Hangup/>")
        body = "".join(parts)
        return f'<?xml version="1.0"?><Response>{body}</Response>'

    def _render_out_of_hours(self, node: Node) -> str:
        speech = _esc(node.speech)
        return f'<?xml version="1.0"?><Response><Say>{speech}</Say><Hangup/></Response>'

    def _redirect(self, node_id: str) -> str:
        return f'<?xml version="1.0"?><Response><Redirect>{self._step_url(node_id)}</Redirect></Response>'

    def _step_url(self, node_id: str) -> str:
        return f"{self._base}/ivr/step?node={node_id}"

    def _gather_url(self, node_id: str) -> str:
        return f"{self._base}/ivr/gather?node={node_id}"


def _esc(text: str) -> str:
    """Minimal XML escaping for TwiML <Say> text."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
