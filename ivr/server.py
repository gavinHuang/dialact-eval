"""
IVR mock server.

All routes are relative to the mount point so the app can be mounted at any
prefix (e.g. /ivr in dialact-eval, /ivr-mock in voice-agent) without path
duplication.

Endpoints (relative to mount prefix):
    POST /twiml            -- Twilio entry point (start node)
    POST /step?node=ID     -- Render a node
    POST /gather?node=ID   -- Handle DTMF from a menu
    GET  /token            -- Twilio Access Token for browser softphone
    GET  /phone            -- Browser softphone UI
    GET  /health           -- Health check

    -- Visual simulator (no Twilio required) --
    GET  /ui               -- Visual IVR flow tree + agent simulator UI
    GET  /flow             -- IVR config as JSON graph (nodes + edges)
    GET  /simulate?goal=X  -- SSE stream: LLM agent navigates the IVR

Environment variables:
    IVR_CONFIG        Path to YAML flow config (default: flows/example.yaml)
    IVR_BASE_URL      Public base URL including mount prefix
                      (e.g. https://xxxx.ngrok.io/ivr-mock)
    VOICE_AGENT_URL   voice-agent server URL for LLM sessions (default: http://localhost:3040)
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_TWIML_APP_SID   TwiML App SID for browser SDK
    TWILIO_CALLER_ID       Outbound caller ID
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .config import load_config, IVRConfig
from .engine import TwiMLEngine
from .simulate import IVRSimulator, flow_to_graph

app = FastAPI(title="IVR Mock")

_config: Optional[IVRConfig] = None
_engine: Optional[TwiMLEngine] = None


def _get_engine() -> TwiMLEngine:
    global _config, _engine
    if _engine is None:
        config_path = os.getenv("IVR_CONFIG", str(Path(__file__).parent / "flows" / "example.yaml"))
        base_url = os.getenv("IVR_BASE_URL", "")
        _config = load_config(config_path)
        _engine = TwiMLEngine(_config, base_url=base_url)
    return _engine


# ── TwiML Endpoints ────────────────────────────────────────────────────────


@app.post("/twiml")
async def twiml_entry():
    """Entry point — Twilio calls this when a call arrives."""
    engine = _get_engine()
    return PlainTextResponse(engine.render_entry(), media_type="application/xml")


@app.post("/step")
async def ivr_step(node: str = Query(...)):
    """Render a node."""
    engine = _get_engine()
    try:
        xml = engine.render_node(node)
    except KeyError:
        return PlainTextResponse(
            '<?xml version="1.0"?><Response><Say>Configuration error.</Say><Hangup/></Response>',
            media_type="application/xml",
            status_code=200,
        )
    return PlainTextResponse(xml, media_type="application/xml")


@app.post("/gather")
async def ivr_gather(request: Request, node: str = Query(...)):
    """Handle DTMF input."""
    form = await request.form()
    digits = form.get("Digits", "")
    engine = _get_engine()
    try:
        xml = engine.render_gather(node, digits)
    except (KeyError, ValueError):
        xml = '<?xml version="1.0"?><Response><Say>Configuration error.</Say><Hangup/></Response>'
    return PlainTextResponse(xml, media_type="application/xml")


# ── Softphone / Token ──────────────────────────────────────────────────────


@app.get("/token")
async def ivr_token():
    """Generate Twilio Access Token for browser softphone."""
    try:
        from twilio.jwt.access_token import AccessToken
        from twilio.jwt.access_token.grants import VoiceGrant

        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        twiml_app_sid = os.environ["TWILIO_TWIML_APP_SID"]

        token = AccessToken(account_sid, auth_token, identity="browser")
        grant = VoiceGrant(outgoing_application_sid=twiml_app_sid, incoming_allow=True)
        token.add_grant(grant)

        return JSONResponse({"token": token.to_jwt()})
    except KeyError as e:
        return JSONResponse({"error": f"Missing env var: {e}"}, status_code=500)
    except ImportError:
        return JSONResponse({"error": "twilio package not installed"}, status_code=500)


@app.get("/phone", response_class=HTMLResponse)
async def phone_ui():
    """Serve the browser softphone UI."""
    html_path = Path(__file__).parent / "phone.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<html><body><p>phone.html not found</p></body></html>", status_code=404)


# ── Visual simulator (no Twilio) ───────────────────────────────────────────


@app.get("/ui", response_class=HTMLResponse)
async def ivr_ui():
    """Visual IVR flow tree + LLM agent simulator UI."""
    html_path = Path(__file__).parent / "ivr_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<html><body><p>ivr_ui.html not found</p></body></html>", status_code=404)


@app.get("/flow")
async def ivr_flow():
    """Return the IVR config as a JSON graph (nodes + edges) for UI rendering."""
    engine = _get_engine()
    return JSONResponse(flow_to_graph(_config))


@app.get("/simulate")
async def ivr_simulate(goal: str = Query(..., description="Agent goal for the simulation")):
    """
    Run an LLM agent through the IVR flow locally (no Twilio).

    Returns a Server-Sent Events stream with events:
        event: step  — one SimStep per node visited (JSON)
        event: done  — simulation finished
        event: error — error message
    """
    engine = _get_engine()
    voice_agent_url = os.getenv("VOICE_AGENT_URL", "http://localhost:3040")
    log_dir = os.getenv("IVR_SIM_LOG_DIR") or None
    simulator = IVRSimulator(_config, goal=goal, voice_agent_url=voice_agent_url, log_dir=log_dir)

    async def event_stream():
        try:
            async for step in simulator.run():
                data = json.dumps(step.to_dict())
                yield f"event: step\ndata: {data}\n\n"
                await asyncio.sleep(0)  # allow other coroutines to run
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            error = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {error}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Health ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Config reload (test helper) ────────────────────────────────────────────


def reload_config(config_yaml: str) -> None:
    """
    Reload the IVR engine from a YAML string.
    Used in tests to inject an in-memory config.
    """
    import yaml
    from .config import parse_config

    global _config, _engine
    data = yaml.safe_load(config_yaml)
    _config = parse_config(data)
    base_url = os.getenv("IVR_BASE_URL", "")
    _engine = TwiMLEngine(_config, base_url=base_url)
