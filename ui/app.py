"""
app.py — FastAPI backend for the dialact-eval chat UI.

Provides:
  GET  /          → serves the chat UI
  POST /session   → create a new chat session with goal/context
  GET  /session/{id} → get session info and history
  WS   /ws/{id}  → WebSocket for real-time streaming conversation

The UI mimics the voice agent experience in a browser:
  - User types or speaks (via Web Speech API)
  - Agent responds in real-time (token streaming over WebSocket)
  - Full conversation history maintained per session
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.context import CallContext
from core.language import EvalLanguageModel

app = FastAPI(title="dialact-eval chat UI", version="0.1.0")

# Mount static files
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# =============================================================================
# SESSION STORE  (in-memory; replace with Redis for multi-process deployments)
# =============================================================================

class Session:
    def __init__(self, session_id: str, ctx: CallContext):
        self.id = session_id
        self.ctx = ctx
        self.model = EvalLanguageModel(ctx=ctx)
        self.conversation: list[dict] = []  # [{"role": "user"|"agent", "text": str}]
        self.started = False

_sessions: dict[str, Session] = {}


# =============================================================================
# HTTP ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = _STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


class SessionRequest(BaseModel):
    goal: str
    agent_name: Optional[str] = None
    agent_role: str = "a professional assistant"
    agent_tone: str = "friendly and concise"
    agent_background: Optional[str] = None
    caller_name: Optional[str] = None
    caller_context: Optional[str] = None
    constraints: list[str] = []
    success_criteria: Optional[str] = None


@app.post("/session")
async def create_session(req: SessionRequest):
    """Create a new chat session. Returns session_id."""
    ctx = CallContext(
        goal=req.goal,
        agent_name=req.agent_name,
        agent_role=req.agent_role,
        agent_tone=req.agent_tone,
        agent_background=req.agent_background,
        caller_name=req.caller_name,
        caller_context=req.caller_context,
        constraints=req.constraints,
        success_criteria=req.success_criteria,
    )
    session_id = str(uuid.uuid4())
    _sessions[session_id] = Session(session_id, ctx)
    return {"session_id": session_id, "goal": ctx.goal}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session info and conversation history."""
    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return {
        "session_id": session_id,
        "goal": session.ctx.goal,
        "agent_name": session.ctx.agent_name,
        "conversation": session.conversation,
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free its resources."""
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


# =============================================================================
# WEBSOCKET — REAL-TIME STREAMING
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket protocol:
      Client → Server:
        {"type": "message", "text": "Hello"}   — user utterance
        {"type": "start"}                        — trigger [CALL_STARTED] opening
        {"type": "reset"}                        — clear history

      Server → Client:
        {"type": "token", "text": "Hello"}       — streaming token
        {"type": "done", "text": "<full>",        — turn complete
                         "hangup": bool,
                         "dtmf": str|null}
        {"type": "error", "message": "..."}      — error
        {"type": "session_info", ...}             — sent on connect
    """
    await websocket.accept()

    session = _sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    # Send session info on connect
    await websocket.send_json({
        "type": "session_info",
        "session_id": session_id,
        "goal": session.ctx.goal,
        "agent_name": session.ctx.agent_name,
        "conversation": session.conversation,
    })

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "message")

            if msg_type == "reset":
                session.model.reset()
                session.conversation.clear()
                session.started = False
                await websocket.send_json({"type": "reset_ok"})
                continue

            if msg_type == "start":
                user_text = "[CALL_STARTED]"
                session.started = True
            elif msg_type == "message":
                user_text = msg.get("text", "").strip()
                if not user_text:
                    continue
                session.conversation.append({"role": "user", "text": user_text})
            else:
                continue

            # Stream the agent's response
            full_response_tokens: list[str] = []

            async def on_token(token: str) -> None:
                full_response_tokens.append(token)
                await websocket.send_json({"type": "token", "text": token})

            result = await session.model.stream_generate(user_text, on_token=on_token)
            full_text = result.text or "".join(full_response_tokens)

            if full_text.strip():
                session.conversation.append({"role": "agent", "text": full_text})

            await websocket.send_json({
                "type": "done",
                "text": full_text,
                "hangup": result.hangup,
                "dtmf": result.dtmf_digits,
                "hold_continue": result.hold_continue,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# =============================================================================
# ENTRYPOINT
# =============================================================================

def serve(port: int = 8080, reload: bool = False):
    """Start the UI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
