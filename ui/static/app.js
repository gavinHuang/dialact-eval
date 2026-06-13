/**
 * dialact-eval chat UI
 *
 * Provides:
 * - Session creation and WebSocket management
 * - Real-time token streaming display
 * - Browser Web Speech API for mic input (STT)
 * - Optional browser TTS read-aloud for agent responses
 */

// ── State ─────────────────────────────────────────────────────────────────

let sessionId = null;
let ws = null;
let streamingBubble = null;
let streamingTokens = [];
let ttsEnabled = false;
let micActive = false;
let recognition = null;

// ── DOM ───────────────────────────────────────────────────────────────────

const setupPanel    = document.getElementById('setup-panel');
const chatPanel     = document.getElementById('chat-panel');
const goalInput     = document.getElementById('goal-input');
const agentNameInput= document.getElementById('agent-name');
const agentRoleInput= document.getElementById('agent-role');
const callerCtxInput= document.getElementById('caller-context');
const constraintsInput = document.getElementById('constraints');
const startBtn      = document.getElementById('start-btn');
const backBtn       = document.getElementById('back-btn');
const messagesEl    = document.getElementById('messages');
const userInput     = document.getElementById('user-input');
const sendBtn       = document.getElementById('send-btn');
const statusText    = document.getElementById('status-text');
const sessionIdLabel= document.getElementById('session-id-label');
const headerAgentName = document.getElementById('header-agent-name');
const headerGoal    = document.getElementById('header-goal');
const micBtn        = document.getElementById('mic-btn');
const ttsBtn        = document.getElementById('tts-btn');
const resetBtn      = document.getElementById('reset-btn');

// ── Setup ─────────────────────────────────────────────────────────────────

startBtn.addEventListener('click', async () => {
  const goal = goalInput.value.trim();
  if (!goal) {
    goalInput.focus();
    goalInput.style.borderColor = 'var(--red)';
    return;
  }
  goalInput.style.borderColor = '';

  const constraints = constraintsInput.value
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean);

  startBtn.disabled = true;
  startBtn.textContent = 'Starting…';

  try {
    const resp = await fetch('/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        goal,
        agent_name: agentNameInput.value.trim() || null,
        agent_role: agentRoleInput.value.trim() || 'a professional assistant',
        caller_context: callerCtxInput.value.trim() || null,
        constraints,
      }),
    });
    const data = await resp.json();
    sessionId = data.session_id;

    headerAgentName.textContent = agentNameInput.value.trim() || 'Agent';
    headerGoal.textContent = goal;
    sessionIdLabel.textContent = sessionId.slice(0, 8) + '…';

    setupPanel.classList.add('hidden');
    chatPanel.classList.remove('hidden');

    connectWebSocket();
  } catch (err) {
    setStatus('Error: ' + err.message, true);
  } finally {
    startBtn.disabled = false;
    startBtn.textContent = 'Start Conversation';
  }
});

backBtn.addEventListener('click', () => {
  if (ws) { ws.close(); ws = null; }
  chatPanel.classList.add('hidden');
  setupPanel.classList.remove('hidden');
  messagesEl.innerHTML = '';
  sessionId = null;
});

// ── WebSocket ──────────────────────────────────────────────────────────────

function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);
  setStatus('Connecting…');

  ws.onopen = () => {
    setStatus('Connected');
    // Trigger opening line
    ws.send(JSON.stringify({ type: 'start' }));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleServerMessage(msg);
  };

  ws.onerror = () => setStatus('WebSocket error', true);

  ws.onclose = () => {
    setStatus('Disconnected');
    ws = null;
  };
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case 'session_info':
      // Restore prior conversation if reconnecting
      if (msg.conversation && msg.conversation.length > 0) {
        msg.conversation.forEach(turn => {
          appendMessage(turn.role === 'user' ? 'user' : 'agent', turn.text);
        });
      }
      break;

    case 'token':
      if (!streamingBubble) {
        streamingBubble = createStreamingBubble();
      }
      streamingTokens.push(msg.text);
      updateStreamingBubble(streamingTokens.join(''));
      break;

    case 'done':
      finalizeStreamingBubble(msg);
      setStatus('Ready');
      sendBtn.disabled = false;
      if (ttsEnabled && msg.text && msg.text.trim()) {
        speak(msg.text);
      }
      if (msg.hangup) {
        appendSystemMessage('📞 Call ended by agent');
        setStatus('Call ended');
      }
      break;

    case 'reset_ok':
      messagesEl.innerHTML = '';
      appendSystemMessage('Conversation reset');
      break;

    case 'error':
      setStatus('Error: ' + msg.message, true);
      sendBtn.disabled = false;
      break;
  }
}

// ── Send messages ──────────────────────────────────────────────────────────

function sendMessage(text) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!text.trim()) return;

  appendMessage('user', text);
  userInput.value = '';
  autoResizeInput();
  sendBtn.disabled = true;
  setStatus('Thinking…');

  ws.send(JSON.stringify({ type: 'message', text }));
}

sendBtn.addEventListener('click', () => sendMessage(userInput.value));

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(userInput.value);
  }
});

userInput.addEventListener('input', autoResizeInput);

function autoResizeInput() {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
}

// ── Message DOM helpers ────────────────────────────────────────────────────

function appendMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const roleLabel = document.createElement('div');
  roleLabel.className = 'message-role';
  roleLabel.textContent = role === 'user' ? 'You' : role === 'agent' ? (headerAgentName.textContent || 'Agent') : 'System';
  wrap.appendChild(roleLabel);

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);

  messagesEl.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function appendSystemMessage(text) {
  const wrap = document.createElement('div');
  wrap.className = 'message system';
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
}

function createStreamingBubble() {
  streamingTokens = [];
  const wrap = document.createElement('div');
  wrap.className = 'message agent';
  wrap.id = 'streaming-wrap';

  const roleLabel = document.createElement('div');
  roleLabel.className = 'message-role';
  roleLabel.textContent = headerAgentName.textContent || 'Agent';
  wrap.appendChild(roleLabel);

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble streaming';
  bubble.id = 'streaming-bubble';
  wrap.appendChild(bubble);

  messagesEl.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function updateStreamingBubble(text) {
  if (streamingBubble) {
    streamingBubble.textContent = text;
    scrollToBottom();
  }
}

function finalizeStreamingBubble(msg) {
  if (streamingBubble) {
    streamingBubble.classList.remove('streaming');
    streamingBubble.textContent = msg.text || streamingTokens.join('');

    // Add DTMF badge if needed
    if (msg.dtmf) {
      const badge = document.createElement('div');
      badge.className = 'dtmf-badge';
      badge.textContent = `📟 DTMF: ${msg.dtmf}`;
      streamingBubble.parentElement.appendChild(badge);
    }
    if (msg.hangup) {
      const badge = document.createElement('div');
      badge.className = 'hangup-badge';
      badge.textContent = '📵 Hangup signalled';
      streamingBubble.parentElement.appendChild(badge);
    }
  }
  streamingBubble = null;
  streamingTokens = [];
  scrollToBottom();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Mic (Web Speech API STT) ───────────────────────────────────────────────

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onresult = (event) => {
    let interim = '';
    let final = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) final += t;
      else interim += t;
    }
    userInput.value = final || interim;
    autoResizeInput();
  };

  recognition.onend = () => {
    micActive = false;
    micBtn.classList.remove('recording');
    micBtn.title = 'Hold to speak';
    // Auto-send if we have text
    const text = userInput.value.trim();
    if (text) sendMessage(text);
  };

  recognition.onerror = (e) => {
    micActive = false;
    micBtn.classList.remove('recording');
    if (e.error !== 'aborted') setStatus('Mic error: ' + e.error, true);
  };

  micBtn.addEventListener('click', () => {
    if (!micActive) {
      micActive = true;
      micBtn.classList.add('recording');
      micBtn.title = 'Listening… click to stop';
      setStatus('Listening…');
      recognition.start();
    } else {
      recognition.stop();
    }
  });
} else {
  micBtn.title = 'Speech recognition not supported in this browser';
  micBtn.style.opacity = '0.4';
  micBtn.style.cursor = 'not-allowed';
}

// ── TTS (browser read-aloud) ───────────────────────────────────────────────

function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 1.05;
  utt.pitch = 1.0;
  window.speechSynthesis.speak(utt);
}

ttsBtn.addEventListener('click', () => {
  ttsEnabled = !ttsEnabled;
  ttsBtn.classList.toggle('active', ttsEnabled);
  ttsBtn.title = ttsEnabled ? 'Read-aloud ON — click to disable' : 'Enable read-aloud';
  if (!ttsEnabled && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
});

// ── Reset ─────────────────────────────────────────────────────────────────

resetBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
});

// ── Status ─────────────────────────────────────────────────────────────────

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.style.color = isError ? 'var(--red)' : 'var(--text-muted)';
}
