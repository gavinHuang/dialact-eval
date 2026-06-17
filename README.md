# dialact-eval

Evaluation and simulation tools for the dialact voice-agent platform.

## Packages

| Package | Description |
|---------|-------------|
| `core/` | Shared LLM session client (`LLMClient`) and call context |
| `ui/` | Chat playground — human-to-agent interactive testing in the browser |
| `ivr/` | IVR flow simulator — config-driven, visual, no Twilio needed |
| `eval/` | Conversation simulation eval — two-agent batch testing with scored metrics |

---

## Chat Playground (Human-to-Agent)

Manually test the voice agent in your browser — you play the caller, the agent responds in real time.

### Start

```bash
dialact-eval ui --port 8081
# Open: http://localhost:8081
```

### User guide

Fill in the setup form:

- **Call Goal** *(required)* — what the agent is trying to accomplish, e.g. *"Book a flight to Sydney for next Friday"*
- **Agent Name / Role** — persona the LLM adopts
- **Caller Context** — account numbers, booking refs, or any background the agent should know
- **Constraints** — one per line, e.g. *"Never reveal account balances"*

Click **Start Conversation**. Type messages as the caller (or hold the mic button to speak via Web Speech API). The agent streams responses token by token. Use the 🔊 button to have responses read aloud.

### Developer guide

**Architecture (`ui/`):**

| File | Role |
|------|------|
| `app.py` | FastAPI server — session store, WebSocket handler, static files |
| `static/index.html` | Single-page UI — setup form + streaming chat panel |

**What happens per turn:**

```
User types → WebSocket /ws/{session_id}
                ↓
            LLMClient.stream_generate(message)
                ↓
            POST /llm/sessions/{id}/stream  (voice-agent)
                ↓  SSE token stream
            LanguageModel (Groq llama-3.3-70b)
                ↓  tokens
            WebSocket → browser (streamed)
```

`LLMClient` in `core/language.py` manages the stateful session against voice-agent's `/llm/sessions` API. Conversation history is kept server-side; the browser just sends the latest message each turn.

---

## Conversation Simulation Eval

Automated batch testing: two LLM agents simulate the two sides of a phone call, with each agent turn scored against configurable metrics. No humans required.

### Start

```bash
dialact-eval eval run path/to/scenarios.yaml
dialact-eval eval run path/to/scenarios.yaml --output-dir eval/reports
dialact-eval eval run path/to/scenarios.yaml --judge   # LLM-as-judge scoring (needs OPENAI_API_KEY)
```

### User guide

**List scenarios in a file:**
```bash
dialact-eval eval list path/to/scenarios.yaml
```

**Run with filtering:**
```bash
dialact-eval eval run scenarios.yaml --filter billing
```

**Output:** a pass/fail table in the terminal, plus JSON and Markdown reports in `--output-dir`.

### Scenario format (YAML)

Two formats are supported:

**Two-agent** — both sides played by LLMs:
```yaml
scenarios:
  - id: cancel-flight
    description: "Agent cancels a flight booking"
    difficulty: medium
    agent:
      goal: "Cancel the caller's flight booking for next Friday"
      identity: "Alex, a travel agent"
    answerer:
      goal: "You want to cancel your Sydney flight. Booking ref XYZABC."
      opening_line: "Hi, I need to cancel a flight."
    required_phrases:
      - "cancelled"
      - "confirmation"
    max_turns: 20
    timeout: 120
```

**Scripted** — agent responds to a fixed sequence of callee lines:
```yaml
scenarios:
  - id: hold-handling
    description: "Agent handles being put on hold"
    agent:
      goal: "Book a table for two at 7pm"
    script:
      - callee_says: "Welcome to La Maison, please hold."
        expected_phrases: ["of course", "sure", "no problem"]
      - callee_says: "Sorry for the wait, how can I help?"
        expected_phrases: ["table", "two", "7"]
```

### Developer guide

**Architecture (`eval/`):**

| File | Role |
|------|------|
| `dataset.py` | Loads YAML scenarios into `EvalScenario` dataclasses |
| `runner.py` | Runs scenarios, produces `ScenarioEvalResult`; writes JSON + Markdown reports |
| `metrics.py` | Custom `deepeval` metrics scored per turn |

**Metrics (all deterministic except Goal Adherence):**

| Metric | What it checks |
|--------|---------------|
| `ConversationalToneMetric` | Response is natural, concise, non-robotic |
| `CallProtocolMetric` | Agent follows call protocol (greeting, confirmation, closing) |
| `SuccessPhrasesMetric` | Response contains required phrases from the scenario |
| `ScopeAdherenceMetric` | Agent doesn't ask for out-of-scope information |
| `GoalAdherenceMetric` | LLM-as-judge: does the agent actively pursue its goal? (`--judge` flag) |

**What happens per scenario:**

```
load_scenario_dataset(path)
    ↓
For each scenario:
  two-agent  →  LLMClient(caller_goal) ↔ LLMClient(answerer_goal)
                alternating .generate() calls until hangup or max_turns
  scripted   →  LLMClient(agent_goal).generate(each callee line)
    ↓
  score each agent turn with metrics
  check required_phrases in full transcript
    ↓
print table  →  write JSON + Markdown to output_dir
```

Both the caller and answerer use `LLMClient` from `core/language.py`, so the same production `LanguageModel` drives both sides of the simulated call.

---

## IVR Visual Simulator

Test IVR menus without a real phone number or Twilio account.

### Start

```bash
dialact-eval ui --port 8081
# Open: http://localhost:8081/ivr/ui
```

### User guide

Open `http://localhost:8081/ivr/ui`. You'll see two panels:

**Left — Flow tree:** The IVR menu structure rendered as a graph. Each box is a node, color-coded by type (blue = menu, green = say, red = hangup, purple = softphone). Arrows show which digit leads where.

**Right — Simulation log:** Empty until you run.

**To run:**
1. Type a goal in the header — e.g. *"Navigate to the billing department"*
2. Click **▶ Run Agent**
3. Watch the log fill in real time: each node visited, what the IVR said, what digit the agent pressed
4. The path through the flow tree lights up green as the agent walks it, with the current node highlighted blue

No phone number, no Twilio account, no cost.

### Developer guide

**Config** (`ivr/flows/example.yaml`): YAML file defines nodes and transitions. Drop in a new YAML file and set `IVR_CONFIG=/path/to/file.yaml` to test any IVR.

**Architecture — three components in `ivr/`:**

| File | Role |
|------|------|
| `config.py` | Parses YAML into `IVRConfig` / `Node` dataclasses |
| `engine.py` | `TwiMLEngine` — renders TwiML XML for real Twilio calls |
| `simulate.py` | `IVRSimulator` — walks the flow locally; `flow_to_graph()` converts config to JSON for the UI |
| `server.py` | FastAPI sub-app, mounted at `/ivr` in `ui/app.py` |
| `ivr_ui.html` | Single-file frontend (SVG graph + SSE log) |

**What happens when you click Run:**

```
Browser → GET /ivr/simulate?goal=...  (SSE stream)
            ↓
        IVRSimulator.run()
            ↓
        For each node:
          say/pause/hold  →  emit SimStep, follow next
          menu            →  build prompt from IVR speech + goal
                          →  LLMClient.generate()  (core.language)
                          →  extract digit from response
                          →  emit SimStep with digit + next_node
          hangup          →  emit SimStep, stop
            ↓
        Browser receives each "step" SSE event
          → appends to log panel
          → highlights node + edge on SVG tree
```

`LLMClient` (from `core.language`) delegates to voice-agent's `/llm/sessions` API — the same `LanguageModel` used in production calls. When voice-agent is not running, the simulator falls back to pressing `"1"` at every menu, which is useful for offline flow validation.

**To test a different IVR flow:**
```bash
IVR_CONFIG=ivr/flows/my_company.yaml dialact-eval ui --port 8081
```

### API endpoints

All routes are relative to the `/ivr` mount prefix:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ivr/ui` | Visual simulator UI |
| `GET` | `/ivr/flow` | IVR config as JSON graph |
| `GET` | `/ivr/simulate?goal=...` | SSE stream: LLM agent navigating the flow |
| `POST` | `/ivr/twiml` | Twilio entry point (real calls) |
| `POST` | `/ivr/step?node=ID` | Render a node as TwiML |
| `POST` | `/ivr/gather?node=ID` | Handle DTMF and route to next node |
| `GET` | `/ivr/phone` | Browser softphone UI |
| `GET` | `/ivr/token` | Twilio Access Token for browser softphone |
