"""
IVR integration tests.

Tests run against the FastAPI app directly via httpx ASGITransport —
no real Twilio calls needed.

Set IVR_E2E=1 + TWILIO_* env vars to enable real end-to-end call test.
"""

import os
import pytest
from xml.etree import ElementTree as ET

from ivr.config import parse_config, IVRConfig
from ivr.engine import TwiMLEngine
from ivr.server import app, reload_config


# ── Config validation ──────────────────────────────────────────────────────


def test_parse_simple_config(simple_flow):
    config = parse_config(__import__("yaml").safe_load(simple_flow))
    assert config.name == "Simple Test IVR"
    assert config.start == "welcome"
    assert "welcome" in config.nodes
    assert "main_menu" in config.nodes
    assert config.nodes["main_menu"].type == "menu"
    assert config.nodes["main_menu"].routes == {"1": "option_one", "2": "option_two"}


def test_config_rejects_unknown_start():
    import pytest
    with pytest.raises(ValueError, match="Start node"):
        parse_config({
            "name": "Bad",
            "start": "nonexistent",
            "nodes": {"a": {"type": "hangup"}},
        })


def test_config_rejects_unknown_destination():
    with pytest.raises(ValueError, match="unknown node"):
        parse_config({
            "name": "Bad",
            "start": "a",
            "nodes": {
                "a": {"type": "say", "say": "Hi", "next": "ghost"},
            },
        })


def test_config_gather_defaults(simple_flow):
    import yaml
    config = parse_config(yaml.safe_load(simple_flow))
    gather = config.nodes["main_menu"].gather
    assert gather.timeout == 5
    assert gather.num_digits == 1


# ── TwiML engine ───────────────────────────────────────────────────────────


def _engine(flow_yaml: str, base="http://test") -> TwiMLEngine:
    import yaml
    config = parse_config(yaml.safe_load(flow_yaml))
    return TwiMLEngine(config, base_url=base)


def _parse(xml: str) -> ET.Element:
    return ET.fromstring(xml)


def test_entry_redirects_to_start(simple_flow):
    engine = _engine(simple_flow)
    xml = engine.render_entry()
    root = _parse(xml)
    redirect = root.find("Redirect")
    assert redirect is not None
    assert "welcome" in redirect.text


def test_render_say_node(simple_flow):
    engine = _engine(simple_flow)
    xml = engine.render_node("welcome")
    root = _parse(xml)
    say = root.find("Say")
    assert say is not None
    assert "Welcome" in say.text
    redirect = root.find("Redirect")
    assert "main_menu" in redirect.text


def test_render_menu_node(simple_flow):
    engine = _engine(simple_flow)
    xml = engine.render_node("main_menu")
    root = _parse(xml)
    gather = root.find("Gather")
    assert gather is not None
    assert gather.attrib["numDigits"] == "1"
    assert "gather" in gather.attrib["action"]
    assert "main_menu" in gather.attrib["action"]
    say = gather.find("Say")
    assert "Press 1" in say.text


def test_render_hangup_node(simple_flow):
    engine = _engine(simple_flow)
    xml = engine.render_node("hangup")
    root = _parse(xml)
    assert root.find("Hangup") is not None


def test_gather_routes_digit(simple_flow):
    engine = _engine(simple_flow)
    xml = engine.render_gather("main_menu", "1")
    root = _parse(xml)
    redirect = root.find("Redirect")
    assert "option_one" in redirect.text


def test_gather_routes_default_on_unknown_digit(simple_flow):
    engine = _engine(simple_flow)
    xml = engine.render_gather("main_menu", "9")  # not in routes
    root = _parse(xml)
    redirect = root.find("Redirect")
    # default is main_menu
    assert "main_menu" in redirect.text


def test_gather_no_input_reprompts(simple_flow):
    """Empty digits → falls back to default (re-prompt)."""
    engine = _engine(simple_flow)
    xml = engine.render_gather("main_menu", "")
    root = _parse(xml)
    redirect = root.find("Redirect")
    assert "main_menu" in redirect.text


# ── New node types: hold and out-of-hours ─────────────────────────────────

HOLD_FLOW = """
name: Hold Flow
start: welcome

nodes:
  welcome:
    type: say
    say: "Welcome."
    next: wait

  wait:
    type: hold
    say: "Please hold."
    repeat: 3
    interval: 10
    next: done

  done:
    type: hangup
"""

OUT_OF_HOURS_FLOW = """
name: Out Of Hours Flow
start: greeting

nodes:
  greeting:
    type: out-of-hours
    say: "We are currently closed. Please call back during business hours."
"""


def test_parse_hold_node():
    import yaml
    config = parse_config(yaml.safe_load(HOLD_FLOW))
    node = config.nodes["wait"]
    assert node.type == "hold"
    assert node.repeat == 3
    assert node.interval == 10
    assert node.next == "done"
    assert node.speech == "Please hold."


def test_parse_out_of_hours_node():
    import yaml
    config = parse_config(yaml.safe_load(OUT_OF_HOURS_FLOW))
    node = config.nodes["greeting"]
    assert node.type == "out-of-hours"
    assert "closed" in node.speech


def test_render_hold_node():
    engine = _engine(HOLD_FLOW)
    xml = engine.render_node("wait")
    root = _parse(xml)
    pauses = root.findall("Pause")
    assert len(pauses) == 3
    assert pauses[0].attrib["length"] == "10"
    says = root.findall("Say")
    assert len(says) == 3
    redirect = root.find("Redirect")
    assert redirect is not None
    assert "done" in redirect.text


def test_render_hold_node_no_speech():
    import yaml
    flow = """
name: Silent Hold
start: wait
nodes:
  wait:
    type: hold
    repeat: 2
    interval: 5
    next: end
  end:
    type: hangup
"""
    engine = _engine(flow)
    xml = engine.render_node("wait")
    root = _parse(xml)
    pauses = root.findall("Pause")
    assert len(pauses) == 2
    # no Say elements when speech is empty
    assert root.findall("Say") == []


def test_render_out_of_hours_node():
    engine = _engine(OUT_OF_HOURS_FLOW)
    xml = engine.render_node("greeting")
    root = _parse(xml)
    say = root.find("Say")
    assert say is not None
    assert "closed" in say.text
    assert root.find("Hangup") is not None
    assert root.find("Redirect") is None


# ── FastAPI endpoint tests ─────────────────────────────────────────────────


@pytest.mark.anyio
async def test_health(client_simple):
    async with client_simple as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_post_twiml_entry(client_simple):
    async with client_simple as c:
        r = await c.post("/twiml")
    assert r.status_code == 200
    assert "Redirect" in r.text
    assert "welcome" in r.text


@pytest.mark.anyio
async def test_step_say_node(client_simple):
    async with client_simple as c:
        r = await c.post("/ivr/step?node=welcome")
    assert r.status_code == 200
    assert "Welcome" in r.text
    assert "Redirect" in r.text


@pytest.mark.anyio
async def test_step_menu_node(client_simple):
    async with client_simple as c:
        r = await c.post("/ivr/step?node=main_menu")
    assert r.status_code == 200
    assert "Gather" in r.text
    assert "Press 1" in r.text


@pytest.mark.anyio
async def test_gather_valid_digit(client_simple):
    async with client_simple as c:
        r = await c.post("/ivr/gather?node=main_menu", data={"Digits": "1"})
    assert r.status_code == 200
    assert "option_one" in r.text


@pytest.mark.anyio
async def test_gather_invalid_digit_reprompts(client_simple):
    async with client_simple as c:
        r = await c.post("/ivr/gather?node=main_menu", data={"Digits": "9"})
    assert r.status_code == 200
    assert "main_menu" in r.text


@pytest.mark.anyio
async def test_step_unknown_node_returns_200(client_simple):
    """Unknown node should return graceful error TwiML, not 500."""
    async with client_simple as c:
        r = await c.post("/ivr/step?node=does_not_exist")
    assert r.status_code == 200
    assert "Hangup" in r.text


# ── Full call flow simulation ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_full_flow_option_one(client_simple):
    """
    Simulates a complete call:
    /twiml → welcome → main_menu → gather(1) → option_one → goodbye → hangup
    """
    async with client_simple as c:
        # Entry point
        r = await c.post("/twiml")
        assert "welcome" in r.text

        # Welcome node: say + redirect to main_menu
        r = await c.post("/ivr/step?node=welcome")
        assert "Welcome" in r.text
        assert "main_menu" in r.text

        # Main menu: gather
        r = await c.post("/ivr/step?node=main_menu")
        assert "Gather" in r.text

        # Press 1
        r = await c.post("/ivr/gather?node=main_menu", data={"Digits": "1"})
        assert "option_one" in r.text

        # Option one: say + redirect to goodbye
        r = await c.post("/ivr/step?node=option_one")
        assert "option one" in r.text.lower()
        assert "goodbye" in r.text

        # Goodbye: say + redirect to hangup
        r = await c.post("/ivr/step?node=goodbye")
        assert "Goodbye" in r.text
        assert "hangup" in r.text

        # Hangup
        r = await c.post("/ivr/step?node=hangup")
        assert "Hangup" in r.text


@pytest.mark.anyio
async def test_full_flow_option_two(client_simple):
    """Simulates pressing 2 at main menu."""
    async with client_simple as c:
        r = await c.post("/ivr/gather?node=main_menu", data={"Digits": "2"})
        assert "option_two" in r.text

        r = await c.post("/ivr/step?node=option_two")
        assert "option two" in r.text.lower()


@pytest.mark.anyio
async def test_deep_navigation(client_deep):
    """Navigate deep flow: root → level_a → leaf_a1."""
    async with client_deep as c:
        r = await c.post("/ivr/gather?node=root", data={"Digits": "1"})
        assert "level_a" in r.text

        r = await c.post("/ivr/gather?node=level_a", data={"Digits": "1"})
        assert "leaf_a1" in r.text

        r = await c.post("/ivr/step?node=leaf_a1")
        assert "Leaf A1" in r.text


@pytest.mark.anyio
async def test_back_navigation(client_deep):
    """Navigate to level_a then press * to go back to root."""
    async with client_deep as c:
        r = await c.post("/ivr/gather?node=level_a", data={"Digits": "*"})
        assert "root" in r.text


@pytest.mark.anyio
async def test_softphone_node():
    """Softphone node should produce a <Dial><Client> response."""
    import yaml
    from ivr.config import parse_config
    from ivr.engine import TwiMLEngine

    config = parse_config(yaml.safe_load("""
name: Softphone Test
start: op
nodes:
  op:
    type: softphone
    say: "Hold please."
"""))
    engine = TwiMLEngine(config)
    xml = engine.render_node("op")
    root = _parse(xml)
    say = root.find("Say")
    assert say is not None
    assert "Hold" in say.text
    dial = root.find("Dial")
    assert dial is not None
    client = dial.find("Client")
    assert client is not None
    assert client.text == "browser"


@pytest.mark.anyio
async def test_pause_node():
    """Pause node should produce a <Pause> element."""
    import yaml
    from ivr.config import parse_config
    from ivr.engine import TwiMLEngine

    config = parse_config(yaml.safe_load("""
name: Pause Test
start: wait
nodes:
  wait:
    type: pause
    length: 3
    next: end
  end:
    type: hangup
"""))
    engine = TwiMLEngine(config)
    xml = engine.render_node("wait")
    root = _parse(xml)
    pause = root.find("Pause")
    assert pause is not None
    assert pause.attrib["length"] == "3"
    assert "end" in root.find("Redirect").text


# ── Optional real end-to-end test ─────────────────────────────────────────


@pytest.mark.skipif(
    not os.getenv("IVR_E2E"),
    reason="Set IVR_E2E=1 to run real Twilio call test",
)
def test_e2e_real_call():
    """
    Real end-to-end test: make a Twilio call to the IVR server.

    Requires:
        IVR_E2E=1
        TWILIO_ACCOUNT_SID
        TWILIO_AUTH_TOKEN
        TWILIO_PHONE_NUMBER  (your Twilio number to call from)
        TWILIO_CALLER_ID     (number to call)
        IVR_BASE_URL         (public URL, e.g. https://xxxx.ngrok.io)
    """
    from twilio.rest import Client

    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = os.environ["TWILIO_PHONE_NUMBER"]
    to_number = os.environ["TWILIO_CALLER_ID"]
    base_url = os.environ["IVR_BASE_URL"]

    client = Client(account_sid, auth_token)
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=f"{base_url}/twiml",
    )

    print(f"\nCall SID: {call.sid}")
    print("Call initiated. Check your phone.")
    assert call.sid.startswith("CA")
