"""
pytest fixtures for IVR tests.
"""

import pytest
import httpx
from httpx import AsyncClient, ASGITransport

from ivr.server import app, reload_config


SIMPLE_FLOW = """
name: Simple Test IVR
start: welcome

nodes:
  welcome:
    type: say
    say: "Welcome."
    next: main_menu

  main_menu:
    type: menu
    say: "Press 1 for option one. Press 2 for option two."
    gather:
      timeout: 5
      num_digits: 1
    routes:
      "1": option_one
      "2": option_two
    default: main_menu

  option_one:
    type: say
    say: "You chose option one."
    next: goodbye

  option_two:
    type: say
    say: "You chose option two."
    next: goodbye

  goodbye:
    type: say
    say: "Goodbye."
    next: hangup

  hangup:
    type: hangup
"""

DEEP_FLOW = """
name: Deep Navigation IVR
start: root

nodes:
  root:
    type: menu
    say: "Press 1 for A or 2 for B."
    gather:
      timeout: 3
      num_digits: 1
    routes:
      "1": level_a
      "2": level_b
    default: root

  level_a:
    type: menu
    say: "Level A. Press 1 to continue or star to go back."
    gather:
      timeout: 3
      num_digits: 1
    routes:
      "1": leaf_a1
      "*": root
    default: level_a

  level_b:
    type: say
    say: "You are in level B."
    next: end

  leaf_a1:
    type: say
    say: "Leaf A1."
    next: end

  end:
    type: hangup
"""


@pytest.fixture
def simple_flow() -> str:
    return SIMPLE_FLOW


@pytest.fixture
def deep_flow() -> str:
    return DEEP_FLOW


@pytest.fixture
def client_simple(simple_flow):
    """AsyncClient with SIMPLE_FLOW loaded."""
    reload_config(simple_flow)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def client_deep(deep_flow):
    """AsyncClient with DEEP_FLOW loaded."""
    reload_config(deep_flow)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")
