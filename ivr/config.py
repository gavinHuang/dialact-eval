"""
IVR flow configuration loader.

YAML format:
    name: My IVR
    start: main_menu

    nodes:
      main_menu:
        type: menu
        say: "Press 1 for sales, 2 for support, or 0 to speak to an operator."
        gather:
          timeout: 5
          num_digits: 1
        routes:
          "1": sales
          "2": support
          "0": operator
        default: main_menu

      sales:
        type: say
        text: "Connecting you to sales."
        next: goodbye

      support:
        type: say
        text: "Connecting you to support."
        next: goodbye

      operator:
        type: softphone
        say: "Please hold while we connect you."

      goodbye:
        type: say
        text: "Thank you for calling. Goodbye."
        next: hangup

      hangup:
        type: hangup
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class GatherConfig:
    timeout: int = 5
    num_digits: int = 1


@dataclass
class Node:
    id: str
    type: str  # say | menu | softphone | hangup | pause

    # say / menu
    text: Optional[str] = None
    say: Optional[str] = None

    # say: next node after speaking
    next: Optional[str] = None

    # menu: gather config + routing
    gather: GatherConfig = field(default_factory=GatherConfig)
    routes: Dict[str, str] = field(default_factory=dict)
    default: Optional[str] = None

    # pause
    length: int = 1  # seconds

    # hold
    repeat: int = 1     # number of times to repeat the hold message
    interval: int = 5   # seconds per hold pause

    @property
    def speech(self) -> str:
        """Text to speak (unified: prefer 'say', fall back to 'text')."""
        return self.say or self.text or ""


@dataclass
class IVRConfig:
    name: str
    start: str
    nodes: Dict[str, Node]

    def get(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            raise KeyError(f"IVR node not found: {node_id!r}")
        return self.nodes[node_id]


def load_config(path: str) -> IVRConfig:
    """Load an IVR flow config from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return parse_config(data)


def parse_config(data: Dict[str, Any]) -> IVRConfig:
    """Parse an IVR flow config from a dict (e.g. from YAML)."""
    name = data.get("name", "IVR")
    start = data["start"]
    raw_nodes = data.get("nodes", {})

    nodes: Dict[str, Node] = {}
    for node_id, raw in raw_nodes.items():
        raw_gather = raw.get("gather", {})
        gather = GatherConfig(
            timeout=raw_gather.get("timeout", 5),
            num_digits=raw_gather.get("num_digits", 1),
        )
        nodes[node_id] = Node(
            id=node_id,
            type=raw["type"],
            text=raw.get("text"),
            say=raw.get("say"),
            next=raw.get("next"),
            gather=gather,
            routes={str(k): v for k, v in raw.get("routes", {}).items()},
            default=raw.get("default"),
            length=raw.get("length", 1),
            repeat=raw.get("repeat", 1),
            interval=raw.get("interval", 5),
        )

    _validate(start, nodes)
    return IVRConfig(name=name, start=start, nodes=nodes)


def _validate(start: str, nodes: Dict[str, Node]) -> None:
    if start not in nodes:
        raise ValueError(f"Start node {start!r} not found in nodes")

    for node in nodes.values():
        for dest in list(node.routes.values()) + [node.next, node.default]:
            if dest and dest not in nodes:
                raise ValueError(
                    f"Node {node.id!r} references unknown node {dest!r}"
                )
