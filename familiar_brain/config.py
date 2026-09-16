from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Server:
    host: str
    port: int


@dataclass(frozen=True)
class Upstream:
    base_url: str
    generate_slot: int
    classify_slot: int
    classify_timeout_s: float


@dataclass(frozen=True)
class Gate:
    follow_up_window_s: float
    # Consecutive follow-up grants allowed before the gate is handed back to the
    # classifier. Defaulted so a brain.toml written before this existed still loads.
    max_follow_ups: int = 3


@dataclass(frozen=True)
class Paths:
    persona: str
    sessions_dir: str
    # Both defaulted so a brain.toml written before they existed still loads, and both
    # tolerate a missing file: an absent lexicon means no corrections, an absent players
    # file means no names to scrub.
    lexicon: str = "config/lexicon.tsv"
    players: str = "config/players.txt"


@dataclass(frozen=True)
class Knowledge:
    db_path: str
    notes_dir: str
    chunks_per_corpus: int
    max_context_chars: int
    retention_days: int


@dataclass(frozen=True)
class Robot:
    # The robot daemon, read-only: the brain GETs /api/state/doa to aim glances and
    # never writes to it (the conversation app owns the head). Empty disables DoA.
    daemon_url: str = ""
    doa_timeout_s: float = 0.15


@dataclass(frozen=True)
class Motion:
    # No two gestures closer than this, whatever their source.
    min_gesture_gap_s: float = 6.0
    # No two reactions to table lines closer than this; also skips the reaction call.
    reaction_gap_s: float = 15.0
    glance_gap_s: float = 12.0
    # How long after the follow-up window closes head tracking stays on.
    tracking_linger_s: float = 4.0
    # Thinking mode ends after this long without an addressed turn.
    mode_timeout_s: float = 300.0


@dataclass(frozen=True)
class Config:
    server: Server
    upstream: Upstream
    gate: Gate
    paths: Paths
    knowledge: Knowledge
    # Both sections are optional so a brain.toml written before they existed still loads.
    robot: Robot = Robot()
    motion: Motion = Motion()


def load_config(path: Path | str) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    return Config(
        server=Server(**raw["server"]),
        upstream=Upstream(**raw["upstream"]),
        gate=Gate(**raw["gate"]),
        paths=Paths(**raw["paths"]),
        knowledge=Knowledge(**raw["knowledge"]),
        robot=Robot(**raw.get("robot", {})),
        motion=Motion(**raw.get("motion", {})),
    )
