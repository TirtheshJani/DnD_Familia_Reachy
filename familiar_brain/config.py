from __future__ import annotations

import os
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
    classify_cooldown_s: float = 30.0


@dataclass(frozen=True)
class Gate:
    follow_up_window_s: float
    # Consecutive follow-up grants allowed before the gate is handed back to the
    # classifier. Defaulted so a brain.toml written before this existed still loads.
    max_follow_ups: int = 3
    # Slot-1 LLM classifier is opt-in; disabled by default to eliminate ReadTimeout stalls.
    llm_classifier: bool = False


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
    embedder: str = "bge"


@dataclass(frozen=True)
class Motion:
    # No two gestures closer than this, whatever their source.
    min_gesture_gap_s: float = 6.0
    # No two reactions to table lines closer than this; also skips the reaction call.
    # Was 15 while every reaction cost a classify call; most now come from the free
    # keyword rules (motion.TABLE_RULES), so a shorter gap costs the GPU nothing.
    reaction_gap_s: float = 10.0
    # Thinking mode ends after this long without an addressed turn.
    mode_timeout_s: float = 300.0
    # Slot-1 LLM reaction classifier is opt-in; disabled by default to eliminate ReadTimeout stalls.
    llm_reaction: bool = False


@dataclass(frozen=True)
class Speakers:
    enabled: bool = False
    backend: str = "auto"
    store_path: str = "data/speakers.json"
    tap_url: str = "http://127.0.0.1:8767"
    threshold: float = 0.72
    min_margin: float = 0.03
    max_age_s: float = 15.0
    onnx_model_path: str = "data/models/speaker.onnx"


@dataclass(frozen=True)
class Behavior:
    humor: int = 75
    honesty: int = 90
    expressiveness: int = 60
    curiosity: int = 50
    bravery: int = 55
    # Default True for backward compatibility with minimal configs/tests; config/brain.toml
    # sets camera_enabled = false so Thursday's table starts with Eyes Closed until consented.
    camera_enabled: bool = True


@dataclass(frozen=True)
class Config:
    server: Server
    upstream: Upstream
    gate: Gate
    paths: Paths
    knowledge: Knowledge
    # Optional: load_config passes {} for a missing section, so a brain.toml written
    # before it existed still loads.
    motion: Motion = Motion()
    speakers: Speakers = Speakers()
    behavior: Behavior = Behavior()


# Settings for the DoA glances, gone since head tracking became always on (deviation
# P14). A brain.toml written before then still has them, so they are dropped on load
# rather than failing it, and its [robot] section is never read; any other unknown key
# still fails, so a typo is not silent.
RETIRED_MOTION_KEYS = frozenset({"glance_gap_s", "tracking_linger_s"})


def load_config(path: Path | str) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    k_raw = dict(raw["knowledge"])
    env_embedder = os.environ.get("FAMILIAR_EMBEDDER", "").strip().lower()
    if env_embedder:
        k_raw["embedder"] = env_embedder
    elif "embedder" in k_raw:
        k_raw["embedder"] = str(k_raw["embedder"]).strip().lower()
    if k_raw.get("embedder", "bge") not in ("bge", "deterministic"):
        raise ValueError(f"knowledge.embedder must be 'bge' or 'deterministic', got {k_raw.get('embedder')!r}")
    speakers_cfg = Speakers(**{"enabled": True, **raw["speakers"]}) if "speakers" in raw else Speakers()
    behavior_cfg = Behavior(**raw["behavior"]) if "behavior" in raw else Behavior()
    return Config(
        server=Server(**raw["server"]),
        upstream=Upstream(**raw["upstream"]),
        gate=Gate(**raw["gate"]),
        paths=Paths(**raw["paths"]),
        knowledge=Knowledge(**k_raw),
        motion=Motion(**{k: v for k, v in raw.get("motion", {}).items() if k not in RETIRED_MOTION_KEYS}),
        speakers=speakers_cfg,
        behavior=behavior_cfg,
    )
