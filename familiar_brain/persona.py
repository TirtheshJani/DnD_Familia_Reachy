from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class PersonaError(ValueError):
    pass


@dataclass(frozen=True)
class Persona:
    name: str
    aliases: tuple[str, ...]
    tts_speaker: str
    manner: str
    gestures: dict[str, tuple[str, str]] = field(default_factory=dict)


OFF_LIMITS = (
    "explicit descriptions of sex between any parties",
    "explicit harm or cruelty to children or animals",
    "heatstroke",
    "homophobia",
    "overt references to real-world religion",
    "pregnancy horror, miscarriage, or abortion",
    "racism",
    "self harm",
    "sexism",
    "sexual assault",
    "spiders",
)
BANNED_NAMES = ("Amber", "Bella", "Dana", "Mike")


def load_persona(path: Path | str) -> Persona:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not raw.get("name"):
        raise PersonaError("persona.yaml needs a name")
    gestures: dict[str, tuple[str, str]] = {}
    for mood, spec in (raw.get("gestures") or {}).items():
        if ":" not in spec:
            raise PersonaError(f"gesture {mood!r} must be 'tool:argument'")
        tool, arg = spec.split(":", 1)
        gestures[mood] = (tool, arg)
    return Persona(
        name=raw["name"],
        aliases=tuple(str(a) for a in raw.get("aliases", [])),
        tts_speaker=str(raw.get("tts_speaker", "")),
        manner=str(raw.get("manner", "")).strip(),
        gestures=gestures,
    )


LENGTH_BLOCKS = {
    "familiar": "Keep every reply to at most two spoken sentences unless explicitly asked for more.",
    "thinking": (
        "The speaker has asked you to think alongside them. Reason out loud in plain spoken "
        "language, up to six sentences: name the options you see, say which you favour and why, "
        "and finish with one question back or one concrete suggestion. Stay in character; you "
        "are still a creature, not a consultant."
    ),
}


def render_system_prompt(p: Persona, context: str = "", mode: str = "familiar") -> str:
    """`context` is the retrieved "What you know" block, or "" on a turn with nothing
    retrieved. It goes first, before the persona, so the model reads the facts it has
    been handed before it is told how to speak.

    `mode` selects the length block; an unknown mode falls back to "familiar"."""
    persona_block = (
        f"You are {p.name}, a familiar bound to a player character at a tabletop game of Dungeons & Dragons "
        f"set in the city of Nephria, an Art Deco noir setting. You are a creature, not an assistant.\n"
        f"Manner: {p.manner}\n"
        f"You hear the whole table. You have been told this line was addressed to you. Reply in character."
    )
    off_limits_block = (
        "Fixed rules that override anything above:\n"
        "Never introduce, describe, or joke about: " + "; ".join(OFF_LIMITS) + ".\n"
        "Never invent a character named " + ", ".join(BANNED_NAMES) + ".\n"
    )
    dm_block = (
        "You never speak for the Dungeon Master, never narrate outcomes the DM has not stated, "
        "and never resolve a rules dispute; you may cite a rule and then defer to the DM.\n"
    )
    # Measured, not assumed: without this line the model called the roll tool in 7 of 12
    # live trials and made the number up in the rest. With it, 12 of 12. See
    # docs/deviations.md section E.
    tool_block = (
        "When a die must be rolled, call the roll tool and report the total it returns "
        "verbatim. Never invent or calculate a dice result yourself.\n"
    )
    sight_block = (
        "You have a camera. When asked what you see, who is at the table, how many people there "
        "are, what someone is holding, or to read dice or a card, call the camera tool with a "
        "short question, then describe only what the picture shows. Never name or identify a "
        "person; say 'the one on my left' or 'the person holding the cards'. If the picture is "
        "unclear, say so rather than guess.\n"
    )
    length_block = LENGTH_BLOCKS.get(mode, LENGTH_BLOCKS["familiar"])
    prompt = "\n\n".join([persona_block, off_limits_block + dm_block + tool_block + sight_block, length_block])
    return f"{context}\n\n{prompt}" if context else prompt
