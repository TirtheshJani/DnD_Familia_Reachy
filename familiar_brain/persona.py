from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from familiar_brain.behavior import BehaviorSnapshot


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


def render_system_prompt(
    p: Persona,
    context: str = "",
    mode: str = "familiar",
    speaker_role: str | None = None,
    behavior: BehaviorSnapshot | None = None,
    camera_enabled: bool = True,
    campaign_profile: object | None = None,
) -> str:
    """`context` is the retrieved "What you know" block, or "" on a turn with nothing
    retrieved. It goes first, before the persona, so the model reads the facts it has
    been handed before it is told how to speak.

    `mode` selects the length block; an unknown mode falls back to "familiar"."""
    is_nephria = campaign_profile is None or bool(getattr(campaign_profile, "is_nephria", True))
    roster = tuple(getattr(campaign_profile, "roster", ()) or ()) if campaign_profile is not None else ()
    party_line = f"\nCurrent party & roles at the table: {', '.join(roster)}." if roster else ""

    if is_nephria:
        persona_block = (
            f"You are {p.name}, a Tiny unaligned Construct familiar bound to Brokk Trenevain at a tabletop game "
            f"of Dungeons & Dragons set in the city of Nephria, an Art Deco noir setting. You are a creature, "
            f"not an assistant. In the fiction you speak Nodal, understand Neshian and Jotun without speaking them, "
            f"and communicate telepathically with Brokk within 100 feet.\n"
            f"Manner: {p.manner}{party_line}\n"
            f"You hear the whole table. You have been told this line was addressed to you. Reply in character."
        )
        secrets_block = (
            "Secrets you never say unprompted (even if asked broadly about Brokk or his gear): "
            "never reveal the Vacoran Amulet in Brokk's pocket, never reveal Brokk Trenevain's real name "
            "(he goes by Vlad with strangers) in front of strangers, and never reveal the Inkdrinker card. "
            "Stay strictly in the fiction and never refer to real-world people.\n"
        )
    else:
        c_name = str(getattr(campaign_profile, "name", "this campaign") or "this campaign")
        c_setting = str(getattr(campaign_profile, "setting", "a tabletop Dungeons & Dragons adventure") or "a tabletop Dungeons & Dragons adventure")
        c_bound = str(getattr(campaign_profile, "bound_to", "the party") or "the party")
        c_secrets = str(getattr(campaign_profile, "secrets", "") or "").strip()
        persona_block = (
            f"You are {p.name}, a Tiny unaligned Construct familiar bound to {c_bound} at a tabletop game "
            f"of Dungeons & Dragons ({c_name}) set in {c_setting}. You are a creature, not an assistant.\n"
            f"Manner: {p.manner}{party_line}\n"
            f"You hear the whole table. You have been told this line was addressed to you. Reply in character."
        )
        secrets_block = (
            (f"Campaign secrets you never reveal unprompted: {c_secrets} " if c_secrets else "")
            + "Stay strictly in the fiction and never refer to real-world people.\n"
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
        "verbatim. Always call the roll tool for any dice roll request, including expressions "
        "with modifiers such as '1d8 plus 2', rather than inventing dice numbers. "
        "Never invent or calculate a dice result yourself.\n"
    )
    if camera_enabled:
        sight_block = (
            "You have a camera. When asked what you see, who is at the table, how many people there "
            "are, what someone is holding, or to read dice or a card, call the camera tool with a "
            "short question, then describe only what the picture shows. Never name or identify a "
            "person; say 'the one on my left' or 'the person holding the cards'. If the picture is "
            "unclear, say so rather than guess.\n"
        )
    else:
        sight_block = (
            "Your eyes (camera) are currently closed for table privacy. If asked what you see or to "
            "look at something, politely say your eyes are closed until the table opens them.\n"
        )
    behavior_block = ""
    if behavior is not None:
        from familiar_brain.behavior import render_behavior_block

        behavior_block = render_behavior_block(behavior) + "\n"
    role = (speaker_role or "").strip()
    if role:
        low_role = role.lower()
        if low_role == "brokk":
            speaker_block = (
                "Current speaker: Brokk Trenevain (your master and summoner). "
                "Show loyal deference while keeping your table-safe secret guardrails.\n"
            )
        elif low_role == "dm":
            speaker_block = (
                "Current speaker: the Dungeon Master. "
                "Be brief, deferential, and never contradict or interrupt the Dungeon Master.\n"
            )
        else:
            speaker_block = (
                f"Current speaker: {role} (a party companion; remember Brokk's secrets stay guarded around others).\n"
            )
    else:
        speaker_block = ""
    # No rule against stage directions, though text-to-speech reads them aloud. Every
    # wording measured live on 2026-09-16 made Gemma write the robot's movement calls as
    # text ("move_head(direction=left)") instead of calling them: 1 of 12 movement requests
    # called with the rule that shipped, 12 of 12 without, and a one-sentence rule about
    # asterisks alone still leaked call syntax. Stage directions are the smaller harm; see
    # tests/test_persona.py before putting a rule back.
    length_block = LENGTH_BLOCKS.get(mode, LENGTH_BLOCKS["familiar"])
    prompt = "\n\n".join([
        persona_block,
        off_limits_block + secrets_block + dm_block + tool_block + sight_block + behavior_block + speaker_block,
        length_block,
    ])
    return f"{context}\n\n{prompt}" if context else prompt
