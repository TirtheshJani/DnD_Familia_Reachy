from __future__ import annotations

from pathlib import Path

import pytest

from familiar_brain.behavior import (
    BehaviorSettings,
    BehaviorSnapshot,
    render_behavior_block,
    scale_reaction_gaps,
    tilt_threat_mood,
)
from familiar_brain.motion import Choreographer, table_mood
from familiar_brain.persona import Persona, render_system_prompt
from familiar_brain.tools import behavior_setting


def _persona() -> Persona:
    return Persona(
        name="Rocky",
        aliases=("rocky",),
        tts_speaker="bm_fable",
        manner="Dry and observant.",
        gestures={
            "attentive": ("play_emotion", "attentive"),
            "amused": ("play_emotion", "happy"),
            "joyful": ("play_emotion", "joyful"),
            "angry": ("play_emotion", "angry"),
            "curious": ("play_emotion", "curious"),
            "alarmed": ("play_emotion", "surprised"),
            "sad": ("play_emotion", "sad"),
            "scared": ("play_emotion", "scared"),
            "cheer": ("play_emotion", "success"),
        },
    )


def test_behavior_settings_persist_clamp_and_presets(tmp_path: Path):
    p = tmp_path / "behavior.json"
    settings = BehaviorSettings(p)
    snap = settings.snapshot()
    assert snap.humor == 75
    assert snap.honesty == 90
    assert snap.camera_enabled is False

    updated = settings.update(humor=150, honesty=-10, bravery=80, camera_enabled=True)
    assert updated.humor == 100
    assert updated.honesty == 0
    assert updated.bravery == 80
    assert updated.camera_enabled is True

    reloaded = BehaviorSettings(p).snapshot()
    assert reloaded == updated

    preset_snap = settings.update(preset="quiet_table")
    assert preset_snap.expressiveness == 25
    assert preset_snap.curiosity == 20

    with pytest.raises(ValueError):
        settings.update(unknown_dial=50)
    with pytest.raises(ValueError):
        settings.update(preset="nonexistent")


def test_render_behavior_block_preserves_dice_and_rules_truthfulness():
    low_honesty = BehaviorSnapshot(humor=95, honesty=10)
    block = render_behavior_block(low_honesty)
    assert "Humor 95%" in block
    assert "Honesty 10%" in block
    assert "never falsify dice roll totals" in block

    prompt = render_system_prompt(_persona(), behavior=low_honesty, camera_enabled=False)
    assert "eyes (camera) are currently closed" in prompt
    assert "never falsify dice roll totals" in prompt


def test_voice_dial_tool_and_parser(tmp_path: Path):
    settings = BehaviorSettings(tmp_path / "behavior.json")
    events: list[dict] = []
    behavior_setting.bind(settings, on_change=events.append)

    assert behavior_setting.parse_voice_setting_command("Rocky, set humor to 60 percent") == ("humor", 60)
    assert behavior_setting.parse_voice_setting_command("Rocky dial your honesty down to 85%") == ("honesty", 85)
    assert behavior_setting.parse_voice_setting_command("Rocky, roll 1d20") is None

    res_set = behavior_setting.run({"action": "set", "setting": "humor", "value": 45})
    assert "45 percent" in res_set
    assert settings.snapshot().humor == 45
    assert len(events) == 1

    res_get = behavior_setting.run({"action": "get", "setting": "humor"})
    assert "45 percent" in res_get

    res_preset = behavior_setting.run({"action": "preset", "preset": "playful_gremlin"})
    assert "playful_gremlin" in res_preset
    assert settings.snapshot().humor == 95


def test_choreographer_respects_expressiveness_bravery_and_new_creature_moods():
    assert scale_reaction_gaps(6.0, 10.0, 0) is None
    assert tilt_threat_mood("scared", 85) == "angry"
    assert tilt_threat_mood("angry", 20) == "scared"
    assert tilt_threat_mood("scared", 55) == "scared"

    assert table_mood("He betrayed us! What a traitor!") == "angry"
    assert table_mood("We found a secret passage behind the portrait!") == "curious"
    assert table_mood("That's amazing, a treasure hoard!") == "joyful"

    now = [100.0]
    snap = [BehaviorSnapshot(expressiveness=0, bravery=55)]
    choreo = Choreographer(
        _persona(),
        min_gap_s=6.0,
        reaction_gap_s=10.0,
        clock=lambda: now[0],
        behavior_provider=lambda: snap[0],
    )
    # Expressiveness 0 pauses all non-verbal table reactions
    assert choreo.for_table("He went down! Get him up!", now[0]) == []

    # High bravery tilts a threat reaction from scared -> angry
    snap[0] = BehaviorSnapshot(expressiveness=80, bravery=85)
    items = choreo.for_table("He went down! Get him up!", now[0])
    assert len(items) == 1
    assert '"angry"' in items[0]["arguments"]
