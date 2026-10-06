import json

from familiar_brain.transcript import (
    PIECE_CAP,
    dedupe_revisions,
    is_revision,
    read_rows,
    transcript_pieces,
)


def _r(text, tag=None, ts=0.0):
    row = {"ts": ts, "text": text, "decision": "IGNORE", "spoke": False}
    if tag:
        row["tag"] = tag
    return row


def _texts(rows):
    return [r["text"] for r in rows]


# The pairs below are shaped on data/sessions/20260916-182232-0189cc.jsonl, where
# speech-to-speech re-sent each utterance 2 to 6 times as the player talked on.

def test_a_revision_replaces_the_line_it_revises():
    rows = [_r("Would you like to pick up I can see that?"),
            _r("Would you like to piss up? I can't see it.")]
    assert _texts(dedupe_revisions(rows)) == ["Would you like to piss up? I can't see it."]


def test_a_growing_run_on_keeps_only_its_last_version():
    rows = [_r("Um the general vibe."),
            _r("Um the general vibe uh this in the estates."),
            _r("Um the general vibe uh this in the estates. You're the parrot, right?"),
            _r("Um the general vibe uh this was in the estates. You're the parrot, right? Yes.")]
    assert _texts(dedupe_revisions(rows)) == [rows[-1]["text"]]


def test_a_revision_that_grows_at_the_front_still_replaces():
    rows = [_r("Are y'all selecting all the attainments you got already or the ones you want?"),
            _r("I put out the man. Are y'all selecting all the appealments you got already? "
               "Or the ones you want to do.")]
    assert _texts(dedupe_revisions(rows)) == [rows[1]["text"]]


def test_a_revision_that_lost_the_front_keeps_the_words_it_lost():
    # 2026-09-16, 18:44:50 then 18:44:54: the transcriber's window slid past the start.
    rows = [_r("Look where somebody died. It's not everything. "
               "Um it's a good primary. There's a couple other things that'll be a good one."),
            _r("It's not everything. Um it's it's a good primary. There's a couple other things "
               "that will be. Well the next one is resurrected.")]
    assert _texts(dedupe_revisions(rows)) == [
        "Look where somebody died. It's not everything. "
        "Um it's it's a good primary. There's a couple other things that will be. "
        "Well the next one is resurrected."]


def test_different_lines_are_all_kept():
    rows = [_r("Like you fly."), _r("Great they are."), _r("I do have a dark gift.")]
    assert _texts(dedupe_revisions(rows)) == _texts(rows)


def test_only_neighbours_are_compared():
    # The same words an hour apart are two things said, not a revision.
    rows = [_r("Roll for initiative."), _r("The goblin lunges."), _r("Roll for initiative.")]
    assert _texts(dedupe_revisions(rows)) == _texts(rows)


def test_a_deliberate_note_is_never_folded_and_does_not_break_a_revision_chain():
    rows = [_r("The innkeeper is called Bram."),
            _r("Bram owes us ten gold.", tag="deliberate"),
            _r("The innkeeper is called Bram and he limps."),
            _r("Bram owes us ten gold.", tag="deliberate")]
    out = dedupe_revisions(rows)
    assert _texts(out) == ["The innkeeper is called Bram and he limps.",
                           "Bram owes us ten gold.", "Bram owes us ten gold."]
    assert [r.get("tag") for r in out] == [None, "deliberate", "deliberate"]


def test_rows_without_text_are_dropped():
    assert dedupe_revisions([_r(""), {"ts": 1, "decision": "IGNORE"}, _r("Hello.")]) == [_r("Hello.")]


def test_is_revision_needs_most_of_the_earlier_words():
    assert is_revision("the door creaks open", "the door creaks open slowly")
    assert not is_revision("the door creaks open", "a dragon lands on the roof")


def test_pieces_pack_whole_lines_and_leave_out_deliberate_notes():
    lines = [_r("The innkeeper is Bram."), _r("Note this.", tag="deliberate"), _r("He limps.")]
    assert transcript_pieces(lines) == ["The innkeeper is Bram.\nHe limps."]


def test_pieces_already_made_do_not_change_as_the_log_grows():
    lines = [_r(f"Line number {i} says something about the cistern and the barkeep.") for i in range(80)]
    early, late = transcript_pieces(lines[:40]), transcript_pieces(lines)
    assert len(early) > 1 and late[:len(early) - 1] == early[:-1]
    assert all(len(p) <= PIECE_CAP for p in late)


def test_a_line_longer_than_the_cap_is_split():
    long = " ".join(f"Sentence {i} is here." for i in range(200))
    pieces = transcript_pieces([_r(long)])
    assert len(pieces) > 1 and all(len(p) <= PIECE_CAP + 40 for p in pieces)


def test_read_rows_ignores_a_torn_last_line(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(_r("Whole.")) + "\n" + '{"ts": 2, "text": "torn', encoding="utf-8")
    assert _texts(read_rows(p)) == ["Whole."]


def test_a_short_line_echoed_by_a_later_one_is_kept():
    """A short line folded into any later line that repeated 60 percent of its words, so a
    question about a fact deleted the fact. A short line now folds only into a line that
    begins with it, which is how its re-hearings look on 2026-09-16 ("So excited." became
    "I'm so excited. ..."; "Um the general vibe." grew at the end)."""
    for first, then in [("The innkeeper is called Bram.", "Who is the innkeeper called again?"),
                        ("The password is moonfire.", "Did he say the password is moonfire or sunfire?"),
                        ("Goblins!", "How many goblins are there?"),
                        ("What do you see?", "I see a door. What do you want to do?")]:
        assert _texts(dedupe_revisions([_r(first), _r(then)])) == [first, then]
    for first, then in [("Yeah, just expresses stuff.", "Yeah, it just expresses stuff. Did you build it?"),
                        ("So excited.", "I'm so excited. Literally my favorite spell in the game.")]:
        assert _texts(dedupe_revisions([_r(first), _r(then)])) == [then]
