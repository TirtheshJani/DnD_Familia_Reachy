from pathlib import Path

from familiar_brain.knowledge import build_context
from familiar_brain.retrieval import Chunk, Store
from tests.fakes.embedder import FakeEmbedder


def test_build_context_caps_per_corpus_and_total(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("rules", "r.pdf", f"Rule {i}", "grapple " * 200, {"edition": "2014"}) for i in range(6)])
    s.add_chunks([Chunk("notes", "n.md", "Note", "grapple in the notes", {})])
    ctx = build_context(s, "how does grapple work", chunks_per_corpus=3, max_chars=1500)
    assert ctx.startswith("What you know:") and ctx.count("Rule ") <= 3 and len(ctx) <= 1600
    assert "Note" in ctx or len(ctx) >= 1400  # notes included unless the cap was hit first


def test_build_context_empty_store_returns_empty_string(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert build_context(s, "anything", chunks_per_corpus=3, max_chars=1000) == ""


def test_build_context_includes_homebrew_before_rules(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("rules", "srd.pdf", "Critical Hits", "crit doubles dice", {"edition": "2014"}),
                  Chunk("homebrew", "homebrew/house-rules.md", "Injuries", "crit may become an injury", {})])
    ctx = build_context(s, "what happens on a crit", chunks_per_corpus=3, max_chars=2000)
    assert ctx.index("[homebrew]") < ctx.index("[rules]")


def test_build_context_names_the_edition_of_a_rules_hit(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("rules", "srd52.pdf", "Grappling", "grapple text", {"edition": "2024"})])
    assert "(2024 rules)" in build_context(s, "grapple", chunks_per_corpus=3, max_chars=2000)


DECLARATIONS = [
    ("Sovereignty", "the Realm belongs to the Sovereign and is governed through her"),
    ("Deference", "no public honour may exceed that owed to her"),
    ("Dominion", "the Realm's lands are held in her name by faithful stewards"),
    ("Covenant", "everyone living under her peace owes duty in return"),
    ("Divine Right", "divinity belongs first to her"),
    ("Civilization", "wherever civilisation advances, her will advances with it"),
    ("Beauty", "creation without beauty dishonours the Realm"),
    ("Self", "no one shall be made to live against their nature"),
    ("Revelry", "the fruits of labour are meant to be enjoyed"),
    ("Hospitality", "no guest under her protection shall be denied bread, roof, or judgment"),
    ("Blood", "her blood binds its bearer to the Realm and lends her authority to that bearer"),
    ("Intercession", "no matter in the Realm is beyond her hand if she chooses to extend it"),
]


def test_every_item_of_a_long_canon_list_reaches_the_prompt(tmp_path):
    """build_context cuts each chunk at CHUNK_CHARS, and file 10's Twelve Declarations
    section was one 1067 character chunk, cut mid-word inside declaration 8. A question
    about Hospitality ranked file 10 first and the model saw no Hospitality text."""
    from familiar_brain.chunking import markdown_chunks
    from familiar_brain.knowledge import CHUNK_CHARS
    md = ("### The Twelve Declarations\nThe Sovereign's personal proclamations, spoken at the "
          "Realm's founding and preserved unaltered by the Gold Masks. Every act of government "
          "derives legitimacy from them; nothing lawful may contradict them. In paraphrase:\n\n"
          + "\n".join(f"{i}. {name}: {text}." for i, (name, text) in enumerate(DECLARATIONS, 1)))
    assert len(md) > 1000
    chunks = markdown_chunks(md, source="10-nesh-canon.md", corpus="notes")
    assert all(len(c.text) <= CHUNK_CHARS for c in chunks)
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks(chunks)
    for name, _ in DECLARATIONS:
        ctx = build_context(s, f"What does the {name} declaration say?", chunks_per_corpus=3, max_chars=4800)
        assert f"{name}:" in ctx, name


def test_the_seed_recipe_corrects_eris_and_runs_on_macos_too():
    """File 04 calls Eris a Styxian god and file 10 a Titan of Memory; retrieval cannot
    tell the files apart, so the recipe corrects the copy. sed -i with no suffix fails on
    BSD and macOS sed, which would leave the stale chunk in place silently."""
    recipe = (Path(__file__).parent.parent / "docs" / "notes-policy.md").read_text(encoding="utf-8")
    assert not [line for line in recipe.splitlines() if line.startswith("sed ")]
    assert "Eris is not one of them: she is a Titan of Memory" in recipe


def test_the_context_says_the_notes_outrank_the_memory_of_play(tmp_path):
    """The campaign memory is the familiar's recollection of play, at weight 2.0 in its
    corpus. A misheard name lands there and reaches the prompt beside the DM's canon, so the
    prompt has to say which one wins: the merge prompt saying so could not act on it, since
    the notes are never passed to the merge."""
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("notes", "npcs.md", "Ferryman", "The ferryman is called Odo.", {}),
                  Chunk("sessions", "campaign-memory.md", "Campaign memory: NPCs", "- Otto the ferryman", {}, 2.0)])
    ctx = build_context(s, "who is the ferryman", chunks_per_corpus=3, max_chars=4800)
    assert "[notes]" in ctx and "[sessions]" in ctx
    assert "[notes] are the Dungeon Master's canon" in ctx


def test_long_rules_do_not_squeeze_out_the_sessions_corpus(tmp_path):
    """sessions is read last, and build_context stopped at the first line over budget, so
    once long rules and homebrew chunks were indexed the campaign memory was the first thing
    cut. Each corpus's best hit now goes in before any corpus's second."""
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    for corpus in ("homebrew", "rules"):
        s.add_chunks([Chunk(corpus, f"{corpus}.md", f"{corpus} {i}", "where were we " * 100, {})
                      for i in range(3)])
    s.add_chunks([Chunk("sessions", "campaign-memory.md", f"Campaign memory {i}", "where were we: " + "x " * 180,
                        {}, 2.0) for i in range(3)])
    ctx = build_context(s, "Rocky, where were we?", chunks_per_corpus=3, max_chars=4800)
    assert len(ctx) <= 4800 + 300
    assert ctx.count("[sessions]") >= 2
    # Still read in corpus order.
    assert ctx.index("[homebrew]") < ctx.index("[rules]") < ctx.index("[sessions]")
