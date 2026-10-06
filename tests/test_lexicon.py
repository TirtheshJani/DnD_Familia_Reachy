from pathlib import Path

from familiar_brain.lexicon import Lexicon

TSV = ("Nephria\tNefrea|Nephria-Nephria\nStyx\tsticks\n"
       "Graywater Crossing\tGreywater Crossing|the Crossing\n"
       "Thraka\tThraca|Brocka|Taraka\nAlar\tMalar|Molar|Pilar\n")


def make(tmp: Path) -> Lexicon:
    p = tmp / "lexicon.tsv"; p.write_text(TSV, encoding="utf-8"); return Lexicon.load(p)


def test_exact_variant_replaced_case_insensitively(tmp_path):
    lx = make(tmp_path)
    assert lx.correct("we crossed the river sticks near nefrea") == "we crossed the river Styx near Nephria"


def test_multiword_variant_and_no_partial_word_hits(tmp_path):
    lx = make(tmp_path)
    assert lx.correct("meet at the Greywater Crossing") == "meet at the Graywater Crossing"
    assert lx.correct("chopsticks are useful") == "chopsticks are useful"


def test_fuzzy_close_match_for_long_tokens_only(tmp_path):
    lx = make(tmp_path)
    assert lx.correct("Thrakka swung first") == "Thraka swung first"   # one letter off, fuzzy
    assert lx.correct("Alan swung first") == "Alan swung first"        # short common word, left alone


def test_empty_lexicon_is_identity(tmp_path):
    p = tmp_path / "e.tsv"; p.write_text("", encoding="utf-8")
    assert Lexicon.load(p).correct("anything at all") == "anything at all"


def test_missing_file_is_an_empty_lexicon(tmp_path):
    """config/lexicon.tsv is gitignored-adjacent and may simply not be written yet.
    A brain that will not start because of that is worse than one with no corrections."""
    assert Lexicon.load(tmp_path / "nope.tsv").correct("anything at all") == "anything at all"


def test_canonical_spelling_is_left_alone(tmp_path):
    lx = make(tmp_path)
    assert lx.correct("Nephria and Thraka and Styx") == "Nephria and Thraka and Styx"


def test_shipped_lexicon_corrects_heard_forms_of_rocky():
    lex = Lexicon.load(Path(__file__).parent.parent / "config" / "lexicon.tsv")
    assert lex.correct("Rockie, roll a d20") == "Rocky, roll a d20"
    assert lex.correct("hey Rockey") == "hey Rocky"


SHIPPED = Path(__file__).parent.parent / "config" / "lexicon.tsv"


def test_shipped_lexicon_corrects_heard_forms_of_the_nesh_canon():
    """plans/10-nesh-canon.md renamed the gate settlement and the preserving Titan, and
    files 01 to 09 still carry the old forms, so the old forms are the likeliest to be
    said or heard. The compounds are the ones speech recognition splits into two words."""
    lex = Lexicon.load(SHIPPED)
    assert lex.correct("we waited at North Reach Plaza") == "we waited at Northreach Crossing"
    assert lex.correct("back to north reach crossing") == "back to Northreach Crossing"
    assert lex.correct("Thanatos, the Memory Tide") == "Thatos, the Memory Tide"
    assert lex.correct("the gate warden called my name") == "the Gatewarden called my name"
    assert lex.correct("the Iron Bank of Mali burn") == "the Iron Bank of Maliburn"
    assert lex.correct("bound after a sen night") == "bound after a sennight"
    assert lex.correct("the Mailburn treasury") == "the Maliburn treasury"   # one swap, fuzzy
    assert lex.correct("a Neeshian farewell") == "a Neshian farewell"         # one letter, fuzzy


def test_shipped_lexicon_leaves_ordinary_words_near_the_nesh_canon_alone():
    """Eris and Nesh are under the six-letter fuzzy floor, and their nearest heard forms
    (iris, error, mesh, fresh) are everyday words, so they get no variants at all. The
    rest are checked against the words a fuzzy match would most plausibly steal."""
    lex = Lexicon.load(SHIPPED)
    for s in ("that's a good point", "her iris was gold", "a rules error", "chain mesh",
              "fresh bread", "a trip to Malibu", "flights to Melbourne", "two crowns each",
              "they wore gold masks", "tonight at midnight", "a fortnight away",
              "the garden gateway", "Northreach Crossing at dusk", "Thatos and Eris"):
        assert lex.correct(s) == s, s


def test_shipped_lexicon_keeps_the_plurals_of_the_nesh_canon():
    """A fuzzy hit replaces the whole token with the canonical spelling, so without their
    own rows "Neshians" (0.93 to Neshian) and "sennights" (0.94 to sennight) would lose
    the plural. A canonical row is never corrected, which is what keeps them."""
    lex = Lexicon.load(SHIPPED)
    assert lex.correct("the Neshians waited two sennights") == "the Neshians waited two sennights"


def test_shipped_lexicon_keeps_possessives_and_plurals_of_the_canon():
    """The fuzzy pass reads an apostrophe or hyphen as part of the word, so a correctly
    heard "Gatewarden's" scored 0.909 against "Gatewarden" and lost its possessive, and
    "Neshian's" became the plural "Neshians". The corrected text is what is gated, logged
    and sent to retrieval and the model."""
    lex = Lexicon.load(SHIPPED)
    assert lex.correct("Rocky, the Gatewarden's papers please") == "Rocky, the Gatewarden's papers please"
    assert lex.correct("Neshian's song, the Neshians' songs") == "Neshian's song, the Neshians' songs"
    assert lex.correct("a sennight's walk") == "a sennight's walk"
    assert lex.correct("Maliburn's vault") == "Maliburn's vault"
    assert lex.correct("the gate warden's office") == "the Gatewarden's office"
    assert lex.correct("gatewardens were rude") == "gatewardens were rude"
    assert lex.correct("Nephria's walls") == "Nephria's walls"


def test_a_misheard_possessive_is_corrected_and_keeps_its_suffix(tmp_path):
    (tmp_path / "l.tsv").write_text("Gatewarden\t\n", encoding="utf-8")
    lex = Lexicon.load(tmp_path / "l.tsv")
    assert lex.correct("the Gatewardin's papers") == "the Gatewarden's papers"
