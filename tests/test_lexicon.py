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
