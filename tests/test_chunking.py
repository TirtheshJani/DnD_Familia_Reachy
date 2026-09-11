from pathlib import Path
import pymupdf
from familiar_brain.chunking import markdown_chunks, pdf_chunks_by_bookmarks, split_soft


def test_markdown_chunks_by_heading_carry_title():
    md = "# Backstory\nDocks of Nephria.\n\n## Contacts\nThe barkeep.\nOwes gold.\n"
    out = markdown_chunks(md, source="pc.md", corpus="notes")
    assert [c.title for c in out] == ["Backstory", "Contacts"]
    assert out[1].text.startswith("The barkeep") and out[1].source == "pc.md"


def test_split_soft_respects_cap_and_sentence_boundary():
    text = "Sentence one is here. " * 100
    parts = split_soft(text, cap=300)
    assert all(len(p) <= 330 for p in parts) and all(p.rstrip().endswith(".") for p in parts)


def test_pdf_chunks_use_bookmarks_and_edition(tmp_path: Path):
    doc = pymupdf.open()
    for i, (title, body) in enumerate([("Grappling", "You can grapple. " * 20), ("Fireball", "A bright streak. " * 20)]):
        page = doc.new_page(); page.insert_text((72, 72), body)
    doc.set_toc([[1, "Grappling", 1], [1, "Fireball", 2]])
    p = tmp_path / "srd.pdf"; doc.save(p)
    out = pdf_chunks_by_bookmarks(p, corpus="rules", edition="2014")
    assert [c.title for c in out] == ["Grappling", "Fireball"]
    assert all(c.meta["edition"] == "2014" for c in out) and "grapple" in out[0].text.lower()
