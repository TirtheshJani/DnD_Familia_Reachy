from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from familiar_brain.knowledge import CHUNK_CHARS
from familiar_brain.retrieval import Chunk

SOFT_CAP = 1200


def split_soft(text: str, cap: int = SOFT_CAP) -> list[str]:
    """Split on sentence ends so pieces stay under cap (plus one sentence of slack).

    Not after a list number ("8. Self: ..."), which would leave the number on one piece
    and its item on the next.
    """
    sents = re.split(r"(?<=[.!?])(?<!\d\.)\s+", text.strip())
    parts, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > cap:
            parts.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def markdown_chunks(md: str, *, source: str, corpus: str) -> list[Chunk]:
    out: list[Chunk] = []
    title, buf = Path(source).stem, []
    def flush():
        body = "\n".join(buf).strip()
        if body:
            # Notes split at the cap the prompt cuts each chunk to (knowledge.CHUNK_CHARS),
            # not SOFT_CAP: at 1200, file 10's Twelve Declarations was one chunk cut inside
            # declaration 8, so 9 to 12 never reached Rocky however well they ranked.
            for piece in split_soft(body, CHUNK_CHARS):
                out.append(Chunk(corpus=corpus, source=source, title=title, text=piece))
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            flush(); buf = []; title = m.group(1).strip()
        else:
            buf.append(line)
    flush()
    return out


def pdf_chunks_by_bookmarks(path: Path | str, *, corpus: str, edition: str) -> list[Chunk]:
    doc = pymupdf.open(str(path))
    toc = [(t, max(pg, 1)) for _lvl, t, pg in doc.get_toc()]
    if not toc:
        toc = [(f"Page {i + 1}", i + 1) for i in range(len(doc))]
    out: list[Chunk] = []
    for i, (title, start) in enumerate(toc):
        end = toc[i + 1][1] if i + 1 < len(toc) else len(doc)
        text = " ".join(doc[p].get_text() for p in range(start - 1, max(end, start)))
        text = re.sub(r"\s+", " ", text).strip()
        # trim the leading text that belongs to the previous section when two bookmarks share a page
        idx = text.find(title)
        if 0 < idx < 400:
            text = text[idx:]
        for piece in split_soft(text):
            out.append(Chunk(corpus=corpus, source=Path(path).name, title=title, text=piece,
                             meta={"edition": edition, "page": start}))
    return out
