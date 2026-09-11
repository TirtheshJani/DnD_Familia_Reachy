from __future__ import annotations

import logging
from pathlib import Path

from familiar_brain.chunking import markdown_chunks, pdf_chunks_by_bookmarks
from familiar_brain.retrieval import Store

log = logging.getLogger("familiar.indexing")


def index_rules(store: Store, pdf: Path, *, edition: str) -> int:
    """Replace every chunk from this PDF. Indexing the same file twice is a no-op, not a duplicate."""
    store.delete_source(Path(pdf).name)
    n = store.add_chunks(pdf_chunks_by_bookmarks(pdf, corpus="rules", edition=edition))
    log.info("indexed %d rule chunks from %s (edition %s)", n, Path(pdf).name, edition)
    return n


def index_notes(store: Store, notes_dir: Path) -> dict[str, int]:
    """Index every note, skipping files unchanged since they were last indexed.

    Anything under `homebrew/` becomes the `homebrew` corpus, which lookup_rule
    searches before the SRD, because the table's own rules beat the book. Everything
    else is `notes`. Source keys are stored posix-style so a store built on Windows
    and one built in WSL agree, which matters because both run against the same repo.
    `.txt` is read too; markdown_chunks treats a file with no headings as one section,
    so plain text needs no separate chunker.
    """
    # sources() is per corpus, so both have to be merged or every homebrew file would
    # look unknown to a notes-only lookup and be re-chunked on every pass.
    known = {**store.sources("notes"), **store.sources("homebrew")}
    indexed = skipped = 0
    files = sorted(list(Path(notes_dir).glob("**/*.md")) + list(Path(notes_dir).glob("**/*.txt")))
    for f in files:
        src = f.relative_to(notes_dir).as_posix()
        corpus = "homebrew" if src.startswith("homebrew/") else "notes"
        if src in known and f.stat().st_mtime <= known[src]:
            skipped += 1
            continue
        store.delete_source(src)
        store.add_chunks(markdown_chunks(f.read_text(encoding="utf-8"), source=src, corpus=corpus))
        indexed += 1
    log.info("notes indexed=%d skipped=%d from %s", indexed, skipped, notes_dir)
    return {"indexed": indexed, "skipped": skipped}


def open_store(cfg) -> Store:
    """The single seam the app and the CLI share, and the one the tests monkeypatch."""
    from familiar_brain.embeddings import BgeEmbedder
    Path(cfg.knowledge.db_path).parent.mkdir(parents=True, exist_ok=True)
    return Store(cfg.knowledge.db_path, embedder=BgeEmbedder())


def index_command(args) -> None:
    from familiar_brain.config import load_config
    cfg = load_config(args.config)
    store = open_store(cfg)
    if args.corpus == "rules":
        if not args.path or not args.edition:
            raise SystemExit("index rules <pdf> --edition 2014|2024")
        n = index_rules(store, Path(args.path), edition=args.edition)
        print(f"indexed {n} rule chunks from {args.path} (edition {args.edition})")
    else:
        print(index_notes(store, Path(args.path or cfg.knowledge.notes_dir)))
