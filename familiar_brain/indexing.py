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
    """Index every markdown note, skipping files unchanged since they were last indexed.

    Source keys are stored posix-style so a store built on Windows and one built in
    WSL agree, which matters because both run against the same repo.
    """
    known = store.sources("notes")
    indexed = skipped = 0
    for md in sorted(Path(notes_dir).glob("**/*.md")):
        src = md.relative_to(notes_dir).as_posix()
        if src in known and md.stat().st_mtime <= known[src]:
            skipped += 1
            continue
        store.delete_source(src)
        store.add_chunks(markdown_chunks(md.read_text(encoding="utf-8"), source=src, corpus="notes"))
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
