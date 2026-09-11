# Notes policy

`data/notes/` is what the familiar remembers. Only two kinds of file go in it:
1. Fiction: world, city, factions, characters, NPCs, places, session recaps written as story.
2. Rules: the table's homebrew, under `data/notes/homebrew/`.

Never in `data/notes/`: recorder transcripts or their AI summaries, anything about the people at the table (health, jobs, habits, scheduling), or player names. Those recordings live in `data/raw/`, which the brain never reads and git never sees.

When adding a session recap, write it in fiction from the raw transcript, check it against `config/players.txt`, then delete or archive the raw file.

## Seeding a fresh clone

`.gitignore` excludes `data/` wholesale, so `data/notes/` is local only and a fresh
clone starts empty. The tracked source of truth for the seed corpus is `plans/0*.md`.
Rebuild the working copy with:

```bash
mkdir -p data/notes/homebrew data/raw
cp plans/0*.md data/notes/
mv data/notes/05-house-rules.md data/notes/homebrew/house-rules.md
```

Edit the copies under `data/notes/`, not the originals under `plans/`, unless you mean
to change the seed. Anything added to `data/notes/` afterwards, session recaps most of
all, exists only on that machine until somebody writes it back into `plans/`.
