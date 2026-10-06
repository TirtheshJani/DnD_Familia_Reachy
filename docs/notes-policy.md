# Notes policy

`data/notes/` is what the familiar remembers. Only two kinds of file go in it:
1. Fiction: world, city, factions, characters, NPCs, places, session recaps written as story.
2. Rules: the table's homebrew, under `data/notes/homebrew/`.

Never in `data/notes/`: recorder transcripts or their AI summaries, anything about the people at the table (health, jobs, habits, scheduling), or player names. Those recordings live in `data/raw/`, which the brain never reads and git never sees.

When adding a session recap, write it in fiction from the raw transcript, check it against `config/players.txt`, then delete or archive the raw file.

## Seeding a fresh clone

`.gitignore` excludes `data/` wholesale, so `data/notes/` is local only and a fresh
clone starts empty. The tracked source of truth for the seed corpus is the numbered
setting files, `plans/00-README.md` through `plans/10-nesh-canon.md`. The glob below
takes every `NN-*.md` and leaves out the dated plan documents (`plans/2026-...`), which
are not setting. An earlier `plans/0*.md` silently dropped file 10, the DM's canon.
Rebuild the working copy with:

```bash
mkdir -p data/notes/homebrew data/raw
cp plans/[0-9][0-9]-*.md data/notes/
mv data/notes/05-house-rules.md data/notes/homebrew/house-rules.md
# File 10 renames the gate settlement and makes Eris a Titan of Memory. Files 03, 04
# and 09 still say otherwise, and retrieval tags a chunk only with its heading, never
# its file, so a stale chunk would read as canon beside file 10. Correct the copies
# only. perl rather than sed -i, which BSD and macOS sed read differently.
perl -pi -e 's/through the North Reach Plaza/through Northreach Crossing/; s/^### North Reach Plaza/### Northreach Crossing/' data/notes/03-nephria-city.md
perl -pi -e 's/^\| North Reach Plaza \| \| entry slum \|/| Northreach Crossing | | entry slum |/' data/notes/09-lexicon.md
perl -pi -e 's/may worship Styxian gods such as Eris\./may worship Styxian gods. Eris is not one of them: she is a Titan of Memory (see 10-nesh-canon.md)./' data/notes/04-factions-and-powers.md
```

Edit the copies under `data/notes/`, not the originals under `plans/`, unless you mean
to change the seed. Anything added to `data/notes/` afterwards, session recaps most of
all, exists only on that machine until somebody writes it back into `plans/`.
