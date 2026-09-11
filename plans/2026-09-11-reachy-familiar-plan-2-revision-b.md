# Reachy Familiar, Plan 2 Revision B: Campaign Context Changes

> **For agentic workers:** This file extends `2026-09-11-reachy-familiar-plan-2-live-loop-and-knowledge.md`. Execute the base plan's Tasks 1 to 8 first, then the tasks below in the order given (they slot in before the base plan's Task 9 and after its Task 11 as marked), then the base plan's Task 12. Same sub-skills, same ground rules.

**Why this revision exists:** the DM's worldbuilding recordings and the 10 September session transcript were reviewed on 2026-09-11. Five things changed: the notes corpus must be curated fiction, not raw recordings; the summariser needs a player-name guard; the table's homebrew overrides the SRD; the STT will mangle setting vocabulary; and deep night gives Plan 3 a free behaviour. The curated context set lives in `nephria-context/` and becomes the initial `data/notes/`.

**Spec deltas** (append to the Plan 2 spec's Decisions section):
- Notes corpus contains only curated fiction and rules files. Raw recorder transcripts and summaries are never placed under `data/notes/`. The curated set at `nephria-context/` is the seed.
- A fourth corpus, `homebrew`, is indexed from `data/notes/homebrew/`. `lookup_rule` searches it first and prefixes hits with "house rule"; SRD hits follow.
- `config/players.txt` lists the real first names of people at the table. The summariser drops any summary line containing one and reports how many it dropped. The file is gitignored.
- A lexicon (`config/lexicon.tsv`) maps mishearings to canonical setting terms. Transcript text is corrected before logging, gating, and retrieval.
- Plan 3 gets a deep-night rule: when the DM declares deep night, the familiar goes still and mute until it is declared over.

---

### Task R1: Seed the notes folder and protect it (after base Task 8)

**Files:**
- Create: `data/notes/` from `nephria-context/*.md` (copy, not move), `data/notes/homebrew/house-rules.md` (move `05-house-rules.md` here), `docs/notes-policy.md`
- Modify: `.gitignore` (add `data/raw/`, `config/players.txt`), `README.md`

- [ ] **Step 1: Lay out the folder**

```bash
mkdir -p data/notes/homebrew data/raw
cp nephria-context/*.md data/notes/
git mv data/notes/05-house-rules.md data/notes/homebrew/house-rules.md 2>/dev/null || mv data/notes/05-house-rules.md data/notes/homebrew/house-rules.md
mv <the seven recorder .txt files> data/raw/     # never indexed; gitignored
```

- [ ] **Step 2: Write `docs/notes-policy.md`**

```markdown
# Notes policy

`data/notes/` is what the familiar remembers. Only two kinds of file go in it:
1. Fiction: world, city, factions, characters, NPCs, places, session recaps written as story.
2. Rules: the table's homebrew, under `data/notes/homebrew/`.

Never in `data/notes/`: recorder transcripts or their AI summaries, anything about the people at the table (health, jobs, habits, scheduling), or player names. Those recordings live in `data/raw/`, which the brain never reads and git never sees.

When adding a session recap, write it in fiction from the raw transcript, check it against `config/players.txt`, then delete or archive the raw file.
```

- [ ] **Step 3: Update `.gitignore` and README**

Append `data/raw/` and `config/players.txt` to `.gitignore`. Add to `README.md` a one-line pointer to `docs/notes-policy.md`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: seed curated Nephria notes; notes policy; raw recordings excluded"
```

---

### Task R2: Homebrew corpus and `.txt` support in the indexer (after R1, before base Task 9)

**Files:**
- Modify: `familiar_brain/indexing.py` (`index_notes`), `familiar_brain/knowledge.py` (`CORPORA`)
- Test: `tests/test_indexing.py` (add), `tests/test_knowledge.py` (add)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_indexing.py`:
```python
def test_homebrew_subfolder_goes_to_homebrew_corpus_and_txt_is_indexed(tmp_path: Path):
    notes = tmp_path / "notes"; (notes / "homebrew").mkdir(parents=True)
    (notes / "homebrew" / "house-rules.md").write_text("# Injuries\nOn a crit you may take an injury instead of double damage.\n")
    (notes / "lore.txt").write_text("The Cobalt River separates Nephria from Xeria.\n")
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert index_notes(store, notes) == {"indexed": 2, "skipped": 0}
    assert store.search("injury crit", corpus="homebrew", k=1)[0].title == "Injuries"
    assert store.search("Cobalt River", corpus="notes", k=1)[0].source == "lore.txt"
    assert all(h.title != "Injuries" for h in store.search("injury", corpus="notes", k=3))  # house rule not in notes corpus
```

Append to `tests/test_knowledge.py`:
```python
def test_build_context_includes_homebrew_before_rules(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("rules", "srd.pdf", "Critical Hits", "crit doubles dice", {"edition": "2014"}),
                  Chunk("homebrew", "homebrew/house-rules.md", "Injuries", "crit may become an injury", {})])
    ctx = build_context(s, "what happens on a crit", chunks_per_corpus=3, max_chars=2000)
    assert ctx.index("[homebrew]") < ctx.index("[rules]")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_indexing.py tests/test_knowledge.py -v`
Expected: the two new tests fail (homebrew corpus empty; `.txt` not indexed).

- [ ] **Step 3: Modify `index_notes`**

Replace the file loop with:
```python
    for f in sorted(list(Path(notes_dir).glob("**/*.md")) + list(Path(notes_dir).glob("**/*.txt"))):
        src = str(f.relative_to(notes_dir))
        corpus = "homebrew" if src.startswith("homebrew/") or src.startswith("homebrew\\") else "notes"
        if src in known_all and f.stat().st_mtime <= known_all[src]:
            skipped += 1; continue
        store.delete_source(src)
        store.add_chunks(markdown_chunks(f.read_text(encoding="utf-8"), source=src, corpus=corpus))
        indexed += 1
```
and before the loop compute `known_all = {**store.sources("notes"), **store.sources("homebrew")}`. `markdown_chunks` already treats a file with no headings as one titled section, so `.txt` needs no new chunker.

- [ ] **Step 4: Modify `knowledge.CORPORA`**

```python
CORPORA = ("homebrew", "rules", "notes", "sessions")
```

- [ ] **Step 5: Run to verify they pass, then the whole suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(knowledge): homebrew corpus from notes/homebrew; index .txt notes; homebrew first in context"
```

---

### Task R3: `lookup_rule` searches homebrew first (replaces base Task 9's `lookup_rule.run`)

**Files:**
- Modify: `familiar_brain/tools/lookup_rule.py`
- Test: `tests/test_lookup_recall.py` (add)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lookup_recall.py`:
```python
def test_lookup_rule_prefers_house_rule(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([
        Chunk("homebrew", "homebrew/house-rules.md", "Sanity skills", "Revelation, Weird, Grounding, Entropic Paradox are Sanity skills.", {}),
        Chunk("rules", "srd51.pdf", "Wisdom", "Wisdom measures perception and insight.", {"edition": "2014"}),
    ])
    lookup_rule.bind(s)
    out = lookup_rule.run({"query": "how do Sanity skills work"})
    assert out.startswith("House rule") and "Sanity skills" in out
    assert "2014 rules" in out  # SRD still follows
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_lookup_recall.py -v`
Expected: the new test fails on `startswith("House rule")`.

- [ ] **Step 3: Replace `run` in `lookup_rule.py`**

```python
def run(args: dict) -> str:
    if _STORE is None:
        return "The rulebooks are not open."
    query = str(args.get("query", ""))
    lines = []
    house = _STORE.search(query, corpus="homebrew", k=2)
    for h in house:
        lines.append(f"House rule, {h.title}: {h.text[:500]}")
    best = {}
    for h in _STORE.search(query, corpus="rules", k=6):
        ed = h.meta.get("edition", "?")
        best.setdefault(ed, h)
    for ed, h in sorted(best.items()):
        lines.append(f"{ed} rules, {h.title}: {h.text[:600]}")
    if not lines:
        return "I found nothing on that in the rulebooks."
    if house:
        lines.append("The house rule takes precedence at this table; defer to the Dungeon Master.")
    return "\n".join(lines)
```

Update the tool's `SPEC["description"]` to: "Look up a rule. Returns the table's house rule first if one exists, then the best passage from the 2014 and 2024 rules. Say which you are citing and defer to the Dungeon Master."

- [ ] **Step 4: Run to verify, commit**

Run: `uv run pytest tests/test_lookup_recall.py -v`
```bash
git add -A && git commit -m "feat(tools): lookup_rule returns house rules before SRD"
```

---

### Task R4: Lexicon correction of transcripts (after R3, before base Task 10)

**Files:**
- Create: `familiar_brain/lexicon.py`, `config/lexicon.tsv`
- Modify: `familiar_brain/app.py` (apply before `state.remember`), `familiar_brain/config.py` (`paths.lexicon`)
- Test: `tests/test_lexicon.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_lexicon.py`:
```python
from pathlib import Path
from familiar_brain.lexicon import Lexicon

TSV = "Nephria\tNefrea|Nephria-Nephria\nStyx\tsticks\nGraywater Crossing\tGreywater Crossing|the Crossing\nThraka\tThraca|Brocka|Taraka\nAlar\tMalar|Molar|Pilar\n"


def make(tmp: Path) -> Lexicon:
    p = tmp / "lexicon.tsv"; p.write_text(TSV); return Lexicon.load(p)


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
    p = tmp_path / "e.tsv"; p.write_text("")
    assert Lexicon.load(p).correct("anything at all") == "anything at all"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_lexicon.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `familiar_brain/lexicon.py`**

```python
from __future__ import annotations

import difflib
import re
from pathlib import Path


class Lexicon:
    """Maps mishearings to canonical setting terms. TSV: canonical<TAB>variant|variant..."""

    def __init__(self, canon: list[str], variants: dict[str, str]):
        self.canon = canon
        self.variants = variants  # lowercase variant -> canonical
        longest_first = sorted(variants, key=len, reverse=True)
        self._pat = re.compile(r"\b(" + "|".join(re.escape(v) for v in longest_first) + r")\b", re.IGNORECASE) if variants else None
        self._canon_lower = {c.lower(): c for c in canon}

    @classmethod
    def load(cls, path: Path | str) -> "Lexicon":
        canon, variants = [], {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            c = parts[0].strip()
            canon.append(c)
            for v in (parts[1].split("|") if len(parts) > 1 else []):
                if v.strip():
                    variants[v.strip().lower()] = c
        return cls(canon, variants)

    def correct(self, text: str) -> str:
        if not self.canon:
            return text
        if self._pat:
            text = self._pat.sub(lambda m: self.variants[m.group(1).lower()], text)
        single = [c for c in self.canon if " " not in c]
        def fix(m: re.Match) -> str:
            w = m.group(0)
            if len(w) < 6 or w.lower() in self._canon_lower:
                return w
            hit = difflib.get_close_matches(w.lower(), [c.lower() for c in single], n=1, cutoff=0.86)
            return self._canon_lower[hit[0]] if hit else w
        return re.sub(r"[A-Za-z][A-Za-z'\-]+", fix, text)
```

- [ ] **Step 4: Write `config/lexicon.tsv`** from `nephria-context/09-lexicon.md`: one line per canonical term, variants pipe-separated, only where the table lists mishearings. Start with these lines and grow it after the live loop:

```
Nephria	Nefrea|Nephria-Nephria
Xeria	
Kulchith	
Styx	sticks
Blackwater	black water
Graywater Crossing	Greywater Crossing|Greywater
The Greyest Water	Greyist Water
The Drowned Pearl	Trial Pearl
Trinivane	Trinifane|Trinity Estate
ADA	Ada
Thraka	Thraca|Brocka|Taraka
Stormak	Tormek|Tarmac
Alar	Malar|Molar|Pilar
Nixon	Nexon
Jam	Jem|Gem
Mr Boom Boom	Mr Booze
Entropic Paradox	Void Paradox
```

Add `lexicon = "config/lexicon.tsv"` under `[paths]` in `brain.toml` and the `Paths` dataclass.

- [ ] **Step 5: Apply in `app.py`**

In `create_app`: `lexicon = Lexicon.load(cfg.paths.lexicon)`. In the `/v1/responses` handler, immediately after `text = latest_user_text(req)`: `text = lexicon.correct(text)`. The corrected text is what gets logged, gated, retrieved on, and forwarded (replace the last user item's content in `outgoing` with the corrected text so the model sees canonical names). Add an integration assertion in `tests/test_app_knowledge.py`: post "Thraca, hello" and check the logged text reads "Thraka, hello".

- [ ] **Step 6: Run everything, commit**

Run: `uv run pytest -q`
```bash
git add -A && git commit -m "feat(brain): lexicon correction of transcripts before gate, log, and retrieval"
```

---

### Task R5: Player-name guard in the summariser (after base Task 11)

**Files:**
- Create: `config/players.txt` (gitignored; TJ fills it), `config/players.example.txt`
- Modify: `familiar_brain/summariser.py`
- Test: `tests/test_summariser.py` (add)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_summariser.py`:
```python
from familiar_brain.summariser import scrub_players


def test_scrub_players_drops_lines_and_counts():
    md = "## Scenes\n- The cistern\n- James rolled a natural twenty\n## NPCs met\n- Elias the barkeep\n"
    out, dropped = scrub_players(md, players=["James", "Elias"])
    assert dropped == 2 and "James" not in out and "Elias" not in out and "- The cistern" in out


def test_scrub_players_is_word_bounded_and_case_insensitive():
    out, dropped = scrub_players("- Jameson Street\n- jax laughed\n", players=["James", "Jax"])
    assert dropped == 1 and "Jameson Street" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_summariser.py -v`
Expected: FAIL, `scrub_players` not found.

- [ ] **Step 3: Implement**

Add to `summariser.py`:
```python
import re


def load_players(path: Path | str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def scrub_players(md: str, *, players: list[str]) -> tuple[str, int]:
    if not players:
        return md, 0
    pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in players) + r")\b", re.IGNORECASE)
    kept, dropped = [], 0
    for line in md.splitlines():
        if pat.search(line):
            dropped += 1
        else:
            kept.append(line)
    return "\n".join(kept) + ("\n" if md.endswith("\n") else ""), dropped
```

In `summarise_session`, after `md = llm.complete(...)`:
```python
    md, dropped = scrub_players(md, players=players)
    if dropped:
        log.warning("summariser dropped %d line(s) naming players from %s", dropped, Path(path).name)
```
with `players: list[str] = ()` added as a keyword argument, and `create_app` passing `load_players("config/players.txt")` through `summarise_in_background`.

- [ ] **Step 4: Write `config/players.example.txt`** and tell TJ to copy it to `config/players.txt` and edit

```
# One real first name per line. Gitignored. Used only to drop summary lines that mention people at the table.
# Names heard in the 10 Sept recording that look like players rather than characters; verify each:
James
Elias
Jax
Jacob
Harrison
Adrian
Riley
Gabe
```

- [ ] **Step 5: Run everything, commit**

Run: `uv run pytest -q`
```bash
git add -A && git commit -m "feat(summariser): drop summary lines naming players; players.txt gitignored"
```

---

### Task R6: Live-loop script additions (edit base Task 3's `docs/live-loop.md`)

- [ ] **Step 1:** Add a "Vocabulary" block to the script, read after line 13: "Nephria. Xeria. Kulchith. Styx. Blackwater. Graywater Crossing. Thraka. Stormak. Alar. Nixon. Clover. Jam. Trinivane. Ildevar. Durnagall." Expected: each appears in the session log corrected to canonical spelling; any that do not get their heard form added to `config/lexicon.tsv`.
- [ ] **Step 2:** Add a line 15 to the script: "Deep night. Everyone quiet." Expected in Plan 2: IGNORE. Note for Plan 3: this is the trigger for the deep-night mute behaviour.
- [ ] **Step 3:** Commit with the base plan's Task 3.

---

### Task R7: Plan 3 note (no code now)

Append to `docs/superpowers/specs/2026-09-10-reachy-familiar-design.md` section 4.7:

"Deep night. When an utterance matches the deep-night rule (the phrase 'deep night' spoken by the DM, or a Speak Now note 'deep night'), the familiar enters a still state: antennas down, head lowered, gate forced to IGNORE, camera closed, until an utterance matches 'deep night is over' or Speak Now 'morning'. This is a Plan 3 motion behaviour built on the Plan 2 mute state."

---

## Self-review against the revision

Curation: R1 (folder, policy, gitignore). Homebrew before SRD: R2 (corpus, context order), R3 (tool). Lexicon: R4 (module, config, wiring, integration assertion) and R6 (live check). Player guard: R5. Deep night: R6 (script line) and R7 (spec note). `.txt` support: R2. Ordering dependency: R2 before base Task 9 because `lookup_rule` in R3 assumes the `homebrew` corpus; R4 before base Task 10 because `app.py` is edited in both and R4's wiring sits next to `state.remember`.

Type consistency: `Store.sources(corpus)` from base Task 6 is reused in R2; `Chunk` positional order unchanged; `Lexicon.load(path).correct(text)` is the only lexicon API; `scrub_players(md, players=...) -> (str, int)` matches its tests and its call site.
