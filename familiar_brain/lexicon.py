from __future__ import annotations

import difflib
import re
from pathlib import Path

# Below this length a token is too likely to be an ordinary English word, or a real
# name that merely resembles a setting term, for fuzzy matching to be safe. "Alan"
# against "Alar" is the case that motivated the floor.
MIN_FUZZY_LEN = 6
FUZZY_CUTOFF = 0.86


class Lexicon:
    """Maps mishearings to canonical setting terms. TSV: canonical<TAB>variant|variant..."""

    def __init__(self, canon: list[str], variants: dict[str, str]):
        self.canon = canon
        self.variants = variants  # lowercase variant -> canonical
        # Longest first so "Greywater Crossing" wins over any shorter variant inside it.
        longest_first = sorted(variants, key=len, reverse=True)
        self._pat = (re.compile(r"\b(" + "|".join(re.escape(v) for v in longest_first) + r")\b",
                                re.IGNORECASE) if variants else None)
        self._canon_lower = {c.lower(): c for c in canon}

    @classmethod
    def load(cls, path: Path | str) -> "Lexicon":
        p = Path(path)
        if not p.exists():  # a config may point at a lexicon nobody has written yet
            return cls([], {})
        canon: list[str] = []
        variants: dict[str, str] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
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
        """Replace known mishearings, then fuzzily repair long tokens close to a canonical term."""
        if not self.canon:
            return text
        if self._pat:
            text = self._pat.sub(lambda m: self.variants[m.group(1).lower()], text)
        # Fuzzy matching is single-word only: a multi-word canonical term cannot be
        # recovered from one token, and its variants are already handled above.
        single = [c for c in self.canon if " " not in c]
        single_lower = [c.lower() for c in single]

        def fix(m: re.Match) -> str:
            w = m.group(0)
            if len(w) < MIN_FUZZY_LEN or w.lower() in self._canon_lower:
                return w
            hit = difflib.get_close_matches(w.lower(), single_lower, n=1, cutoff=FUZZY_CUTOFF)
            return self._canon_lower[hit[0]] if hit else w

        return re.sub(r"[A-Za-z][A-Za-z'\-]+", fix, text)
