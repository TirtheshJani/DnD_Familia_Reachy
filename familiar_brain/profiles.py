"""Switchable Campaign / Team Profiles, dynamic Party Roster, and Privacy Names.

Keeps Rocky's identity, TARS dials, and D&D rules shared across teams while isolating
each team's Party Roster, Notes, Campaign Memory, Sessions, Speaker Voice Profiles,
Real-Player Privacy Names, and setting/secrets prompt so different tables never mix.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_NEPHRIA_ROSTER: list[str] = ["DM", "Brokk", "Krag", "Till", "Eberhard"]
DEFAULT_NEPHRIA_SETTING = "the city of Nephria, an Art Deco noir setting"
DEFAULT_NEPHRIA_BOUND_TO = "Brokk Trenevain"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug[:48] or "campaign"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except PermissionError:
        time.sleep(0.02)
        os.replace(tmp, path)


def read_privacy_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            names.append(line)
    return names


def write_privacy_names(path: Path, names: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for n in names:
        s = str(n).strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(s)
    header = (
        "# One real first name per line. Used to drop summary lines that mention people\n"
        "# at the table so the familiar's memory stays fiction-only.\n"
    )
    body = header + ("\n".join(cleaned) + "\n" if cleaned else "")
    _write_atomic(path, body)
    return cleaned


@dataclass(frozen=True)
class CampaignProfile:
    id: str
    name: str
    setting: str
    bound_to: str
    secrets: str
    roster: tuple[str, ...]
    is_nephria: bool
    notes_dir: Path
    sessions_dir: Path
    speakers_path: Path
    players_path: Path

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "setting": self.setting,
            "bound_to": self.bound_to,
            "is_nephria": self.is_nephria,
            "roster_count": len(self.roster),
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "setting": self.setting,
            "bound_to": self.bound_to,
            "secrets": self.secrets,
            "roster": list(self.roster),
            "is_nephria": self.is_nephria,
            "privacy_names": read_privacy_names(self.players_path),
        }


class CampaignManager:
    """Thread-safe registry for switchable D&D campaigns, rosters, and privacy name lists."""

    def __init__(
        self,
        *,
        data_dir: Path,
        default_notes_dir: Path,
        default_sessions_dir: Path,
        default_speakers_path: Path,
        default_players_path: Path,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.index_path = self.data_dir / "campaigns_index.json"
        self.campaigns_root = self.data_dir / "campaigns"
        self._defaults = {
            "notes_dir": Path(default_notes_dir),
            "sessions_dir": Path(default_sessions_dir),
            "speakers_path": Path(default_speakers_path),
            "players_path": Path(default_players_path),
        }
        self._lock = threading.RLock()
        self._active_id: str = "nephria"
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _default_nephria_record(self) -> dict[str, Any]:
        return {
            "id": "nephria",
            "name": "Shadows over Nephria",
            "setting": DEFAULT_NEPHRIA_SETTING,
            "bound_to": DEFAULT_NEPHRIA_BOUND_TO,
            "secrets": "",
            "roster": list(DEFAULT_NEPHRIA_ROSTER),
            "is_nephria": True,
        }

    def _load(self) -> None:
        with self._lock:
            self._records = {"nephria": self._default_nephria_record()}
            self._active_id = "nephria"
            if not self.index_path.exists():
                return
            try:
                raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                return
            if not isinstance(raw, dict):
                return
            for cid, rec in (raw.get("profiles") or {}).items():
                if not isinstance(cid, str) or not isinstance(rec, dict):
                    continue
                roster = [str(r).strip() for r in (rec.get("roster") or []) if str(r).strip()]
                if not roster:
                    roster = ["DM"]
                self._records[cid] = {
                    "id": cid,
                    "name": str(rec.get("name") or cid).strip(),
                    "setting": str(rec.get("setting") or "").strip(),
                    "bound_to": str(rec.get("bound_to") or "").strip(),
                    "secrets": str(rec.get("secrets") or "").strip(),
                    "roster": roster,
                    "is_nephria": bool(cid == "nephria" or rec.get("is_nephria")),
                }
            if "nephria" not in self._records:
                self._records["nephria"] = self._default_nephria_record()
            active = str(raw.get("active") or "nephria")
            self._active_id = active if active in self._records else "nephria"

    def _save_locked(self) -> None:
        payload = {"active": self._active_id, "profiles": self._records}
        _write_atomic(self.index_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def _materialize(self, rec: dict[str, Any]) -> CampaignProfile:
        cid = rec["id"]
        if cid == "nephria":
            notes_dir = self._defaults["notes_dir"]
            sessions_dir = self._defaults["sessions_dir"]
            speakers_path = self._defaults["speakers_path"]
            players_path = self._defaults["players_path"]
        else:
            root = self.campaigns_root / cid
            notes_dir = root / "notes"
            sessions_dir = root / "sessions"
            speakers_path = root / "speakers.json"
            players_path = root / "players.txt"
        return CampaignProfile(
            id=cid,
            name=rec["name"],
            setting=rec.get("setting", ""),
            bound_to=rec.get("bound_to", ""),
            secrets=rec.get("secrets", ""),
            roster=tuple(rec.get("roster") or ["DM"]),
            is_nephria=bool(rec.get("is_nephria", cid == "nephria")),
            notes_dir=notes_dir,
            sessions_dir=sessions_dir,
            speakers_path=speakers_path,
            players_path=players_path,
        )

    def active(self) -> CampaignProfile:
        with self._lock:
            return self._materialize(self._records[self._active_id])

    def list_profiles(self) -> list[CampaignProfile]:
        with self._lock:
            ordered = ["nephria"] + sorted(k for k in self._records if k != "nephria")
            return [self._materialize(self._records[k]) for k in ordered]

    def create_profile(
        self,
        name: str,
        *,
        setting: str = "",
        bound_to: str = "",
        roster: list[str] | None = None,
        secrets: str = "",
    ) -> CampaignProfile:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("campaign name is required")
        with self._lock:
            base_slug = slugify(clean_name)
            cid = base_slug
            suffix = 2
            while cid in self._records:
                cid = f"{base_slug}-{suffix}"
                suffix += 1
            clean_roster: list[str] = []
            seen: set[str] = set()
            for r in (roster or ["DM"]):
                s = str(r).strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    clean_roster.append(s)
            if "dm" not in seen:
                clean_roster.insert(0, "DM")
            rec = {
                "id": cid,
                "name": clean_name,
                "setting": str(setting or "a tabletop Dungeons & Dragons adventure").strip(),
                "bound_to": str(bound_to or "the party").strip(),
                "secrets": str(secrets or "").strip(),
                "roster": clean_roster,
                "is_nephria": False,
            }
            prof = self._materialize(rec)
            prof.notes_dir.mkdir(parents=True, exist_ok=True)
            prof.sessions_dir.mkdir(parents=True, exist_ok=True)
            if not prof.players_path.exists():
                write_privacy_names(prof.players_path, [])
            self._records[cid] = rec
            self._save_locked()
            return prof

    def switch(self, profile_id: str) -> CampaignProfile:
        cid = str(profile_id or "").strip()
        with self._lock:
            if cid not in self._records:
                raise ValueError(f"unknown campaign profile: {cid!r}")
            self._active_id = cid
            prof = self._materialize(self._records[cid])
            prof.notes_dir.mkdir(parents=True, exist_ok=True)
            prof.sessions_dir.mkdir(parents=True, exist_ok=True)
            self._save_locked()
            return prof

    def delete_profile(self, profile_id: str) -> bool:
        cid = str(profile_id or "").strip()
        with self._lock:
            if cid == "nephria":
                raise ValueError("cannot delete the default nephria profile")
            if cid == self._active_id:
                raise ValueError("switch to another campaign before deleting the active profile")
            if cid not in self._records:
                return False
            self._records.pop(cid, None)
            self._save_locked()
            folder = self.campaigns_root / cid
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            return True

    def add_roster_role(self, role: str) -> list[str]:
        clean = str(role or "").strip()
        if not clean:
            raise ValueError("role name is required")
        with self._lock:
            rec = self._records[self._active_id]
            roster = list(rec.get("roster") or [])
            if all(r.lower() != clean.lower() for r in roster):
                roster.append(clean)
                rec["roster"] = roster
                self._save_locked()
            return list(rec["roster"])

    def remove_roster_role(self, role: str) -> list[str]:
        clean = str(role or "").strip().lower()
        if not clean:
            raise ValueError("role name is required")
        with self._lock:
            rec = self._records[self._active_id]
            roster = [r for r in (rec.get("roster") or []) if r.lower() != clean]
            if not roster:
                roster = ["DM"]
            rec["roster"] = roster
            self._save_locked()
            return list(rec["roster"])

    def add_privacy_name(self, name: str) -> list[str]:
        clean = str(name or "").strip()
        if not clean:
            raise ValueError("name is required")
        with self._lock:
            p = self.active().players_path
            names = read_privacy_names(p)
            if all(n.lower() != clean.lower() for n in names):
                names.append(clean)
            return write_privacy_names(p, names)

    def remove_privacy_name(self, name: str) -> list[str]:
        clean = str(name or "").strip().lower()
        if not clean:
            raise ValueError("name is required")
        with self._lock:
            p = self.active().players_path
            names = [n for n in read_privacy_names(p) if n.lower() != clean]
            return write_privacy_names(p, names)
