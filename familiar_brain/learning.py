"""Catch-up learning: every session gets summarised, whether or not anybody ended it.

The summariser used to run only on POST /api/session/end. The one real session so far,
data/sessions/20260916-182232-0189cc.jsonl, was never ended, so nothing from it was ever
learnt, and apply_retention would have deleted its raw log a week later. Now any raw log
with no summary beside it, other than the session in progress, is queued on brain startup,
on /api/session/start and on /api/learn, and summarised in the background one at a time.

Never while a session is active: the summary is a generation on the same GPU the table is
waiting on (classify calls timed out at 1.5 s live on 2026-09-16, a reply took 60 s). A
summary in hand when a session starts stops before its next model call, with nothing
written, and goes back to the front of the queue; it and the rest wait for the next idle
moment, the next session end or start, or the next brain start.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Callable, TextIO

import requests

from familiar_brain.campaign import campaign_due, refresh_campaign
from familiar_brain.retrieval import Store
from familiar_brain.summariser import Paused, Purged, load_players, summarise_session

log = logging.getLogger("familiar.learning")


def _spawn(target: Callable[[], None]) -> threading.Thread:
    t = threading.Thread(target=target, daemon=True, name="learner")
    t.start()
    return t


def unsummarised(sessions_dir: Path | str, *, active: str | None = None) -> list[Path]:
    """Raw logs with no summary beside them, oldest first, leaving out the active session."""
    return [p for p in sorted(Path(sessions_dir).glob("*.jsonl"))
            if p.stem != active and not p.with_suffix(".summary.md").exists()]


def warn_if_no_players(path: Path | str) -> bool:
    """Say so, loudly, when there is no players file. It is gitignored and absent on a
    fresh machine (absent on the development machine on 2026-10-02), and without it
    scrub_players has no names to look for, so a summary that names somebody at the table
    keeps that line. The file is the table's to write; nothing here invents names for it."""
    if Path(path).exists():
        return False
    log.warning("%s is missing: player names will not be scrubbed from session summaries", path)
    return True


class Learner:
    def __init__(self, sessions_dir: Path | str, *, store: Store, llm, players_path: Path | str,
                 active: Callable[[], str | None] = lambda: None,
                 spawn: Callable[[Callable[[], None]], object] | None = None):
        self.dir = Path(sessions_dir)
        self.store = store
        self.llm = llm
        self.players_path = players_path
        self.active = active
        # Looked up at call time when not given, so a test can swap the module's _spawn.
        self._spawn = spawn
        self._lock = threading.Lock()
        self._queue: list[Path] = []
        self._current: Path | None = None
        self._running = False

    def queue(self, paths: list[Path]) -> None:
        with self._lock:
            for p in paths:
                if p not in self._queue and p != self._current:
                    self._queue.append(Path(p))

    def catch_up(self) -> list[str]:
        """Queue every unsummarised session and start on them if no session is active.
        Returns the ids now waiting or in hand, in the order they will be learnt."""
        self.queue(unsummarised(self.dir, active=self.active()))
        with self._lock:
            ids = ([self._current.stem] if self._current else []) + [p.stem for p in self._queue]
        self.kick()
        return ids

    def waiting(self) -> list[str]:
        with self._lock:
            return [p.stem for p in self._queue]

    def busy(self) -> set[str]:
        """Raw log names queued or being summarised, which retention must leave alone."""
        with self._lock:
            return {p.name for p in self._queue} | ({self._current.name} if self._current else set())

    def kick(self) -> bool:
        """Start the worker if there is work, it is not already running and no session is
        active. Returns whether it started. Work is a queued session, or a campaign memory
        that no longer matches the summaries (one purged since, or a merge that failed)."""
        with self._lock:
            if self._running or self.active() is not None:
                return False
            if not self._queue and not campaign_due(self.dir):
                return False
            self._running = True
        (self._spawn or _spawn)(self._drain)
        return True

    def _drain(self) -> None:
        refreshed = False
        while True:
            with self._lock:
                if self.active() is not None or (not self._queue and refreshed):
                    self._running = False
                    return
                self._current = self._queue.pop(0) if self._queue else None
            if self._current is None:
                # Queue empty: bring the campaign memory up to date once (a purge leaves it
                # to be rebuilt), then look at the queue again in case more arrived.
                refreshed = True
                self._refresh_campaign()
                continue
            refreshed = False
            # Skipped if purged since it was queued, or summarised since: catch_up lists the
            # logs outside the lock, so the worker can finish one between that listing and
            # queue(), which then queues it again.
            if not self._current.exists() or self._current.with_suffix(".summary.md").exists():
                with self._lock:
                    self._current = None
                continue
            try:
                learn_one(self._current, store=self.store, llm=self.llm, players_path=self.players_path,
                          stop=self._session_active)
            except Paused:
                # A session started mid-summary: nothing was written. Back to the front of
                # the queue, so it is the first thing learnt at the next idle moment.
                log.info("learning %s paused: a session started; it waits for the next idle moment",
                         self._current.name)
                with self._lock:
                    self._queue.insert(0, self._current)
            except Purged as e:
                log.info("%s", e)
            except Exception as e:
                # Left unsummarised, so the next catch-up offers it again. Not retried
                # here: a model that is down now is likely down for the next one too.
                log.error("learning %s failed, raw log kept: %s", self._current.name, e)
            finally:
                with self._lock:
                    self._current = None

    def _session_active(self) -> bool:
        return self.active() is not None

    def _refresh_campaign(self) -> None:
        if not campaign_due(self.dir):
            return
        try:
            refresh_campaign(self.dir, store=self.store, llm=self.llm,
                             players=load_players(self.players_path), stop=self._session_active)
        except Exception as e:
            log.error("campaign memory not updated, left for the next catch-up: %s", e)


def learn_one(path: Path, *, store: Store, llm, players_path: Path | str,
              stop: Callable[[], bool] = lambda: False) -> Path:
    """Summarise one session, then merge it into the campaign memory. The players file is
    read afresh each time, so one added while the brain runs is used from the next summary on.

    A failed campaign merge does not fail the session: its summary is written and stands on
    its own, and the memory stays due (campaign.campaign_due), so the next catch-up retries
    the merge without summarising the session again."""
    warn_if_no_players(players_path)
    players = load_players(players_path)
    out = summarise_session(path, store=store, llm=llm, players=players, stop=stop)
    try:
        refresh_campaign(Path(path).parent, store=store, llm=llm, players=players, stop=stop)
    except Exception as e:
        log.error("campaign memory not updated after %s, left for the next catch-up: %s", Path(path).name, e)
    return out


def learn_all(sessions_dir: Path | str, *, store: Store, llm, players_path: Path | str,
              out: TextIO | None = None) -> int:
    """Summarise every unsummarised session now, in order, printing progress. Returns how
    many failed. A failure leaves that raw log in place and moves on to the next."""
    out = out or sys.stdout
    pending = unsummarised(sessions_dir)
    if not pending:
        print(f"nothing to learn: every session in {sessions_dir} has a summary", file=out, flush=True)
        return _learn_campaign(sessions_dir, store=store, llm=llm, players_path=players_path, out=out)
    failed = 0
    for i, p in enumerate(pending, 1):
        print(f"{i}/{len(pending)} {p.stem} ...", file=out, flush=True)
        try:
            summary = learn_one(p, store=store, llm=llm, players_path=players_path)
        except requests.ConnectionError as e:
            failed += 1
            base = getattr(llm, "base_url", "the configured upstream")
            print(f"  failed: could not reach the model at {base}; is llama-server running? "
                  f"Its raw log is kept; run learn again once it is up. ({e})", file=out, flush=True)
            continue
        except Purged:
            print("  purged while it was being learnt; nothing written", file=out, flush=True)
            continue
        except Exception as e:
            failed += 1
            print(f"  failed: {type(e).__name__}: {e}. Its raw log is kept.", file=out, flush=True)
            continue
        print(f"  wrote {summary.name}", file=out, flush=True)
    print(f"learnt {len(pending) - failed} of {len(pending)}", file=out, flush=True)
    return failed + _learn_campaign(sessions_dir, store=store, llm=llm, players_path=players_path, out=out)


def _learn_campaign(sessions_dir: Path | str, *, store: Store, llm, players_path: Path | str,
                    out: TextIO) -> int:
    """Finish the campaign memory if a merge is still owed (learn_one does one after each
    summary; this catches a purge, or a merge that failed). Returns 1 if it failed."""
    if not campaign_due(sessions_dir):
        print("campaign memory is up to date", file=out, flush=True)
        return 0
    try:
        n = refresh_campaign(sessions_dir, store=store, llm=llm, players=load_players(players_path))
    except Exception as e:
        print(f"  campaign memory not updated: {type(e).__name__}: {e}. Run learn again.", file=out, flush=True)
        return 1
    print(f"campaign memory updated ({n} merge(s))", file=out, flush=True)
    return 0


def learn_command(args) -> None:
    from familiar_brain import indexing
    from familiar_brain.config import load_config
    from familiar_brain.llm import LLMClient
    cfg = load_config(args.config)
    store = indexing.open_store(cfg)
    llm = LLMClient(base_url=cfg.upstream.base_url, classify_slot=cfg.upstream.classify_slot,
                    classify_timeout_s=cfg.upstream.classify_timeout_s,
                    generate_slot=cfg.upstream.generate_slot)
    if learn_all(cfg.paths.sessions_dir, store=store, llm=llm, players_path=cfg.paths.players):
        raise SystemExit(1)
