import json
from pathlib import Path

import pytest

from familiar_brain import indexing
from familiar_brain.app import create_app
from familiar_brain.retrieval import Chunk, Store
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    fake = FakeLLM()
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    store.add_chunks([
        Chunk("rules", "srd51.pdf", "Grappling", "Grapple rules text 2014.", {"edition": "2014"}),
        Chunk("homebrew", "homebrew/house-rules.md", "Injuries", "On a crit take an injury.", {}),
    ])
    monkeypatch.setattr(indexing, "open_store", lambda cfg: store)
    (tmp_path / "persona.yaml").write_text("name: Understudy\naliases: []\ntts_speaker: A\nmanner: dry\n")
    (tmp_path / "notes").mkdir()
    lex = tmp_path / "lexicon.tsv"
    lex.write_text("Thraka\tThraca|Brocka\nNephria\tNefrea\n", encoding="utf-8")
    (tmp_path / "brain.toml").write_text(f"""
[server]
host = "127.0.0.1"
port = 0
[upstream]
base_url = "{fake.base_url}"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 1.0
[gate]
follow_up_window_s = 8
[paths]
persona = "{(tmp_path / 'persona.yaml').as_posix()}"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
lexicon = "{lex.as_posix()}"
[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
""")
    app = create_app(tmp_path / "brain.toml")
    app.config["TESTING"] = True
    yield app.test_client(), fake, store, tmp_path
    fake.close()


def _addressed(client, text):
    resp = client.post("/v1/responses", json={"input": [{"role": "user", "content": text}],
                                              "stream": True, "tools": []})
    resp.get_data()  # consume the SSE stream so the generator's finally block runs
    return resp


def test_addressed_turn_carries_what_you_know(env):
    client, fake, _, _ = env
    client.post("/api/session/start")
    _addressed(client, "Understudy, how does grapple work?")
    gen = [r for r in fake.requests if r["path"].endswith("/responses")]
    assert gen, "the addressed turn never reached the upstream"
    instr = gen[-1]["body"]["instructions"]
    assert "What you know:" in instr and "Grappling" in instr
    assert instr.index("What you know:") < instr.index("You are Understudy")


def test_ignored_turn_does_no_generation(env):
    client, fake, _, _ = env
    client.post("/api/session/start")
    client.post("/v1/responses", json={"input": [{"role": "user", "content": "the door creaks"}],
                                       "stream": True}).get_data()
    assert not any(r["path"].endswith("/responses") for r in fake.requests)


def test_retrieval_is_timed_and_the_tools_are_offered(env, caplog):
    client, fake, _, _ = env
    client.post("/api/session/start")
    with caplog.at_level("INFO", logger="familiar.timing"):
        _addressed(client, "Understudy, what is a grapple?")
    assert any("retrieval_ms" in r.message for r in caplog.records), "retrieval was not timed"
    tools = {t["name"] for t in fake.requests[-1]["body"]["tools"]}
    assert {"roll", "initiative", "note", "lookup_rule", "recall"} <= tools


def test_lexicon_corrects_the_logged_and_forwarded_text(env):
    """R4: the corrected text is what gets logged, gated, retrieved on and forwarded."""
    client, fake, _, tmp = env
    sid = client.post("/api/session/start").get_json()["session"]
    _addressed(client, "Understudy, Thraca went to Nefrea")
    rows = [json.loads(l) for l in (tmp / "sessions" / f"{sid}.jsonl").read_text().splitlines() if l.strip()]
    assert rows[-1]["text"] == "Understudy, Thraka went to Nephria"
    sent = fake.requests[-1]["body"]["input"][-1]
    assert "Thraka" in json.dumps(sent) and "Thraca" not in json.dumps(sent)


def test_reindex_endpoint_reports_counts(env):
    client, _, _, tmp = env
    (tmp / "notes" / "pc.md").write_text("# PC\nDocks of Nephria.\n")
    assert client.post("/api/reindex").get_json() == {"indexed": 1, "skipped": 0}
