# Pinned versions and measurements

Everything below was read off the installed software or the live robot. Nothing is
copied from the design document. Host: Zephyrus G14, Ryzen 9 7940HS, 16 GB RAM,
RTX 4080 Laptop 12282 MiB, Windows 11 build 26200, WSL2 Ubuntu 24.04.

## Pinned versions

| Component | Version | Source |
|---|---|---|
| speech-to-speech | 1.0.0 | PyPI |
| openai SDK (the client the brain must satisfy) | 2.28.0 | dependency of speech-to-speech |
| llama.cpp | build 10751, variant `cuda130_hd5b56ce_0` | conda-forge |
| llama-server reported version | 0.3.0-dev, commit 125c6d1 | `llama-server --version` |
| Gemma 4 E4B QAT GGUF | `unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL` (sha `8c5a9e4fd548`) + `mmproj-BF16/F16` | Hugging Face |
| Kokoro-82M TTS | `hexgrad/Kokoro-82M` (`kokoro==0.9.4`, `espeakng-loader==0.2.4`, `bm_fable`) | Hugging Face / PyPI |
| Qwen3-TTS (benchmarked fallback) | `Serveurperso/Qwen3-TTS-GGUF` `0.6b-customvoice-Q8_0` / `BF16` | Hugging Face |
| Parakeet-TDT | `nvidia/parakeet-tdt-0.6b-v3` on CUDA | Hugging Face, package default |
| Smart Turn VAD | v3.2, `pipecat-ai/smart-turn-v3`, CPU provider | loaded automatically |
| torch | 2.14.0+cu130, `cuda_available True` | PyPI |
| transformers | 5.17.0 | PyPI |
| Python (WSL2 runtime) | 3.12.3 | Ubuntu 24.04 |
| Python (Windows test runs) | 3.11.6 | python.org |
| uv | 0.12.13 | |
| reachy_mini daemon | 1.10.0, wireless, hardware id `3ce93bd07b69ca28` | `GET /api/daemon/status` |

> **Note (Wave 1 Production Stack):** The **Plan 1 / Plan 2** notes and baseline tables immediately below (`:Q4_0`, `qwen3`, and initial VRAM/latency measurements) are **superseded** by the **M5 Production Stack Upgrade & Benchmarks (2026-11-23)** section at the bottom of this file (`unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL` + `q8_0` KV cache + `Kokoro-82M`).

Two flags on the historical Plan 1 `llama-server` line were load-bearing and easy to lose in a tidy-up:

- **`:Q4_0` (Plan 1 baseline, superseded by `:UD-Q4_K_XL`).** `-hf ...GGUF` with no suffix resolves to Q8_0 at 8.03 GB, which does not
  leave room for Parakeet and TTS on a 12 GB card. Q4_0 was 4.59 GB; the current production stack uses Gemma 4 E4B-it QAT `UD-Q4_K_XL` with `-ctk q8_0 -ctv q8_0`.
- **`--jinja`.** Without it llama-server ignores the `tools` array completely and
  returns `tool_calls: null`, so `roll`, `initiative` and `note` can never fire. Verified
  both ways against the live server.

## Install, without sudo

The distro had no `pip`, `gcc`, `cmake` or `nvcc`, and `sudo` needs a password, so
there is no apt step and no source build. Everything lives under `$HOME`.

```bash
# uv
curl -fsSL https://astral.sh/uv/install.sh | sh          # -> ~/.local/bin/uv

# llama.cpp with CUDA, prebuilt rather than compiled
curl -fsSL -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
CONDA_OVERRIDE_CUDA=13.0 conda create -y -n llama -c conda-forge "llama.cpp=*=*cuda*"
# -> ~/miniforge3/envs/llama/bin/llama-server

# speech-to-speech
uv venv "$HOME/.venvs/s2s" --python 3.12
VIRTUAL_ENV="$HOME/.venvs/s2s" uv pip install speech-to-speech

# CUDA 12 shims, see the next section
VIRTUAL_ENV="$HOME/.venvs/s2s" uv pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

# the brain itself needs a Linux venv, because the repo's .venv holds Windows binaries
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/brain"
uv sync --extra dev
```

### The CUDA 12 versus CUDA 13 trap

`speech-to-speech` pulls torch `2.14.0+cu130`, so the venv ships only
`libcudart.so.13`. The ggml TTS backend (`qwentts-cpp-python`, under
`faster-qwen3-tts`) is built against CUDA 12 and dlopens `libcudart.so.12` and
`libcublas.so.12`. The pipeline therefore dies at TTS setup with:

```
OSError: libcudart.so.12: cannot open shared object file: No such file or directory
```

Fix: install the two cu12 shim wheels above (3.3 MiB and ~600 MB) and put the nvidia
lib directories on `LD_LIBRARY_PATH`, which `scripts/start_all.sh` does. The
alternative is `--qwen3_tts_backend torch`, which avoids the ggml path entirely but
costs more VRAM and latency.

## WSL2 configuration, `%USERPROFILE%\.wslconfig`

```ini
[wsl2]
memory=12GB
processors=8
networkingMode=mirrored

[experimental]
hostAddressLoopback=true
```

Both settings are load-bearing:

- **Networking.** The default NAT mode put the distro on `192.168.94.249`. The robot
  lives on `10.0.0.100` and cannot route there, so the robot could never reach
  speech-to-speech regardless of bind address. Mirrored mode gives the distro the
  host's own LAN address, `10.0.0.168`, which is what the robot must be pointed at.
- **Memory.** The default ceiling was 7 GB, below what Parakeet plus Qwen3-TTS plus
  llama-server need.

Apply with `wsl --shutdown`. Revert by deleting the file and shutting down again.

## Firewall, one elevated step still required

The active network profile is **Public**, and the Hyper-V firewall that governs
mirrored-mode WSL traffic reports `DefaultInboundAction: Block`. So nothing on the LAN,
including the robot, can reach port 8765 until a rule exists. This needs an
**administrator** PowerShell and could not be done from this session:

```powershell
New-NetFirewallHyperVRule -Name "WSL-s2s-8765" -DisplayName "speech-to-speech 8765 (WSL)" `
  -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol TCP -LocalPorts 8765 -Action Allow
```

Confirm afterwards from another device on the LAN:
`curl -v telnet://<host-lan-ip>:8765` should connect rather than hang.

### The host LAN address moves

It was 10.0.0.168 at the start of this session and 10.0.0.95 by the end, because the
address comes from DHCP and mirrored mode hands the distro whatever the host has.
Whatever the robot's connection URL is pointed at will therefore go stale on a lease
change. Either reserve a lease for the laptop on the router, or re-read the address
from `./scripts/check_env.sh` before a session and update the robot.

Note also that WSL has no mDNS resolver, so `reachy-mini.local` fails from inside the
distro even while resolving fine from Windows. Use the robot's address there, or pass
`ROBOT_HOST=<ip>` to `check_env.sh`. The robot was reachable throughout at
`10.0.0.100:8000` (`reachy_mini`, firmware 1.10.0).

## Working commands (Historical Plan 1 Baseline — Superseded by M5 Below)

> **Superseded:** Use `./scripts/start_all.sh` (configured with Gemma 4 E4B-it QAT `UD-Q4_K_XL` and `Kokoro-82M`). The individual `Q4_0` + `qwen3` commands and tables in this section are preserved for historical comparison only.

```bash
# everything at once (current production stack)
./scripts/start_all.sh

# historical Plan 1 individual commands (superseded)
~/miniforge3/envs/llama/bin/llama-server -hf ggml-org/gemma-4-E4B-it-GGUF:Q4_0 \
  -np 2 -c 32768 -fa on --swa-full --jinja --port 8080 --host 127.0.0.1

UV_PROJECT_ENVIRONMENT=~/.venvs/brain uv run familiar-brain config/brain.toml

export LD_LIBRARY_PATH="$(find ~/.venvs/s2s/lib/python*/site-packages/nvidia -name lib -type d | tr '\n' ':')"
~/.venvs/s2s/bin/speech-to-speech serve --host 0.0.0.0 --port 8765 \
  --stt parakeet-tdt --tts qwen3 \
  --llm_backend responses-api --model_name familiar \
  --responses_api_base_url http://127.0.0.1:8090/v1 --responses_api_api_key "" \
  --responses_api_stream --enable_live_transcription --thresh 0.6 --chat_size 12
```

Note there is no `--mode realtime`; `serve` is the realtime server in 1.0.0. See
`docs/protocol-notes.md` section 7 for the full flag reconciliation.

## Measurements (Historical Plan 1 Baseline — Superseded by M5 Below)

VRAM, measured with `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
against the 12282 MiB total (historical `:Q4_0` + `Qwen3-TTS` baseline):

| State | VRAM used | of 12282 MiB |
|---|---|---|
| idle, nothing loaded | 731 MiB | 6 percent |
| Parakeet only | 2473 MiB | 20 percent |
| Gemma Q4_0 plus Parakeet | 8201 MiB | 67 percent |
| full stack, all three models resident | 11540 MiB | 94 percent |

Only about 740 MiB of headroom remains. Anything added later, the SmolVLM2 path in
particular, will not fit; Plan 3 should use the Gemma mmproj rather than a second
vision model, and the design document's documented fallback of the 0.6B TTS is the
lever to pull if more room is needed.

Turn latency, from `timing.py` lines in `data/brain.log`. Target 1.5 s from end of
speech to first audio, ceiling 2.5 s.

Before and after defaulting reasoning effort to none, same four turns, same machine:

| Path | before | after |
|---|---|---|
| IGNORE, gate only, no generation | 1943 ms | 1387 ms |
| ADDRESSED by name, first token | 9278 ms | 183 ms |
| ADDRESSED by name, total | 9780 ms | 620 ms |
| follow-up inside the window, total | 8557 ms | 505 ms |
| ADDRESSED with a dice roll, total | 2727 ms | 218 to 1216 ms |

The 1.5 s target is met on every addressed path, including turns that execute a tool
and therefore run two generations. The IGNORE path is dominated by the layer 3
classifier, which is the one place still worth tuning: it is a network call the gate
makes before deciding to stay silent.

Gate classifier, one token on slot two:

| Variant | latency |
|---|---|
| as shipped, reasoning on | 286 ms |
| reasoning_effort none (now the default) | 59 ms |

The configured `classify_timeout_s` is 0.6, which was being exceeded whenever the
slot was cold or contended, silently forcing every decision to IGNORE. At 59 ms the
margin is now roughly tenfold.

## Contract verification

`scripts/sdk_contract_probe.py`, run against a live brain with the same openai SDK
speech-to-speech uses, passes all ten checks: the non-streaming warmup body parses as
a `Response`, and an unaddressed line yields exactly one `idle_do_nothing` tool call
with a `call_`-prefixed id and zero spoken text.

Stronger still, the real pipeline started: speech-to-speech reached `Application
startup complete` and served `ws://0.0.0.0:8765/v1/realtime`, which it can only do
after its non-streaming warmup probe gets a valid `Response` back from the brain.

Dice, four consecutive live turns, from the brain's INFO log beside what was spoken:

| tool result | spoken |
|---|---|
| `dice [9, 16], kept [16], total 16` | "The result of your roll with advantage is 16." |
| `dice [2, 11], kept [11], total 11` | "The result is eleven." |
| `dice [2, 1], kept [2], total 2` | "The result is 2." |
| `dice [15, 3], kept [15], total 15` | "The result of your roll is 15." |

Every spoken number matches the tool's total, advantage keeps the higher of two dice,
and no brain-side tool call reached the client.

## What is left, and who has to do it

Three things could not be done from this session.

**1. The firewall rule (needs administrator).** Command in the firewall section above.
Until it exists the robot cannot reach port 8765 no matter what else is correct.

**2. Point the conversation app at the brain.** The robot has the app installed but
nothing running (`GET /api/apps/current-app-status` returns null, and the app lock is
free). Start it from the robot dashboard at `http://10.0.0.100:8000`, install the
`robot_profile/familiar` profile per the app's README, and set the connection to
`ws://<host-lan-ip>:8765/v1/realtime`, reading the address from `check_env.sh` first.
The connection URL and profile live in the app's own interface, not in the daemon REST
API, so this is a manual step.

**3. Say something to it.** Everything up to the microphone is verified, but nobody in
this session could speak in the room. The audio path, Parakeet transcription, the VAD
threshold and Qwen3-TTS playback are the parts still unexercised end to end.

A first pass at the table, roughly the plan's Task 13 Step 3:

```bash
./scripts/start_all.sh
curl -X POST http://127.0.0.1:8090/api/session/start
```

- Say a sentence that does not name the familiar. Expect silence, and an `IGNORE` line
  in `data/brain.log`.
- Say "Understudy, roll me a d20." Expect a spoken total matching the `tool roll`
  line in the log.
- Ask a follow-up within eight seconds. Expect a reply without naming it.
- Keep talking past three follow-ups. Expect it to fall quiet again, which is the
  `max_follow_ups` cap from deviation E6.

```bash
curl -X POST http://127.0.0.1:8090/api/session/end
cat data/sessions/*.jsonl
```

Then tune `--thresh` (VAD) and `[gate] follow_up_window_s` and `max_follow_ups` against
what the table actually sounds like. That is the Plan 4 work.

---

## Plan 2: knowledge layer

### Embedding model

`BAAI/bge-small-en-v1.5`, pinned in `familiar_brain/embeddings.py` as
`BgeEmbedder.MODEL`. Revision actually downloaded and used here:
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. 384 dimensions, CPU only, about 130 MB.
Loaded lazily, so nothing downloads until something is indexed or retrieved. Verified
by `tests/test_embeddings.py::test_bge_embedder_loads_on_cpu_and_has_384_dims`, which is
marked `slow` and excluded from the default run.

### sqlite-vec

`sqlite-vec` 0.1.9. Both KNN forms work on this version, `WHERE embedding MATCH ? AND
k = ?` and `ORDER BY distance LIMIT ?`; the first is used. `corpus` is a `vec0` metadata
column so the KNN search is scoped rather than filtered afterwards, see deviation P6.

### Notes index, first build

`uv run familiar-brain index notes` against the ten curated files in `data/notes/`:

```
index: {'indexed': 10, 'skipped': 0}
chunks: homebrew 11, notes 69
```

Retrieval spot-checked with the real embedder, top hit for each query:

| query | corpus | top hit |
|---|---|---|
| what happens on a critical hit | homebrew | house-rules.md, Injuries, 0.655 |
| what are Sanity skills | homebrew | house-rules.md, Custom skill system, 0.650 |
| what is Styx water | notes | 01-world-cosmology.md, Styx water and memory, 0.634 |
| who is Thraka | notes | 06-roster-and-npcs.md, The Graywater party, 0.558 |
| tell me about the Graywater Crossing | notes | 03-nephria-city.md, The Graywater Crossing, 0.686 |

Every one lands on the right file and the right heading, and the homebrew queries stay
inside the homebrew corpus. The rules corpus is empty until the SRD PDFs are downloaded.

### Environment: SSL_CERT_FILE is broken on this machine

`SSL_CERT_FILE` points at `C:\ProgramData\miniconda3/ssl/cacert.pem`, which does not
exist, so `uv` and `huggingface_hub` trust no certificate authority at all. Every fetch
fails with `invalid peer certificate: UnknownIssuer`, and `uv sync` piped to another
command still reports exit 0. Work around it per shell:

```bash
export SSL_CERT_FILE="$(.venv/Scripts/python.exe -c 'import certifi;print(certifi.where())')"
```

Nothing in the repo depends on this; it is a stale miniconda leftover in the user
environment. Worth removing from the environment permanently.

### Live loop 1

Not yet run. See `docs/live-loop.md`. Paste the `gate-report` output, the final
`--thresh` and `follow_up_window_s` here when it has been.

### Summariser check

Not yet run against a real transcript. Needs `docs/live-loop.md` to have produced
`tests/fixtures/live_session_1.jsonl` and llama-server to be up.

### Embedder load cost

Measured on the development machine, Windows, CPU:

| | first embed (cold) | subsequent embed |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 11863 ms | 21 ms |

The cold load is now paid during `create_app` by `indexing.warm_store`, not on the first
thing said at the table. See deviation P11 for why that mattered more than it looks: the
cold load was longer than the whole follow-up window. Expect `familiar-brain serve` to
take about twelve seconds longer to answer `/api/state` than it did in Plan 1.

---

## M5 Production Stack Upgrade & Benchmarks (2026-11-23)

### WSL2 `/mnt/c` Automount & Windows Launcher

- `/etc/wsl.conf` previously had `[automount] enabled = false`, leaving `/mnt/c` empty. Updated to `[automount] enabled = true` and applied via `wsl --shutdown` so `C:\` automounts cleanly at `/mnt/c` on every WSL2 start.
- Windows one-click launch: `launch_rocky.bat` (at repo root) -> `scripts/launch.ps1`.
  - Supports `-DryRun`, `-SkipRobot`, `-AutoSession`, `-NoAudioTap`, `-RobotHost`.
  - Cleans stale `SSL_CERT_FILE` in WSL2 (`unset SSL_CERT_FILE` + `/etc/ssl/certs/ca-certificates.crt`).
  - Checks Hyper-V firewall rules for **both** TCP `8765` (`speech-to-speech` / `AudioTapProxy`) and TCP `8090` (`familiar-brain` Companion UI). Missing rule remediation (Administrator PowerShell):

```powershell
New-NetFirewallHyperVRule -Name "WSL-s2s-8765" -DisplayName "speech-to-speech 8765 (WSL)" `
  -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol TCP -LocalPorts 8765 -Action Allow
New-NetFirewallHyperVRule -Name "WSL-brain-8090" -DisplayName "familiar-brain 8090 (WSL)" `
  -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol TCP -LocalPorts 8090 -Action Allow
```

### Gemma 4 E4B-it QAT + KV Cache Quantization

Upgraded `llama-server` in `scripts/start_all.sh` to:

```bash
"$LLAMA_BIN" -hf "unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL" --mmproj-auto \
  -np 2 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --swa-full \
  --jinja --reasoning off --reasoning-effort none \
  --port 8080 --host 127.0.0.1
```

Verified on RTX 4080 Laptop (12,282 MiB, baseline desktop `2,297 MiB` used):
- `llama-server` resident VRAM (model + `mmproj` + `q8_0` KV cache, 2 slots x 16k ctx, `--swa-full`): **`7,349–7,356 MiB`** total used (`~5,052 MiB` model delta, saving ~676 MiB vs PTQ `Q4_0` + `f16` KV cache).
- Gate classifier warm latency: **`57.8 ms`** (`reasoning_content: null`).
- `--jinja` tool calling (`roll`): **`589.9 ms`**, valid `tool_calls` JSON (`{"expr":"1d20"}`).
- Vision `mmproj`: **`1,764.2 ms`** cold PNG encode, valid visual answer (`"Red square."`).

### TTS Benchmark on RTX 4080 (Full Stack: Gemma 4 QAT + Parakeet-TDT + TTS)

Kokoro user-space install in `$HOME/.venvs/s2s` without `sudo`:

```bash
unset SSL_CERT_FILE
VIRTUAL_ENV="$HOME/.venvs/s2s" uv pip install "kokoro>=0.9.4" soundfile espeakng-loader phonemizer-fork
VIRTUAL_ENV="$HOME/.venvs/s2s" uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

Measured with `nvidia-smi` (total `12,282 MiB`, active Windows desktop baseline `1,814–2,306 MiB` used):

| TTS Candidate | Steady / Active VRAM Used | Steady Free VRAM Headroom | Synthesis Speed / RTF | Notes |
|---|---|---|---|---|
| `Qwen3-TTS 0.6B BF16` (`ggml`) | `11,792 MiB` | `204 MiB` | ~4.3x real-time | Fails `>= 2.5 GiB` headroom |
| `Qwen3-TTS 0.6B Q8_0` (`ggml`) | `10,933–11,191 MiB` | `805–1,063 MiB` | RTF `~0.23` (~4.3x real-time) | Fails `>= 2.5 GiB` headroom |
| **`Kokoro-82M`** (`hexgrad/Kokoro-82M`, `bm_fable`, CUDA) | **`8,841–9,187 MiB`** (transient synthesis peak `9,447 MiB`) | **`2809–3155 MiB free`** (`2.74–3.08 GiB free`; transient synthesis dip `2549 MiB free`) | **RTF `~0.058`** (~17x real-time; 265–337 ms for 4.6–5.9s audio) | **Selected**: >= 2.5 GiB steady/active headroom (`2809–3155 MiB free`) + ~4x faster TTS |

- `robot_profile/familiar/profile.md` updated to `voice = "bm_fable"`, `scripts/robot_connect.py` defaults `--voice bm_fable`, and `scripts/run_s2s.py` patches `KokoroTTSHandler` (`patch_kokoro_voice_guard`) to ignore invalid non-Kokoro Realtime session voice overrides such as `"Aiden"`.

### Full-Stack End-to-End Turn Latency (`familiar-brain` + Gemma 4 QAT + `data/familiar.db`)

Targets: addressed `median <= 1,500 ms`, `p95 <= 2,500 ms`; unaddressed turns do not stall.

| Workload | `retrieval_ms` | `total_ms` (`median` / `p95`) | Status |
|---|---|---|---|
| Unaddressed table turns (`n=10`, post-follow-up window, Wave 1 Challenge) | `0 ms` | **`3.2 ms`** median / `7.1 ms` max | **PASS** (`0` stalls, `0` speech) |
| Addressed `roll` turns (`n=7`, pre-M2-retrieval-fix baseline w/ 4x `sqlite-vec` scans over `3,183+` SRD 5.2.1 chunks) | `924–976 ms` warm (`1,539 ms` cold) | `2,087.8 ms` median / `4,449.8 ms` p95 | Superseded by M2 retrieval fix (below) |
| **Addressed `roll` turns (post-M2-retrieval-fix + post-s2s `/api/warmup`, Gemma 4 QAT + Kokoro-82M + `audio-tap`, `n=8`)** | Warmed up ahead of Turn 1 | **`1,338.2 ms`** median / **`2,389.6 ms`** p95 (Turn 1: `1,902.2 ms`, first-token median: `1,011.8 ms`; `3,936–4,039 MiB` free VRAM) | **PASS** (`median <= 1,500 ms`, `p95 <= 2,500 ms`) |

### STT Runaway Turn Fix & Stress Test

- `scripts/run_s2s.py` wraps `speech-to-speech` to:
  1. Cap `VADHandler._should_reopen_current_turn` at `S2S_MAX_TURN_AUDIO_S=30.0` and `S2S_MAX_TURN_REVISIONS=8`.
  2. Commit `speculative_turns` on tool-only `EndOfResponse` in `LMOutputProcessor.process` so `IGNORE` (`idle_do_nothing`) turns do not remain uncommitted.
  3. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `PHONEMIZER_ESPEAK_LIBRARY` / `ESPEAK_DATA_PATH` from `espeakng_loader`.
- `scripts/start_all.sh` passes `--max_speech_ms 25000 --smart_turn_max_wait_ms 1200 --unanswered_reopen_ms 1500`.
- **193.1s continuous speech stress test** (`38` x 5.25s 24 kHz PCM16 speech+pause cycles streamed to `ws://127.0.0.1:8765/v1/realtime` against live `Parakeet-TDT` + `Kokoro-82M` + `familiar-brain` + `Gemma 4 QAT`):
  - `38` distinct turns (`turn_1`..`turn_38`), `max_rev = 1`, **`0` CUDA OOM errors**.
  - VRAM stayed flat throughout: `9,169 MiB` (0s) -> `9,181 MiB` (30s) -> `9,231 MiB` (60s) -> `9,180 MiB` (90s) -> `9,183 MiB` (120s) -> `9,182 MiB` (150s) -> `9,183 MiB` (180s) -> `9,187 MiB` (193.1s, `2809 MiB free`).
  - Clean shutdown verified (`LEFTOVER_PROCS: none`).

