#!/usr/bin/env bash
# Starts llama-server, familiar-brain, then speech-to-speech, waiting for each health check.
# Run inside WSL2. Paths match the no-sudo install recorded in docs/versions.md.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

LLAMA_BIN="${LLAMA_BIN:-$HOME/miniforge3/envs/llama/bin/llama-server}"
S2S_VENV="${S2S_VENV:-$HOME/.venvs/s2s}"
# The Windows-side .venv holds Windows binaries, so running in WSL needs its own
# environment. Keeping it outside the repo also avoids clobbering that .venv.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/brain}"
export PATH="$HOME/.local/bin:$PATH"

MODEL="${MODEL:-ggml-org/gemma-4-E4B-it-GGUF}"
LAN_IP=$(ip -4 addr show 2>/dev/null | awk '/inet /{print $2}' | grep -vE '^127\.|^10\.255\.' | cut -d/ -f1 | head -1)
mkdir -p data

wait_for() {
  local url=$1 name=$2
  for _ in $(seq 1 600); do
    if curl -sf "$url" >/dev/null 2>&1; then echo "$name ready"; return 0; fi
    sleep 1
  done
  echo "$name did not come up"; exit 1
}

cleanup() { echo "stopping children"; kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# --jinja is load-bearing: without it llama-server ignores the tools array entirely and
# returns tool_calls: null, so roll, initiative and note can never fire. The :Q4_0
# suffix is equally deliberate, see docs/versions.md.
# Gemma 4 E4B ships mmproj-gemma-4-E4B-it-{BF16,Q8_0}.gguf in the same repo. The
# projector is what lets the camera tool's frame reach the model (Plan 3, sight).
# Set MMPROJ_ARGS=--no-mmproj to measure VRAM without it.
"$LLAMA_BIN" -hf "$MODEL" ${MMPROJ_ARGS:---mmproj-auto} -np 2 -c 32768 -fa on --swa-full --jinja --port 8080 --host 127.0.0.1 \
  > data/llama.log 2>&1 &
# First run downloads several GB, so allow a long wait here.
wait_for http://127.0.0.1:8080/health llama-server

uv run --project "$REPO" familiar-brain serve config/brain.toml > data/brain.log 2>&1 &
wait_for http://127.0.0.1:8090/api/state familiar-brain

# The ggml TTS backend (qwentts-cpp) is built against CUDA 12 and dlopens
# libcudart.so.12 and libcublas.so.12, but torch here resolves to cu13, so only the
# CUDA 13 sonames are present. The cu12 shim wheels supply the missing ones; without
# this path the pipeline dies at TTS setup. See docs/versions.md.
export LD_LIBRARY_PATH="$(find "$S2S_VENV"/lib/python*/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

# No --mode flag in speech-to-speech 1.0.0: the realtime server is the `serve`
# subcommand. See docs/protocol-notes.md section 7.
"$S2S_VENV/bin/speech-to-speech" serve \
  --host 0.0.0.0 --port 8765 \
  --stt parakeet-tdt --tts qwen3 \
  --llm_backend responses-api --model_name familiar \
  --responses_api_base_url http://127.0.0.1:8090/v1 --responses_api_api_key "" \
  --responses_api_stream \
  --enable_live_transcription \
  --thresh 0.6 --chat_size 12 \
  > data/s2s.log 2>&1 &

echo "Point the robot at ws://${LAN_IP}:8765/v1/realtime"
echo "Start a session: curl -X POST http://127.0.0.1:8090/api/session/start"
wait
