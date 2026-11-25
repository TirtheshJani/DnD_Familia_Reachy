#!/usr/bin/env bash
# Starts llama-server, familiar-brain, speech-to-speech (via run_s2s.py), and optional
# AudioTapProxy in order, waiting for every readiness endpoint before connecting the robot.
# Run inside WSL2. Paths match the no-sudo install recorded in docs/versions.md.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

SKIP_ROBOT="${SKIP_ROBOT:-0}"
NATIVE_ROBOT="${NATIVE_ROBOT:-0}"
AUTO_SESSION="${AUTO_SESSION:-0}"
ENABLE_AUDIO_TAP="${ENABLE_AUDIO_TAP:-1}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-robot)   SKIP_ROBOT=1 ;;
    --native-robot) NATIVE_ROBOT=1 ;;
    --auto-session) AUTO_SESSION=1 ;;
    --no-audio-tap) ENABLE_AUDIO_TAP=0 ;;
    -h|--help)
      echo "Usage: $0 [--skip-robot] [--native-robot] [--auto-session] [--no-audio-tap]"
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 2
      ;;
  esac
  shift
done

# Clear stale Windows SSL_CERT_FILE inherited into WSL2 and ensure a valid Linux CA bundle.
if [ -n "${SSL_CERT_FILE:-}" ] && [ ! -f "$SSL_CERT_FILE" ]; then
  unset SSL_CERT_FILE
fi
if [ -z "${SSL_CERT_FILE:-}" ] && [ -f /etc/ssl/certs/ca-certificates.crt ]; then
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
fi

LLAMA_BIN="${LLAMA_BIN:-$HOME/miniforge3/envs/llama/bin/llama-server}"
S2S_VENV="${S2S_VENV:-$HOME/.venvs/s2s}"
# The Windows-side .venv holds Windows binaries, so running in WSL needs its own
# environment. Keeping it outside the repo also avoids clobbering that .venv.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/brain}"
export PATH="$PATH:$HOME/.local/bin"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "$NATIVE_ROBOT" = "1" ]; then
  export FAMILIAR_ROBOT_MODE=native
  unset FAMILIAR_NO_ROBOT || true
elif [ "$SKIP_ROBOT" = "1" ]; then
  export FAMILIAR_NO_ROBOT=1
fi

# Gemma 4 E4B-it QAT (Unsloth Dynamic UD-Q4_K_XL) + Q8_0 KV cache + reasoning disabled.
# Fits alongside Parakeet-TDT and Kokoro-82M with >= 2.5 GiB free VRAM on a 12 GB RTX 4080.
# See docs/versions.md for measured VRAM and latency benchmarks.
MODEL="${MODEL:-unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL}"
LAN_IP=$(ip -4 addr show 2>/dev/null | awk '/inet /{print $2}' | grep -vE '^127\.|^10\.255\.' | cut -d/ -f1 | head -1 || true)
mkdir -p data

PIDS=()
CLEANED_UP=0

cleanup() {
  if [ "$CLEANED_UP" = "1" ]; then
    return
  fi
  CLEANED_UP=1
  echo "stopping children"
  for pid in "${PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 0.3
  for pid in "${PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {
  local url=$1 name=$2 logfile=$3 pid=$4 default_wait=${5:-600}
  local max_wait=${STARTUP_TIMEOUT_S:-$default_wait}
  local deadline=$((SECONDS + max_wait))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited unexpectedly; tail of $logfile:" >&2
      tail -n 30 "$logfile" >&2 || true
      exit 1
    fi
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "$name ready"
      return 0
    fi
    sleep 0.2
  done
  echo "$name did not come up within ${max_wait}s; tail of $logfile:" >&2
  tail -n 30 "$logfile" >&2 || true
  exit 1
}

# Use cached HuggingFace models offline when already downloaded (cuts ~20+ network HEAD checks on boot
# and ensures rock-solid startup on venues with flaky Wi-Fi). Set HF_HUB_OFFLINE=0 to force online checks.
if [ -z "${HF_HUB_OFFLINE:-}" ] && [ -d "$HOME/.cache/huggingface/hub" ]; then
  if ls "$HOME/.cache/huggingface/hub"/models--BAAI--bge-small-en-v1.5 >/dev/null 2>&1 \
     && ls "$HOME/.cache/huggingface/hub"/models--nvidia--parakeet-tdt-0.6b-v3 >/dev/null 2>&1 \
     && ls "$HOME/.cache/huggingface/hub"/models--hexgrad--Kokoro-82M >/dev/null 2>&1; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  fi
fi

BRAIN_PY="python3"
if [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  BRAIN_PY="$UV_PROJECT_ENVIRONMENT/bin/python"
fi

AUDIO_TAP_ACTIVE=0
if [ "$ENABLE_AUDIO_TAP" = "1" ] && [ -f "$REPO/familiar_brain/audio_tap.py" ] && "$BRAIN_PY" -c "import familiar_brain.audio_tap" 2>/dev/null; then
  AUDIO_TAP_ACTIVE=1
  S2S_HOST="127.0.0.1"
  S2S_PORT="8766"
else
  if [ "$ENABLE_AUDIO_TAP" = "1" ]; then
    echo "audio-tap module not present; binding speech-to-speech directly on 0.0.0.0:8765"
  fi
  S2S_HOST="0.0.0.0"
  S2S_PORT="8765"
fi

# Launch llama-server, familiar-brain, and speech-to-speech IN PARALLEL so Companion UI opens
# in ~12-15s while Gemma 4 and Parakeet+Kokoro load concurrently (~55-65s faster cold boot).
"$LLAMA_BIN" -hf "$MODEL" ${MMPROJ_ARGS:---mmproj-auto} \
  -np 2 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --swa-full \
  --jinja --reasoning off --reasoning-effort none \
  --port 8080 --host 127.0.0.1 \
  > data/llama.log 2>&1 &
LLAMA_PID=$!
PIDS+=("$LLAMA_PID")

export FAMILIAR_WARMUP_LLM="${FAMILIAR_WARMUP_LLM:-1}"
if [ -x "$UV_PROJECT_ENVIRONMENT/bin/familiar-brain" ]; then
  "$UV_PROJECT_ENVIRONMENT/bin/familiar-brain" serve config/brain.toml > data/brain.log 2>&1 &
else
  uv run --project "$REPO" familiar-brain serve config/brain.toml > data/brain.log 2>&1 &
fi
BRAIN_PID=$!
PIDS+=("$BRAIN_PID")

# The cu12 shim directories supply libcudart.so.12 / libcublas.so.12 if needed alongside cu13 torch.
export LD_LIBRARY_PATH="$(find "$S2S_VENV"/lib/python*/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

S2S_ARGS=(
  serve
  --host "$S2S_HOST" --port "$S2S_PORT"
  --stt parakeet-tdt
  --tts kokoro --kokoro_model_name hexgrad/Kokoro-82M --kokoro_device cuda --kokoro_voice bm_fable --kokoro_lang_code b
  --llm_backend responses-api --model_name familiar
  --responses_api_base_url http://127.0.0.1:8090/v1 --responses_api_api_key ""
  --responses_api_stream
  --enable_live_transcription
  --thresh 0.6 --chat_size 12
  --max_speech_ms 25000 --smart_turn_max_wait_ms 1200 --unanswered_reopen_ms 1500
)

# Run through scripts/run_s2s.py to bound VAD speculative turns and guard Kokoro voice overrides.
if [ -x "$S2S_VENV/bin/python" ]; then
  "$S2S_VENV/bin/python" "$REPO/scripts/run_s2s.py" "${S2S_ARGS[@]}" > data/s2s.log 2>&1 &
else
  "$S2S_VENV/bin/speech-to-speech" "${S2S_ARGS[@]}" > data/s2s.log 2>&1 &
fi
S2S_PID=$!
PIDS+=("$S2S_PID")

# Wait for familiar-brain first so Wake Daemon can open the Companion UI right away (~15s).
wait_for http://127.0.0.1:8090/api/state familiar-brain data/brain.log "$BRAIN_PID" 120
wait_for http://127.0.0.1:8080/health llama-server data/llama.log "$LLAMA_PID" 600
wait_for "http://127.0.0.1:${S2S_PORT}/v1/pool" speech-to-speech data/s2s.log "$S2S_PID" 240

if [ "$AUDIO_TAP_ACTIVE" = "1" ]; then
  "$BRAIN_PY" -m familiar_brain.audio_tap \
    --listen-host 0.0.0.0 --listen-port 8765 \
    --upstream-host 127.0.0.1 --upstream-port 8766 \
    --control-host 127.0.0.1 --control-port 8767 \
    --config config/brain.toml \
    > data/audio_tap.log 2>&1 &
  TAP_PID=$!
  PIDS+=("$TAP_PID")
  wait_for http://127.0.0.1:8765/v1/pool audio-tap data/audio_tap.log "$TAP_PID" 60
fi

WARMUP_LLM_JSON=false
if [ "${FAMILIAR_WARMUP_LLM:-1}" = "1" ] || [ "${FAMILIAR_WARMUP_LLM:-1}" = "true" ]; then
  WARMUP_LLM_JSON=true
fi
curl -sf -X POST -H "Content-Type: application/json" -d "{\"llm\": ${WARMUP_LLM_JSON}}" http://127.0.0.1:8090/api/warmup >/dev/null || true

DISPLAY_IP="${LAN_IP:-127.0.0.1}"
echo "Companion UI (phone LAN): http://${DISPLAY_IP}:8090"
echo "Point the robot at ws://${DISPLAY_IP}:8765/v1/realtime"

if [ "$NATIVE_ROBOT" = "1" ]; then
  echo "Skipping robot connection (--native-robot: reachy_familiar manages link)"
elif [ "$SKIP_ROBOT" = "1" ]; then
  echo "Skipping robot connection (--skip-robot / SKIP_ROBOT=1)"
else
  ROBOT_ARGS=()
  if [ -n "${ROBOT_HOST:-}" ]; then
    ROBOT_ARGS+=(--robot "$ROBOT_HOST")
  fi
  if [ -n "${LAN_IP:-}" ]; then
    ROBOT_ARGS+=(--host-ip "$LAN_IP")
  fi
  "$BRAIN_PY" "$REPO/scripts/robot_connect.py" "${ROBOT_ARGS[@]}" --port 8765
fi

if [ "$AUTO_SESSION" = "1" ]; then
  curl -sf -X POST http://127.0.0.1:8090/api/session/start >/dev/null && echo "Session auto-started"
else
  echo "Start a session: curl -X POST http://127.0.0.1:8090/api/session/start"
fi

wait
