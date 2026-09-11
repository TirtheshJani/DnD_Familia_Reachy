#!/usr/bin/env bash
# Verifies the WSL2 host can run the full local stack. Prints a report; exits non-zero
# on hard failures. Run from anywhere: ./scripts/check_env.sh
set -u
fail=0

LLAMA_BIN="${LLAMA_BIN:-$HOME/miniforge3/envs/llama/bin/llama-server}"
S2S_VENV="${S2S_VENV:-$HOME/.venvs/s2s}"
ROBOT_HOST="${ROBOT_HOST:-reachy-mini.local}"

echo "== WSL2 =="
if grep -qi microsoft /proc/version; then echo "  WSL detected"; else echo "  not WSL (fine if native Linux)"; fi

echo "== networking mode =="
# The robot connects to speech-to-speech over the LAN. Under WSL2's default NAT the
# distro sits on a private 192.168.x address that the robot cannot route to, so a
# service bound here is unreachable no matter what it binds. Mirrored mode gives the
# distro the host's own LAN address. See docs/versions.md.
lan_ip=$(ip -4 addr show 2>/dev/null | awk '/inet /{print $2}' | grep -vE '^127\.|^10\.255\.' | cut -d/ -f1 | head -1)
echo "  addresses: $(ip -4 -o addr show 2>/dev/null | awk '{print $2"="$4}' | tr '\n' ' ')"
if [ -n "$lan_ip" ] && echo "$lan_ip" | grep -qE '^(10|172|192)\.'; then
  if echo "$lan_ip" | grep -qE '^192\.168\.(1[6-9]|[2-9][0-9])\.'; then
    echo "  WARNING: $lan_ip looks like a WSL NAT address, not a LAN address."
    echo "  The robot will not reach this host. Set networkingMode=mirrored in %USERPROFILE%\\.wslconfig"
    echo "  then run 'wsl --shutdown'."
    fail=1
  else
    echo "  LAN IP (give this to the robot): $lan_ip"
  fi
else
  echo "  could not determine a LAN address"; fail=1
fi

echo "== memory =="
total_mb=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
echo "  MemTotal: ${total_mb} MiB"
if [ "$total_mb" -lt 9000 ]; then
  echo "  WARNING: under 9 GiB. Parakeet plus Qwen3-TTS plus llama-server will not fit."
  echo "  Raise memory= in %USERPROFILE%\\.wslconfig and run 'wsl --shutdown'."
  fail=1
fi

echo "== CUDA =="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
  echo "  nvidia-smi missing"; fail=1
fi

echo "== llama-server =="
if [ -x "$LLAMA_BIN" ]; then
  "$LLAMA_BIN" --version 2>&1 | head -2 | sed 's/^/  /'
  if [ -e "$(dirname "$LLAMA_BIN")/../lib/libggml-cuda.so" ] || ls "$(dirname "$LLAMA_BIN")"/../lib/libggml-cuda* >/dev/null 2>&1; then
    echo "  CUDA backend present"
  else
    echo "  WARNING: no libggml-cuda found; this may be a CPU-only build"
  fi
elif command -v llama-server >/dev/null; then
  llama-server --version 2>&1 | head -2 | sed 's/^/  /'
else
  echo "  llama-server missing at $LLAMA_BIN (see docs/versions.md for the install)"; fail=1
fi

echo "== speech-to-speech =="
if [ -x "$S2S_VENV/bin/speech-to-speech" ]; then
  "$S2S_VENV/bin/python" -c "
import importlib.metadata as m, torch
print('  speech-to-speech', m.version('speech-to-speech'))
print('  torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
" 2>&1 | grep -E '^\s+(speech|torch)' || { echo "  import failed"; fail=1; }
else
  echo "  speech-to-speech missing at $S2S_VENV (see docs/versions.md for the install)"; fail=1
fi

echo "== python =="
echo "  $(python3 --version)"

echo "== robot =="
if curl -sf --max-time 5 "http://${ROBOT_HOST}:8000/api/daemon/status" >/dev/null 2>&1; then
  curl -s --max-time 5 "http://${ROBOT_HOST}:8000/api/daemon/status" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('  name=%s version=%s wlan_ip=%s state=%s' % (d.get('robot_name'), d.get('version'), d.get('wlan_ip'), d.get('state')))" 2>/dev/null \
    || echo "  reachable but status unparseable"
else
  echo "  robot not reachable at ${ROBOT_HOST}:8000 (not fatal for unit tests)"
fi

echo
if [ "$fail" -eq 0 ]; then echo "RESULT: ok"; else echo "RESULT: problems found above"; fi
exit $fail
