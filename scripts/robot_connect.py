"""Point the robot's conversation app at this laptop and select the familiar profile.

The robot runs reachy_mini_conversation_app 0.10.0 (checked 2026-09-16), not the 1.0.1
that robot_profile/familiar/profile.md is written for. 0.10.0 stores a profile as
instructions.txt and tools.txt, and its settings server on port 7860 can write both, so
nothing here needs SSH:

  1. start the app through the daemon if it is not already running
  2. POST /api/v1/backend_config   local mode, ws://<laptop LAN IP>:8765/v1/realtime
  3. POST /api/v1/personalities/save   the familiar profile with only the robot tools
  4. POST /api/v1/personalities/apply  select it and keep it for the next start

Run after scripts/start_all.sh prints "Point the robot at", from Windows or WSL:

    uv run python scripts/robot_connect.py            # host IP detected
    uv run python scripts/robot_connect.py --host-ip 10.0.0.95

The laptop's address comes from DHCP and moves (docs/versions.md), so run this again
whenever it changes. The robot can only reach port 8765 once the Hyper-V firewall rule
from docs/versions.md exists.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import requests

APP = "reachy_mini_conversation_app"
PROFILE = "familiar"
PROFILE_DIR = Path(__file__).resolve().parent.parent / "robot_profile" / PROFILE


def detect_host_ip(robot: str) -> str:
    """The local address the OS would use to reach the robot. No packet is sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((robot, 8000))
        return s.getsockname()[0]


def read_tools(path: Path) -> str:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return "\n".join(ln for ln in lines if ln and not ln.startswith("#"))


def wait_for(url: str, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    sys.exit(f"{url} did not answer within {seconds:.0f} s")


def post(url: str, body: dict) -> dict:
    r = requests.post(url, json=body, timeout=20)
    try:
        payload = r.json()
    except ValueError:
        payload = {"text": r.text}
    if not r.ok or payload.get("ok") is False:
        sys.exit(f"POST {url} failed ({r.status_code}): {json.dumps(payload)}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", default="10.0.0.100")
    ap.add_argument("--host-ip", default=None, help="laptop LAN address the robot connects to")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--voice", default="Aiden")
    args = ap.parse_args()

    host_ip = args.host_ip or detect_host_ip(args.robot)
    daemon = f"http://{args.robot}:8000/api"
    settings = f"http://{args.robot}:7860/api/v1"

    status = requests.get(f"{daemon}/apps/current-app-status", timeout=5).json()
    if not status or (status.get("info") or {}).get("name") != APP:
        print(f"starting {APP} on {args.robot}")
        post(f"{daemon}/apps/start-app/{APP}", {})
    wait_for(f"{settings}/status", 180)

    print(f"backend: ws://{host_ip}:{args.port}/v1/realtime")
    print(" ", post(f"{settings}/backend_config",
                    {"hf_mode": "local", "hf_host": host_ip, "hf_port": args.port}).get("message"))

    voices = requests.get(f"{settings}/voices", timeout=10).json()
    names = voices if isinstance(voices, list) else voices.get("voices", [])
    voice = args.voice if args.voice in names else None
    if voice is None:
        print(f"  voice {args.voice!r} not offered ({names}); keeping the app default")

    saved = post(f"{settings}/personalities/save", {
        "name": PROFILE,
        "instructions": (PROFILE_DIR / "instructions.md").read_text(encoding="utf-8"),
        "tools_text": read_tools(PROFILE_DIR / "tools.txt"),
        **({"voice": voice} if voice else {}),
    })
    print(f"profile saved as {saved['value']}")
    applied = post(f"{settings}/personalities/apply", {"name": saved["value"], "persist": True})
    print(f"  {applied.get('status')} (startup: {applied.get('startup')})")
    print(json.dumps(requests.get(f"{settings}/status", timeout=10).json(), indent=2))


if __name__ == "__main__":
    main()
