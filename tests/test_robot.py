import math
from familiar_brain.robot import RobotDaemon, direction_from_angle


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status
    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError(f"http {self.status}")
    def json(self):
        return self.payload


def test_direction_from_angle_uses_the_daemon_convention():
    """0 = left, pi/2 = front, pi = right (reachy_mini io/protocol.py DoaSnapshot)."""
    assert direction_from_angle(0.2) == "left"
    assert direction_from_angle(math.pi / 2) == "front"
    assert direction_from_angle(math.pi - 0.2) == "right"
    assert direction_from_angle(math.pi / 2 + 0.3) == "front"     # inside the dead zone


def test_doa_reads_angle_and_speech_flag():
    seen = []
    def get(url, timeout):
        seen.append((url, timeout)); return _Resp({"angle": 0.3, "speech_detected": True})
    d = RobotDaemon("http://robot:8000/", timeout_s=0.15, get=get)
    assert d.doa() == (0.3, True)
    assert seen == [("http://robot:8000/api/state/doa", 0.15)]


def test_doa_degrades_to_none_on_null_error_or_timeout():
    assert RobotDaemon("http://r", get=lambda u, timeout: _Resp(None)).doa() is None
    assert RobotDaemon("http://r", get=lambda u, timeout: _Resp({}, status=503)).doa() is None
    def boom(u, timeout): raise TimeoutError("slow")
    assert RobotDaemon("http://r", get=boom).doa() is None
    assert RobotDaemon("", get=boom).doa() is None       # unconfigured: never even calls


def test_doa_never_calls_the_network_when_unconfigured():
    calls = []
    def get(u, timeout):
        calls.append(u); return _Resp({"angle": 0.3, "speech_detected": True})
    assert RobotDaemon("", get=get).doa() is None
    assert RobotDaemon("/", get=get).doa() is None
    assert calls == []


def test_doa_degrades_to_none_on_a_malformed_angle():
    for payload in ({"angle": None}, {"angle": "left"}, {"angle": [1.0]}, [0.3, True],
                    {"angle": float("nan")}, {"angle": float("inf")}, {"angle": "nan"}):
        assert RobotDaemon("http://r", get=lambda u, timeout, p=payload: _Resp(p)).doa() is None
