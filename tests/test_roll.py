import logging
import random
import pytest
from hypothesis import given, strategies as st
from familiar_brain import tools
from familiar_brain.tools import roll as roll_tool
from familiar_brain.tools.roll import roll, RollError


def test_registry_logs_every_tool_result(caplog):
    """Without this there is no record of what the familiar actually rolled, so a
    disputed result at the table cannot be audited afterwards."""
    tools.register(roll_tool)
    with caplog.at_level(logging.INFO, logger="familiar.tools"):
        out = tools.run("roll", {"expression": "1d20"})
    assert "Rolled 1d20" in out
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "roll" in logged and out.split(".")[0] in logged


def test_registry_logs_unknown_tool(caplog):
    with caplog.at_level(logging.WARNING, logger="familiar.tools"):
        assert "Unknown tool" in tools.run("nope", {})
    assert "nope" in " ".join(r.getMessage() for r in caplog.records)


def rng(seed): return random.Random(seed)


def test_simple_d20_plus_mod():
    r = roll("1d20+5", rng=rng(1))
    assert r["total"] == sum(r["dice"]) + 5 and len(r["dice"]) == 1 and 1 <= r["dice"][0] <= 20


def test_keep_highest_and_drop_lowest():
    r = roll("4d6dl1", rng=rng(2))
    assert len(r["dice"]) == 4 and len(r["kept"]) == 3 and r["total"] == sum(r["kept"])
    r2 = roll("2d20kh1", rng=rng(3))
    assert r2["total"] == max(r2["dice"])


def test_advantage_and_disadvantage_on_d20():
    a = roll("1d20+2", advantage="adv", rng=rng(4))
    assert len(a["dice"]) == 2 and a["total"] == max(a["dice"]) + 2
    d = roll("1d20-1", advantage="dis", rng=rng(5))
    assert d["total"] == min(d["dice"]) - 1


def test_natural_flags():
    r = roll("1d20", rng=rng(6))
    assert r["natural"] in (None, 1, 20)


@pytest.mark.parametrize("bad", ["", "d", "1d", "0d6", "1d0", "1d6+", "101d6", "1d1001", "1d6; drop table", "2d6kh3"])
def test_malformed_raises(bad):
    with pytest.raises(RollError):
        roll(bad, rng=rng(0))


@given(n=st.integers(1, 20), s=st.sampled_from([4, 6, 8, 10, 12, 20, 100]), m=st.integers(-20, 20))
def test_total_within_bounds(n, s, m):
    r = roll(f"{n}d{s}{m:+d}", rng=rng(n * s + m))
    assert n + m <= r["total"] <= n * s + m
