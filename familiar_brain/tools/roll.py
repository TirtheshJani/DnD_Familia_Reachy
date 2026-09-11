from __future__ import annotations

import random
import re

_PAT = re.compile(r"^(?P<n>\d+)d(?P<s>\d+)(?:(?P<mode>kh|dl)(?P<k>\d+))?(?P<mod>[+-]\d+)?$")


class RollError(ValueError):
    pass


def roll(expression: str, advantage: str = "none", rng: random.Random | None = None) -> dict:
    rng = rng or random.SystemRandom()
    m = _PAT.match(expression.replace(" ", "").lower())
    if not m:
        raise RollError(f"bad dice expression {expression!r}")
    n, sides = int(m["n"]), int(m["s"])
    if not (1 <= n <= 100) or not (2 <= sides <= 1000):
        raise RollError("dice count 1-100 and sides 2-1000 only")
    mod = int(m["mod"] or 0)
    if advantage not in ("none", "adv", "dis"):
        raise RollError("advantage must be none, adv, or dis")
    if advantage != "none" and (n != 1 or sides != 20):
        raise RollError("advantage applies to a single d20 only")
    if advantage != "none":
        n = 2
    dice = [rng.randint(1, sides) for _ in range(n)]
    kept = list(dice)
    if advantage == "adv":
        kept = [max(dice)]
    elif advantage == "dis":
        kept = [min(dice)]
    elif m["mode"]:
        k = int(m["k"])
        if not (1 <= k <= n):
            raise RollError("keep/drop count out of range")
        kept = sorted(dice, reverse=True)[:k] if m["mode"] == "kh" else sorted(dice)[k:]
    total = sum(kept) + mod
    natural = kept[0] if (sides == 20 and len(kept) == 1 and kept[0] in (1, 20)) else None
    return {"expression": expression, "dice": dice, "kept": kept, "modifier": mod, "total": total, "natural": natural}


SPEC = {
    "type": "function",
    "name": "roll",
    "description": "Roll dice. The tool does all arithmetic; report its total verbatim. Expressions like 1d20+5, 2d6, 4d6dl1, 2d20kh1.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "advantage": {"type": "string", "enum": ["none", "adv", "dis"]},
        },
        "required": ["expression"],
    },
}


def run(args: dict) -> str:
    r = roll(args.get("expression", ""), args.get("advantage", "none"))
    nat = " Natural " + str(r["natural"]) + "!" if r["natural"] else ""
    return f"Rolled {r['expression']}: dice {r['dice']}, kept {r['kept']}, modifier {r['modifier']:+d}, total {r['total']}.{nat}"
