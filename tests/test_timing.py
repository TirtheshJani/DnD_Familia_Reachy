from familiar_brain.timing import TurnTimer


def test_turn_timer_records_named_marks_in_ms():
    t = TurnTimer()
    t.mark("gate")
    t.mark("retrieval")
    rec = t.finish(decision="ADDRESSED", spoke=True)
    assert set(rec) >= {"gate_ms", "retrieval_ms", "total_ms", "decision", "spoke"}
    assert rec["total_ms"] >= rec["gate_ms"] >= 0
