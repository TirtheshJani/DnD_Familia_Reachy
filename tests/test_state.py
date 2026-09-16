from familiar_brain.state import BrainState


def test_cue_is_consumed_once():
    st = BrainState()
    assert st.consume_cue() is None
    st.set_cue("asleep")
    assert st.consume_cue() == "asleep" and st.consume_cue() is None


def test_fresh_state_is_not_tracking_and_in_familiar_mode():
    st = BrainState()
    assert st.tracking is False and st.mode == "familiar" and st.mode_set_at is None
    assert st.last_glance_at == float("-inf")
