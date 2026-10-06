from familiar_brain.state import BrainState


def test_cue_is_consumed_once():
    st = BrainState()
    assert st.consume_cue() is None
    st.set_cue("asleep")
    assert st.consume_cue() == "asleep" and st.consume_cue() is None


def test_fresh_state_is_not_tracking_and_in_familiar_mode():
    st = BrainState()
    assert st.tracking is False and st.mode == "familiar" and st.mode_set_at is None


def test_ending_the_conversation_closes_the_follow_up_window_and_disarms_speak_now():
    st = BrainState()
    st.mark_named(10.0)
    st.window_extension_s = 3.0
    st.count_follow_up()
    st.arm_speak_now()
    st.end_conversation()
    assert st.last_named_at is None and st.window_extension_s is None and st.follow_ups == 0
    assert st.consume_speak_now() is False
