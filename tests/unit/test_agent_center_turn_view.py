from ui.agent_center_turn_view import build_status_event_line


def test_build_status_event_line_formats_trace_row():
    label, line = build_status_event_line(
        event_index=2,
        item={
            "sender": "planner",
            "receiver": "reader",
            "performative": "delegate",
            "content": "analyze paper contributions",
        },
    )
    assert label.startswith("执行中... [阶段:")
    assert line.startswith("2.")
    assert "planner" in line
    assert "reader" in line
    assert "delegate" in line
