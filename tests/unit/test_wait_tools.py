from agent.tools import wait as wait_tools


def test_sleep_waits_for_requested_seconds(monkeypatch):
    captured: dict[str, float] = {}

    def fake_sleep(seconds: float) -> None:
        captured["seconds"] = seconds

    monkeypatch.setattr(wait_tools.time, "sleep", fake_sleep)

    result = wait_tools.sleep.invoke({"seconds": 2.5, "reason": "wait for teammate"})

    assert captured["seconds"] == 2.5
    assert result == "Slept for 2.5 seconds. reason=wait for teammate"


def test_sleep_rejects_invalid_seconds() -> None:
    result = wait_tools.sleep.invoke({"seconds": 0})

    assert result == "Error: seconds must be > 0 and <= 300"
