from types import SimpleNamespace

from utils import task_queue


def test_enqueue_task_falls_back_when_no_workers(monkeypatch):
    called = {"ran": False}

    monkeypatch.setattr(task_queue, "task_queue", object())
    monkeypatch.setattr(task_queue, "_has_active_rq_workers", lambda: False)

    def run_task():
        called["ran"] = True

    result = task_queue.enqueue_task(run_task)

    assert called["ran"]
    assert result == {"mode": "sync", "job_id": None}


def test_enqueue_task_uses_queue_when_workers_exist(monkeypatch):
    class _FakeQueue:
        def enqueue(self, *_args, **_kwargs):
            return SimpleNamespace(id="job-1")

    monkeypatch.setattr(task_queue, "task_queue", _FakeQueue())
    monkeypatch.setattr(task_queue, "_has_active_rq_workers", lambda: True)

    result = task_queue.enqueue_task(lambda: None)

    assert result == {"mode": "queued", "job_id": "job-1"}
