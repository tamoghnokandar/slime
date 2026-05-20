from types import SimpleNamespace

from slime.backends.sglang_utils import sglang_engine


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_external_engine_startup_defaults_to_lightweight_health(monkeypatch):
    captured = {}

    def fake_wait_server_healthy(*, base_url, api_key, is_process_alive, endpoint):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["is_process_alive"] = is_process_alive
        captured["endpoint"] = endpoint

    monkeypatch.delenv("SLIME_SGLANG_STARTUP_HEALTH_ENDPOINT", raising=False)
    monkeypatch.setattr(sglang_engine, "_wait_server_healthy", fake_wait_server_healthy)
    monkeypatch.setattr(sglang_engine.requests, "get", lambda _url: FakeResponse(payload={}))
    monkeypatch.setattr(sglang_engine.SGLangEngine, "_register_router", lambda _self, _server_args: None)

    engine = sglang_engine.SGLangEngine.__new__(sglang_engine.SGLangEngine)
    engine.rank = 0
    engine.worker_type = "regular"
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000

    engine._init_external(expect_server_args={}, external_engine_need_check_fields=[])

    assert captured["base_url"] == "http://127.0.0.1:30000"
    assert captured["api_key"] is None
    assert captured["is_process_alive"]()
    assert captured["endpoint"] == "/health"


def test_wait_server_healthy_uses_timeout_and_fast_polling(monkeypatch):
    sleeps = []
    calls = []
    responses = [FakeResponse(status_code=503), FakeResponse(status_code=200)]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url, headers, timeout):
            calls.append(SimpleNamespace(url=url, headers=headers, timeout=timeout))
            return responses.pop(0)

    monkeypatch.delenv("SLIME_SGLANG_HEALTH_REQUEST_TIMEOUT", raising=False)
    monkeypatch.setattr(sglang_engine.requests, "Session", lambda: FakeSession())
    monkeypatch.setattr(sglang_engine.time, "sleep", sleeps.append)

    sglang_engine._wait_server_healthy(
        base_url="http://127.0.0.1:30000",
        api_key="test-key",
        is_process_alive=lambda: True,
        endpoint="/health",
    )

    assert [call.url for call in calls] == ["http://127.0.0.1:30000/health"] * 2
    assert all(call.timeout == 2 for call in calls)
    assert calls[0].headers["Authorization"] == "Bearer test-key"
    assert sleeps == [0.1]


def test_wait_server_healthy_uses_configured_request_timeout(monkeypatch):
    calls = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url, headers, timeout):
            calls.append(timeout)
            return FakeResponse(status_code=200)

    monkeypatch.setenv("SLIME_SGLANG_HEALTH_REQUEST_TIMEOUT", "0.25")
    monkeypatch.setattr(sglang_engine.requests, "Session", lambda: FakeSession())

    sglang_engine._wait_server_healthy(
        base_url="http://127.0.0.1:30000",
        api_key=None,
        is_process_alive=lambda: True,
        endpoint="/health",
    )

    assert calls == [0.25]
