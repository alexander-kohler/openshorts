"""A Gemini id a new API key cannot call is retried once on the current model.

The failure this covers: clip selection called gemini-2.5-flash, Google answered
404 "no longer available to new users", and the job ended with no metadata.
"""
import types

import pytest

gemini_worker = pytest.importorskip("gemini_worker")


class _Resp:
    text = "{}"
    parsed = None
    candidates = []
    usage_metadata = None


class _Models:
    def __init__(self, fail_ids):
        self.fail_ids = set(fail_ids)
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(model)
        if model in self.fail_ids:
            raise RuntimeError(
                "404 NOT_FOUND. {'error': {'code': 404, 'message': "
                f"'This model models/{model} is no longer available to new users.', "
                "'status': 'NOT_FOUND'}}")
        return _Resp()


def _client(models):
    return types.SimpleNamespace(models=models)


@pytest.fixture(autouse=True)
def _clean():
    gemini_worker.reset_model_fallback()
    yield
    gemini_worker.reset_model_fallback()


def test_retired_model_is_retried_on_the_current_default():
    models = _Models({"gemini-2.5-flash"})
    gemini_worker.generate_content(
        _client(models), model="gemini-2.5-flash", contents="prompt")
    assert models.calls == ["gemini-2.5-flash", gemini_worker.DEFAULT_GEMINI_MODEL]
    assert gemini_worker.served_model == gemini_worker.DEFAULT_GEMINI_MODEL
    # The next call skips the id that already 404'd.
    gemini_worker.generate_content(
        _client(models), model="gemini-2.5-flash", contents="prompt")
    assert models.calls[-1] == gemini_worker.DEFAULT_GEMINI_MODEL
    assert models.calls.count("gemini-2.5-flash") == 1


def test_a_real_request_error_is_not_retried():
    class _Boom:
        calls = 0

        def generate_content(self, **kwargs):
            _Boom.calls += 1
            raise ValueError("400 INVALID_ARGUMENT: bad request")

    with pytest.raises(ValueError):
        gemini_worker.generate_content(
            _client(_Boom()), model="gemini-3.5-flash-lite", contents="prompt")
    assert _Boom.calls == 1


def test_image_calls_do_not_fall_back_to_a_text_model():
    models = _Models({"gemini-3.1-flash-image"})
    with pytest.raises(RuntimeError):
        gemini_worker.generate_content(
            _client(models), model="gemini-3.1-flash-image", contents="prompt",
            fallback=None)
    assert models.calls == ["gemini-3.1-flash-image"]


def test_text_fallback_is_the_text_model_not_flash_lite():
    models = _Models({"gemini-3.7-flash"})
    gemini_worker.generate_content(
        _client(models), model="gemini-3.7-flash", contents="prompt",
        fallback=gemini_worker.DEFAULT_GEMINI_TEXT_MODEL)
    assert models.calls == ["gemini-3.7-flash", gemini_worker.DEFAULT_GEMINI_TEXT_MODEL]


def test_configured_model_uses_env_then_the_default(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert gemini_worker.configured_model() == gemini_worker.DEFAULT_GEMINI_MODEL
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    assert gemini_worker.configured_model() == "gemini-3.5-flash-lite"
    monkeypatch.setenv("GEMINI_MODEL_SAAS", "gemini-3.8-flash")
    assert gemini_worker.configured_model(
        "GEMINI_MODEL_SAAS", "GEMINI_MODEL") == "gemini-3.8-flash"
