"""The BYOK key probe must use a model that exists: a retired model 404s for every valid key,
which _probe_anthropic_key reports as validation_failed."""

import anthropic

from kestrel_backend import main


def test_probe_uses_current_model_and_accepts_valid_key(monkeypatch):
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            return object()

    class FakeClient:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    assert main._probe_anthropic_key("sk-ant-test") == (True, None)
    assert seen["model"] == "claude-haiku-4-5"
    assert seen["max_tokens"] == 1
