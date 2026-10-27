import logging
from kestrel_backend.logging_config import ApiKeyRedactionFilter


def test_filter_masks_anthropic_key():
    f = ApiKeyRedactionFilter()
    rec = logging.LogRecord("x", logging.INFO, "f", 1,
                            "using key sk-ant-abc123DEF456ghi789", None, None)
    f.filter(rec)
    assert "sk-ant-abc123DEF456ghi789" not in rec.getMessage()
    assert "REDACTED" in rec.getMessage()


def test_filter_masks_key_in_args():
    f = ApiKeyRedactionFilter()
    rec = logging.LogRecord("x", logging.INFO, "f", 1,
                            "key=%s", None, None)
    rec.args = ("sk-ant-abc123DEF456ghi789",)
    f.filter(rec)
    assert "sk-ant-abc123DEF456ghi789" not in rec.getMessage()
    assert "REDACTED" in rec.getMessage()


def test_filter_masks_key_in_dict_args():
    f = ApiKeyRedactionFilter()
    # LogRecord.__init__ in Python 3.x unpacks a single-element tuple containing
    # a Mapping into record.args as the dict itself.  Simulate that here.
    rec = logging.LogRecord("x", logging.INFO, "f", 1,
                            "key=%(k)s", None, None)
    rec.args = {"k": "sk-ant-abc123DEF456ghi789"}
    f.filter(rec)
    assert "sk-ant-abc123DEF456ghi789" not in rec.getMessage()
    assert "REDACTED" in rec.getMessage()
