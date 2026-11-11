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



def test_httpx_url_object_with_api_key_is_redacted_even_when_httpx_is_set_to_info(capsys):
    """Greptile #135: LOG_MODULE_LEVELS=httpx:INFO must not write OpenAlex/NCBI keys."""
    import logging as _logging

    import httpx

    from kestrel_backend.logging_config import configure_logging

    configure_logging(log_level="INFO", module_levels={"httpx": "INFO"})
    url = httpx.URL("https://api.openalex.org/works?search=insulin&api_key=SECRETKEY123")
    _logging.getLogger("httpx").info('HTTP Request: %s %s "%s %d %s"', "GET", url, "HTTP/1.1", 200, "OK")
    _logging.getLogger("x").info("ncbi %s", "https://eutils.ncbi.nlm.nih.gov/x?db=pubmed&api_key=NCBIKEY99&term=y")
    out = capsys.readouterr().out
    assert "SECRETKEY123" not in out and "NCBIKEY99" not in out
    assert "api_key=***REDACTED***" in out and "search=insulin" in out and "term=y" in out
