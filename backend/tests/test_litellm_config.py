from pathlib import Path
import yaml

CONFIG = Path(__file__).resolve().parents[2] / "deploy" / "litellm.config.yaml"

def test_config_enables_byok_forwarding_and_is_dbless():
    cfg = yaml.safe_load(CONFIG.read_text())
    gs = cfg["general_settings"]
    assert gs["forward_client_headers_to_llm_api"] is True
    assert gs["forward_llm_provider_auth_headers"] is True   # BYOK
    assert "database_url" not in gs and "database_url" not in cfg  # DB-less
    names = [m["model_name"] for m in cfg["model_list"]]
    assert any("claude" in n for n in names)


def test_config_has_wildcard_passthrough_entry():
    """The SDK usually omits `model` and sends its own unpinned default, which
    won't match the single claude-sonnet-4 alias. A wildcard entry must route
    any Anthropic model id through with the forwarded user key."""
    cfg = yaml.safe_load(CONFIG.read_text())
    wildcard = [m for m in cfg["model_list"] if m["model_name"] == "*"]
    assert len(wildcard) == 1
    assert wildcard[0]["litellm_params"]["model"] == "anthropic/*"
