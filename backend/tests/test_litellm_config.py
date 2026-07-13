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
