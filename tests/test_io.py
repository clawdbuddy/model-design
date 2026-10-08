from pathlib import Path

from src.utils.io import load_config, save_json


def test_load_default_config():
    config = load_config("config/default.yaml")
    assert set(config) == {"constraints", "scoring", "proposal", "search"}
    assert config["scoring"]["method"] == "naswot"


def test_save_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "out.json"
    save_json(path, {"a": 1, "b": [1, 2]})
    assert path.exists()
    import json

    assert json.loads(path.read_text()) == {"a": 1, "b": [1, 2]}
