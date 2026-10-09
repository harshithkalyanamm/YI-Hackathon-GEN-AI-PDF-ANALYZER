from pathlib import Path

from backend.preflight import check_transformer_model


def test_model_preflight_requires_local_config_and_weights(tmp_path: Path) -> None:
    missing = check_transformer_model(tmp_path)
    assert not missing.ready
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"test")
    ready = check_transformer_model(tmp_path)
    assert ready.ready
