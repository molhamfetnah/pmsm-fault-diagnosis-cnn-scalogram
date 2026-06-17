import pytest
from python.config import load_config


def test_load_config_has_required_keys(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "seed: 42\nclasses: [Healthy, InterTurn]\nsignal_types: [current]\n"
        "window_seconds: 0.5\noverlap: 0.5\ntarget_fs: 10000\nimage_size: 224\n"
        "wavelet: amor\npaths: {raw: data/raw, scalograms: s, manifest: m.csv, models: mo, results: r}\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg["classes"] == ["Healthy", "InterTurn"]
    assert cfg["image_size"] == 224


def test_load_config_missing_key_raises(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("seed: 42\n")
    with pytest.raises(ValueError):
        load_config(str(cfg_file))
