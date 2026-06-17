import yaml

REQUIRED = ["seed", "classes", "signal_types", "window_seconds", "overlap",
            "target_fs", "image_size", "wavelet", "paths"]


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    return cfg
