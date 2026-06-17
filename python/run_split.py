from python.config import load_config
from python.manifest import load_manifest, save_manifest
from python.split import assign_splits

if __name__ == "__main__":
    cfg = load_config()
    df = load_manifest(cfg["paths"]["manifest"])
    df = assign_splits(df, seed=cfg["seed"])
    save_manifest(df, cfg["paths"]["manifest"])
    print(df.groupby(["split", "class"]).size())
