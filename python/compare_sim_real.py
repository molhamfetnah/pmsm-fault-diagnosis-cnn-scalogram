import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from python.config import load_config
from python.manifest import load_manifest


def main(cfg):
    df = load_manifest(cfg["paths"]["manifest"])
    df = df[df["signal_type"] == "current"]
    sources = ["sim", "real"]
    classes = cfg["classes"]
    fig, axes = plt.subplots(len(sources), len(classes), figsize=(3 * len(classes), 6))
    for r, src in enumerate(sources):
        for c, klass in enumerate(classes):
            sub = df[(df["source"] == src) & (df["class"] == klass) & (df["scalogram_path"] != "")]
            ax = axes[r][c]
            ax.axis("off")
            ax.set_title(f"{src}/{klass}", fontsize=8)
            if len(sub):
                ax.imshow(Image.open(sub.iloc[0]["scalogram_path"]))
    fig.tight_layout()
    fig.savefig(f"{cfg['paths']['results']}/sim_vs_real_grid.png", dpi=150)


if __name__ == "__main__":
    main(load_config())
