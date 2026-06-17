import json
import os
from python.config import load_config
from python.manifest import load_manifest


def main(cfg):
    df = load_manifest(cfg["paths"]["manifest"])
    lines = ["# Results Summary\n", "## Dataset composition\n"]
    lines.append(df.groupby(["signal_type", "source", "class"]).size().to_string())
    for st in cfg["signal_types"]:
        p = f"{cfg['paths']['results']}/report_{st}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            lines.append(f"\n## {st} model\n- accuracy: {r.get('accuracy')}\n")
            for k in cfg["classes"]:
                if k in r:
                    lines.append(f"- {k} F1: {r[k]['f1-score']:.3f}\n")
    open(f"{cfg['paths']['results']}/summary.md", "w").write("\n".join(map(str, lines)))
    print("wrote results/summary.md")


if __name__ == "__main__":
    main(load_config())
