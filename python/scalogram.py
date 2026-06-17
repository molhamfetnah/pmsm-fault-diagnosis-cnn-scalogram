"""Python Continuous Wavelet Transform scalogram generation.

A MATLAB-free alternative to matlab/scalogram/*.m: renders the magnitude CWT
scalogram of a 1-D signal to a fixed-size RGB PNG using PyWavelets. This makes
the whole pipeline runnable by researchers without MATLAB / Wavelet Toolbox.
"""
import os
import numpy as np
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

from python.config import load_config
from python.manifest import load_manifest, save_manifest


def compute_scalogram(sig, fs, *, wavelet="cmor1.5-1.0", n_scales=128):
    """Return the |CWT| magnitude matrix (n_scales x len(sig))."""
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    scales = np.geomspace(4, 256, n_scales)
    coef, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0 / fs)
    return np.abs(coef)


def save_scalogram_png(sig, fs, out_path, *, image_size=224,
                       wavelet="cmor1.5-1.0", n_scales=128, cmap="jet"):
    """Render and save a square RGB scalogram PNG for SIG."""
    A = compute_scalogram(sig, fs, wavelet=wavelet, n_scales=n_scales)
    A = A / (A.max() + 1e-12)
    rgba = matplotlib.colormaps[cmap](A)          # (n_scales, len, 4) in 0..1
    rgb = (rgba[..., :3] * 255).astype("uint8")
    img = Image.fromarray(rgb).resize((image_size, image_size))
    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    img.save(out_path)
    return out_path


def generate_scalograms(cfg):
    """Render a scalogram PNG for every manifest row and fill scalogram_path."""
    df = load_manifest(cfg["paths"]["manifest"])
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    wavelet = cfg.get("wavelet_py", "cmor1.5-1.0")
    n_scales = cfg.get("n_scales", 128)
    size = cfg["image_size"]
    for i in range(len(df)):
        sid = df.at[i, "signal_id"]
        sig = np.load(os.path.join(seg_dir, sid + ".npy"))
        out_path = os.path.join(cfg["paths"]["scalograms"], df.at[i, "signal_type"],
                                df.at[i, "class"], sid + ".png")
        save_scalogram_png(sig, df.at[i, "fs"], out_path, image_size=size,
                           wavelet=wavelet, n_scales=n_scales)
        df.at[i, "scalogram_path"] = out_path
    save_manifest(df, cfg["paths"]["manifest"])
    return df


if __name__ == "__main__":
    generate_scalograms(load_config())
