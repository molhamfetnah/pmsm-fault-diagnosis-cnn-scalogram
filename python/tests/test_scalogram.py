import numpy as np
from PIL import Image
from python.scalogram import compute_scalogram, save_scalogram_png


def test_compute_scalogram_shape():
    fs = 2000
    sig = np.sin(2 * np.pi * 100 * np.arange(fs) / fs)
    A = compute_scalogram(sig, fs, n_scales=32)
    assert A.shape == (32, fs)
    assert np.all(A >= 0)


def test_save_scalogram_png(tmp_path):
    fs = 2000
    sig = np.sin(2 * np.pi * 100 * np.arange(fs) / fs)
    out = tmp_path / "s.png"
    save_scalogram_png(sig, fs, str(out), image_size=64, n_scales=32)
    img = Image.open(out)
    assert img.size == (64, 64)
    assert img.mode == "RGB"
