"""Reproducible synthetic PMSM stator-current generator.

A MATLAB-free signal source for the pipeline. Each operating state is modelled
with a physically-motivated Motor Current Signature Analysis (MCSA) fingerprint
on top of the fundamental, plus per-recording jitter so segments are not
identical. This is intended for (a) end-to-end pipeline validation and (b) users
without MATLAB; it complements — it does not replace — the MATLAB FOC/Simscape
simulation and the real measured datasets. Fault signatures are stylised
approximations of the literature, not a calibrated electromagnetic model.

Signatures (f0 = supply fundamental, fr = rotor mechanical frequency):
  Healthy          : f0 + small odd harmonics + noise
  InterTurn        : elevated 3rd/5th harmonics + sidebands, scaled by severity
  Demagnetization  : sub-harmonic + (f0 +/- k*fr) sidebands
  Overload         : larger fundamental, stronger harmonic distortion, lower SNR
"""
import os
import numpy as np

from python.config import load_config
from python.manifest import new_manifest

F0 = 50.0     # supply fundamental (Hz)
FR = 12.0     # rotor mechanical frequency (Hz)


def build_signal(klass, severity, *, fs, duration, rng):
    t = np.arange(int(fs * duration)) / fs
    # per-recording variation
    f0 = F0 * (1 + rng.uniform(-0.01, 0.01))
    phi = rng.uniform(0, 2 * np.pi)
    amp = 1.0 + rng.uniform(-0.05, 0.05)
    noise_std = 0.05

    x = amp * np.sin(2 * np.pi * f0 * t + phi)
    # baseline small odd harmonics present in any real winding
    x += 0.04 * np.sin(2 * np.pi * 3 * f0 * t)
    x += 0.02 * np.sin(2 * np.pi * 5 * f0 * t)

    if klass == "InterTurn":
        s = max(severity, 0.05)
        x += 0.30 * s * np.sin(2 * np.pi * 3 * f0 * t + rng.uniform(0, 1))
        x += 0.20 * s * np.sin(2 * np.pi * 5 * f0 * t + rng.uniform(0, 1))
        x += 0.15 * s * np.sin(2 * np.pi * (f0 + 2 * FR) * t)   # sideband
    elif klass == "Demagnetization":
        x += 0.18 * np.sin(2 * np.pi * 0.5 * f0 * t)            # sub-harmonic
        for k in (1, 2, 3):
            x += 0.10 * np.sin(2 * np.pi * (f0 + k * FR) * t)
            x += 0.10 * np.sin(2 * np.pi * (f0 - k * FR) * t)
    elif klass == "Overload":
        x = 1.6 * amp * np.sin(2 * np.pi * f0 * t + phi)
        x += 0.15 * np.sin(2 * np.pi * 2 * f0 * t)              # even harmonic distortion
        x += 0.10 * np.sin(2 * np.pi * 7 * f0 * t)
        noise_std = 0.10

    x += rng.normal(0, noise_std, size=t.shape)
    return x.astype(np.float32)


def generate(cfg, *, n_per_class=6, duration=5.0, out_subdir="synthetic"):
    fs = cfg["target_fs"]
    classes = cfg["classes"]
    out_dir = os.path.join(cfg["paths"]["raw"], "sim", out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    base_seed = cfg["seed"]
    for ci, klass in enumerate(classes):
        for r in range(n_per_class):
            rng = np.random.default_rng(base_seed + 1000 * ci + r)
            severity = 0.10 if klass == "InterTurn" else 0.0
            sig = build_signal(klass, severity, fs=fs, duration=duration, rng=rng)
            if klass == "InterTurn":
                name = f"{klass}_sigma{int(severity*100):02d}_run{r}"
            else:
                name = f"{klass}_run{r}"
            np.save(os.path.join(out_dir, name + ".npy"), sig)
    print(f"generated {len(classes) * n_per_class} synthetic recordings in {out_dir}")


if __name__ == "__main__":
    generate(load_config())
