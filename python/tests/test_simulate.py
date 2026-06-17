import numpy as np
from python.simulate import build_signal


def test_signal_length_and_finite():
    rng = np.random.default_rng(0)
    sig = build_signal("Healthy", 0.0, fs=2000, duration=0.5, rng=rng)
    assert len(sig) == 1000
    assert np.all(np.isfinite(sig))


def test_classes_differ():
    # Different classes should produce measurably different spectra/energy.
    def energy(klass):
        rng = np.random.default_rng(1)
        return float(np.sum(build_signal(klass, 0.1, fs=2000, duration=0.5, rng=rng) ** 2))
    assert energy("Overload") > energy("Healthy")
