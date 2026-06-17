import numpy as np
from python.ingest_real import segment_signal


def test_segment_count_and_length():
    fs = 1000
    samples = np.arange(2000)            # 2 seconds
    segs = segment_signal(samples, fs=fs, window_seconds=0.5, overlap=0.5)
    # window = 500 samples, step = 250 -> (2000-500)/250 + 1 = 7 segments
    assert len(segs) == 7
    assert all(len(s) == 500 for s in segs)


def test_segment_no_overlap():
    segs = segment_signal(np.arange(1000), fs=1000, window_seconds=0.5, overlap=0.0)
    assert len(segs) == 2
