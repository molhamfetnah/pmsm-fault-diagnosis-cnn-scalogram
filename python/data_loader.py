def class_to_index(classes):
    return {c: i for i, c in enumerate(classes)}


def spec_augment(x, y):
    """SpecAugment-style aug for scalograms: random flip + noise + one frequency
    (row) and one time (column) mask. Pure-TF so it runs inside tf.data."""
    import tensorflow as tf
    x = tf.image.random_flip_left_right(x)
    x = x + tf.random.normal(tf.shape(x), stddev=0.02)
    h, w = tf.shape(x)[0], tf.shape(x)[1]
    fh = tf.random.uniform([], 0, h // 5 + 1, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, h - fh + 1, dtype=tf.int32)
    rmask = tf.cast((tf.range(h) < f0) | (tf.range(h) >= f0 + fh), x.dtype)[:, None, None]
    tw = tf.random.uniform([], 0, w // 5 + 1, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, w - tw + 1, dtype=tf.int32)
    cmask = tf.cast((tf.range(w) < t0) | (tf.range(w) >= t0 + tw), x.dtype)[None, :, None]
    return tf.clip_by_value(x * rmask * cmask, 0.0, 1.0), y


def make_dataset(df, split, signal_type, classes, image_size, batch_size, seed,
                 augment=False, aug_mode="flip"):
    # Lazy import: keeps class_to_index testable on machines without TensorFlow.
    import tensorflow as tf

    idx = class_to_index(classes)
    sub = df[(df["split"] == split) & (df["signal_type"] == signal_type)]
    paths = sub["scalogram_path"].tolist()
    labels = [idx[c] for c in sub["class"].tolist()]

    def _load(path, label):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, [image_size, image_size]) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        ds = ds.shuffle(len(paths), seed=seed)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = spec_augment if aug_mode == "spec" else \
            (lambda x, y: (tf.image.random_flip_left_right(x), y))
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), list(classes)
