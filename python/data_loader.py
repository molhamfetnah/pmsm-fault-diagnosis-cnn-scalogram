def class_to_index(classes):
    return {c: i for i, c in enumerate(classes)}


def make_dataset(df, split, signal_type, classes, image_size, batch_size, seed, augment=False):
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
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), list(classes)
