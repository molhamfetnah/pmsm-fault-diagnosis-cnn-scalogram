from tensorflow import keras
from tensorflow.keras import layers, Model, Input


def build_cnn(input_shape=(224, 224, 3), num_classes=4, filters=(32, 64, 128)):
    """Baseline CNN. `filters` is one entry per Conv+MaxPool block, so its length
    sets the network depth (used by the depth ablation; default = 3 blocks)."""
    seq = [keras.Input(shape=input_shape)]
    for f in filters:
        seq += [layers.Conv2D(f, 3, activation="relu"), layers.MaxPooling2D()]
    seq += [
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
    return keras.Sequential(seq)


def _branch(x):
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    # Global average pooling instead of Flatten: a 26x26x128 map flattens to
    # ~86k features (huge dense layer, heavy memory); GAP gives 128 and is the
    # standard, far lighter choice for a two-branch fusion head.
    return layers.GlobalAveragePooling2D()(x)


def build_fusion_cnn(input_shape=(224, 224, 3), num_classes=4):
    ic = Input(shape=input_shape, name="current")
    iv = Input(shape=input_shape, name="vibration")
    merged = layers.concatenate([_branch(ic), _branch(iv)])
    z = layers.Dropout(0.5)(merged)
    z = layers.Dense(128, activation="relu")(z)
    out = layers.Dense(num_classes, activation="softmax")(z)
    return Model(inputs=[ic, iv], outputs=out)
