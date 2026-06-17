import pandas as pd

MANIFEST_COLUMNS = ["signal_id", "source", "signal_type", "class", "severity",
                    "fs", "dataset_name", "recording_id", "split", "scalogram_path"]


def new_manifest():
    return pd.DataFrame({c: pd.Series(dtype="object") for c in MANIFEST_COLUMNS})


def add_record(df, *, signal_id, source, signal_type, klass, severity, fs,
               dataset_name, recording_id):
    row = {"signal_id": signal_id, "source": source, "signal_type": signal_type,
           "class": klass, "severity": severity, "fs": fs,
           "dataset_name": dataset_name, "recording_id": recording_id,
           "split": "", "scalogram_path": ""}
    new = pd.DataFrame([row], columns=MANIFEST_COLUMNS)
    if df.empty:
        return new
    return pd.concat([df, new], ignore_index=True)


def save_manifest(df, path):
    df.to_csv(path, index=False)


def load_manifest(path):
    return pd.read_csv(path, dtype={"split": "string"}).fillna("")
