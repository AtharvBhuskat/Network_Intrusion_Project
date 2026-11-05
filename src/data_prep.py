"""
data_prep.py
Loading, cleaning, label mapping, sampling for KDD Cup 1999 dataset.
Uses sklearn fetch_kddcup99 (10% subset) for manageable size.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

ATTACK_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos",
    "pod": "dos", "smurf": "dos", "teardrop": "dos",
    "mailbomb": "dos", "processtable": "dos", "udpstorm": "dos", "apache2": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l",
    "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l",
    "snmpguess": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "worm": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "httptunnel": "u2r", "ps": "u2r",
    "sqlattack": "u2r", "xterm": "u2r",
}

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_kdd_data():
    """Load KDD Cup 99 10% dataset via sklearn and return a clean DataFrame."""
    print("[data_prep] Fetching KDD Cup 99 (10% subset) via sklearn ...")
    kdd = fetch_kddcup99(subset=None, percent10=True, as_frame=True)
    df = kdd.frame.copy()

    # Decode byte-string columns to regular strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    # Rename 'labels' column to 'label'
    if "labels" in df.columns:
        df.rename(columns={"labels": "label"}, inplace=True)

    # Strip trailing period from labels (e.g., 'normal.' -> 'normal')
    df["label"] = df["label"].str.rstrip(".")

    print(f"[data_prep] Loaded shape: {df.shape}")
    return df


def map_labels(df):
    """Map exact attack names to 5 broad families + add binary target."""
    df = df.copy()
    df["attack_family"] = df["label"].map(ATTACK_MAP).fillna("unknown")
    df["is_suspicious"] = (df["attack_family"] != "normal").astype(int)

    # Report unknowns
    unknowns = df[df["attack_family"] == "unknown"]["label"].unique()
    if len(unknowns) > 0:
        print(f"[data_prep] Unknown labels mapped to 'unknown': {unknowns}")

    print("[data_prep] Attack family distribution:")
    print(df["attack_family"].value_counts())
    return df


def remove_duplicates(df):
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[data_prep] Removed {before - after} duplicate rows ({before} -> {after})")
    return df.reset_index(drop=True)


def sample_data(df, max_rows=100000, random_state=42):
    """Stratified sample to keep the dataset manageable while preserving class ratios."""
    if len(df) <= max_rows:
        print(f"[data_prep] No sampling needed ({len(df)} rows)")
        return df

    # Make sure tiny classes get at least a few rows
    frac = max_rows / len(df)
    parts = []
    for family, group in df.groupby("attack_family"):
        n = max(int(len(group) * frac), min(5, len(group)))
        parts.append(group.sample(n=n, random_state=random_state))
    sampled = pd.concat(parts).reset_index(drop=True)
    print(f"[data_prep] Sampled from {len(df)} to {len(sampled)} rows")
    print(sampled["attack_family"].value_counts())
    return sampled


def split_and_save(df, test_size=0.2, random_state=42):
    """Train/test split and save processed CSVs + label mapper."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["attack_family"]
    )

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[data_prep] Saved train ({len(train_df)}) -> {train_path}")
    print(f"[data_prep] Saved test  ({len(test_df)})  -> {test_path}")

    # Save label mapper
    label_mapper = {
        "attack_map": ATTACK_MAP,
        "families": sorted(df["attack_family"].unique().tolist()),
        "categorical_cols": CATEGORICAL_COLS,
    }
    mapper_path = os.path.join(MODELS_DIR, "label_mapper.pkl")
    joblib.dump(label_mapper, mapper_path)
    print(f"[data_prep] Saved label mapper -> {mapper_path}")

    return train_df, test_df


def run_data_prep():
    """Full data preparation pipeline."""
    df = load_kdd_data()
    df = map_labels(df)
    df = remove_duplicates(df)
    df = sample_data(df, max_rows=100000)
    train_df, test_df = split_and_save(df)

    # Print summary
    print("\n========== DATA PREP SUMMARY ==========")
    print(f"Total sampled rows : {len(train_df) + len(test_df)}")
    print(f"Training rows      : {len(train_df)}")
    print(f"Test rows           : {len(test_df)}")
    print(f"Features            : {len(COLUMN_NAMES)}")
    print(f"Attack families     : {sorted(df['attack_family'].unique().tolist())}")
    print(f"Null values         : {df.isnull().sum().sum()}")
    print("========================================\n")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_data_prep()
