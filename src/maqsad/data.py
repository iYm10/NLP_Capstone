"""Pinned dataset retrieval and leakage-aware splitting."""
import hashlib
import re
from pathlib import Path
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

COMMIT = "eb85513e133cfc67e30aca4968fadbe2594fcb42"
CHECKSUMS = {
    "training_data.csv": "1da727eb763459f436c4e61f52c6a0bc63d5300af8fcc99b5675ae4616dae04a",
    "testing_data.csv": "0f0a77ba8a4a0a9370846a42a93a63244092f1fb30a909595b70e17e299bcf6c",
}
SENTIMENT = {"NEG": 0, "NEU": 1, "POS": 2}


def normalize(text):
    # Preserve negation, emoji, punctuation and Arabic letter distinctions.
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", str(text))
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def download(directory="data/raw"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in CHECKSUMS.items():
        path = directory / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/{COMMIT}/ArSarcasm-v2/{name}"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            content = response.content
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {name}")
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {path}; preserve and inspect the file")
    return directory


def load_file(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(df.columns) != {"tweet", "sarcasm", "sentiment", "dialect"}:
        raise ValueError("Unexpected dataset schema")
    if not df.sarcasm.str.upper().isin(["TRUE", "FALSE"]).all():
        raise ValueError("Invalid sarcasm labels")
    if not df.sentiment.isin(SENTIMENT).all():
        raise ValueError("Invalid sentiment labels")
    df["sarcasm_label"] = df.sarcasm.str.upper().eq("TRUE").astype(int)
    df["sentiment_label"] = df.sentiment.map(SENTIMENT)
    df["text"] = df.tweet.map(normalize)
    if df.text.eq("").any():
        raise ValueError("Empty text after normalization")
    return df


def prepare(directory="data/raw", seed=42):
    directory = download(directory)
    original_train = load_file(directory / "training_data.csv")
    test = load_file(directory / "testing_data.csv")
    # Keep the official test set intact. Remove matching normalized texts from
    # training before validation splitting; never use test labels for selection.
    train = original_train.loc[~original_train.text.isin(set(test.text))].copy()
    overlap_removed = len(original_train) - len(train)
    unique = train.drop_duplicates("text", keep="first").copy()
    duplicates_removed = len(train) - len(unique)
    strata = unique.sarcasm_label.astype(str) + "_" + unique.sentiment_label.astype(str)
    if strata.value_counts().min() < 2:
        strata = unique.sentiment_label
    train, validation = train_test_split(unique, test_size=0.15, random_state=seed, stratify=strata)
    audit = {"seed": seed, "original_train": len(original_train), "test": len(test),
             "train_test_overlap_removed": overlap_removed,
             "train_duplicates_removed": duplicates_removed,
             "test_duplicate_texts": int(test.text.duplicated().sum()),
             "train": len(train), "validation": len(validation)}
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), audit
