"""Load balanced sentiment samples for English, French and Arabic.

Source: the Hugging Face dataset `tyqiangz/multilingual-sentiments`
(configs "english", "french", "arabic"; labels positive / neutral / negative).

Three loading paths, tried in order:
  1. local CSVs in --data-dir  (files: en_train.csv, en_test.csv, fr_train.csv, ...
     each with columns `text,label`)
  2. `datasets.load_dataset`
  3. the auto-converted parquet files on the Hugging Face Hub
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HF_DATASET = "tyqiangz/multilingual-sentiments"
HF_CONFIGS = {"en": "english", "fr": "french", "ar": "arabic"}
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with exactly two columns: text (str) and label (str)."""
    text_col = next(c for c in ("text", "review_body", "sentence") if c in df.columns)
    label_col = next(c for c in ("label", "sentiment") if c in df.columns)
    out = pd.DataFrame({"text": df[text_col].astype(str), "label": df[label_col]})
    if pd.api.types.is_numeric_dtype(out["label"]):
        out["label"] = out["label"].map(ID2LABEL)
    out["label"] = out["label"].astype(str).str.lower().str.strip()
    out = out[out["text"].str.strip().str.len() > 0]
    return out.drop_duplicates("text").reset_index(drop=True)


def _load_raw(lang: str, split: str, data_dir: str | None) -> pd.DataFrame:
    if data_dir:
        path = Path(data_dir) / f"{lang}_{split}.csv"
        return pd.read_csv(path, encoding="utf-8")

    config = HF_CONFIGS[lang]
    errors = []
    try:
        from datasets import load_dataset

        return load_dataset(HF_DATASET, config, split=split).to_pandas()
    except Exception as exc:  # noqa: BLE001 - fall through to the next loader
        errors.append(f"datasets.load_dataset: {exc}")
    try:
        url = f"hf://datasets/{HF_DATASET}@~parquet/{config}/{split}/0000.parquet"
        return pd.read_parquet(url)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"parquet export: {exc}")
    raise RuntimeError(
        f"Could not load {lang}/{split}.\n" + "\n".join(errors)
        + "\nFix: download the data as CSV (text,label) and pass --data-dir."
    )


def load_split(
    lang: str,
    split: str,
    n_per_class: int,
    labels: tuple[str, ...],
    seed: int = 0,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Balanced random sample: `n_per_class` rows for each label in `labels`."""
    df = _normalise(_load_raw(lang, split, data_dir))
    df = df[df["label"].isin(labels)]
    available = min(int((df["label"] == label).sum()) for label in labels)
    if available == 0:
        raise ValueError(f"{lang}/{split}: no rows for one of the labels {labels}")
    if available < n_per_class:
        print(f"  note: {lang}/{split} has only {available} rows for its rarest label; "
              f"using {available}/class instead of {n_per_class}")
        n_per_class = available
    parts = []
    for label in labels:
        pool = df[df["label"] == label]
        parts.append(pool.sample(n_per_class, random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["lang"] = lang
    return out
