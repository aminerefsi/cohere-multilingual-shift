"""The experiment.

Q1. Zero-shot transfer: a classifier trained only on English embeddings is
    applied to French and Arabic. How much do accuracy and calibration drop?
Q2. Abstention: if the model may hand its least-confident 20% to a human,
    how much of the lost accuracy comes back?
Q3. Few-shot adaptation: how many labelled target-language examples are needed
    to close the gap?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import load_split
from .embed import CachedEmbedder
from .metrics import evaluate

SOURCE = "en"
TARGETS = ("fr", "ar")


def _fit(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(X, y)


def run(
    embedder: CachedEmbedder,
    labels: tuple[str, ...] = ("positive", "negative"),
    n_train: int = 300,
    n_test: int = 150,
    n_adapt_pool: int = 128,
    shots: tuple[int, ...] = (0, 4, 16, 64),
    seeds: tuple[int, ...] = (0, 1, 2),
    data_dir: str | None = None,
    out_dir: str = "results",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------
    train = load_split(SOURCE, "train", n_train, labels, 0, data_dir)
    tests = {lang: load_split(lang, "test", n_test, labels, 0, data_dir) for lang in (SOURCE, *TARGETS)}
    pools = {lang: load_split(lang, "train", n_adapt_pool, labels, 0, data_dir) for lang in TARGETS}

    emb = lambda df: embedder.embed(df["text"].tolist())  # noqa: E731
    X_train, y_train = emb(train), train["label"].to_numpy()
    X_test = {lang: emb(df) for lang, df in tests.items()}
    X_pool = {lang: emb(df) for lang, df in pools.items()}

    # ---- Q1 + Q2: zero-shot transfer, calibration, abstention ------------
    clf = _fit(X_train, y_train, seed=0)
    zero_shot = {
        lang: evaluate(tests[lang]["label"].to_numpy(), clf.predict_proba(X_test[lang]), clf.classes_)
        for lang in tests
    }

    # ---- Q3: few-shot adaptation -----------------------------------------
    adaptation = []
    for lang in TARGETS:
        pool_y = pools[lang]["label"].to_numpy()
        for k in shots:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                idx = np.concatenate(
                    [rng.choice(np.flatnonzero(pool_y == lab), size=k, replace=False) for lab in labels]
                ) if k else np.array([], dtype=int)
                X = np.vstack([X_train, X_pool[lang][idx]]) if k else X_train
                y = np.concatenate([y_train, pool_y[idx]]) if k else y_train
                model = _fit(X, y, seed)
                m = evaluate(tests[lang]["label"].to_numpy(), model.predict_proba(X_test[lang]), model.classes_)
                adaptation.append({"lang": lang, "shots_per_class": k, "seed": seed, **m})

    # ---- save -------------------------------------------------------------
    results = {
        "embedder": embedder.name,
        "labels": list(labels),
        "n_train_per_class": len(train) // len(labels),
        "n_test_per_class": {lang: len(df) // len(labels) for lang, df in tests.items()},
        "api_calls_this_run": embedder.api_calls,
        "zero_shot": zero_shot,
        "adaptation": adaptation,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "results.md").write_text(_markdown(results), encoding="utf-8")
    _plot(pd.DataFrame(adaptation), zero_shot, out / "adaptation.png")
    return results


def _markdown(r: dict) -> str:
    names = {"en": "English (in-distribution)", "fr": "French (shifted)", "ar": "Arabic (shifted)"}
    lines = [
        f"# Results: `{r['embedder']}`, trained on English only",
        "",
        f"Labels: {', '.join(r['labels'])}. Train: {r['n_train_per_class']}/class (English). "
        "Test per class: " + ", ".join(f"{k} {v}" for k, v in r["n_test_per_class"].items()) + ".",
        "",
        "## Zero-shot transfer",
        "",
        "| Test language | Accuracy | Macro-F1 | Mean confidence | Overconfidence | ECE | Accuracy @ 80% coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for lang, m in r["zero_shot"].items():
        lines.append(
            f"| {names[lang]} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['mean_confidence']:.3f} | "
            f"{m['overconfidence']:+.3f} | {m['ece']:.3f} | {m['acc_at_80pct_coverage']:.3f} |"
        )
    df = pd.DataFrame(r["adaptation"])
    agg = df.groupby(["lang", "shots_per_class"])["accuracy"].agg(["mean", "std"]).reset_index()
    lines += [
        "",
        "## Few-shot adaptation (accuracy, mean ± std over seeds)",
        "",
        "| Language | Target examples per class | Accuracy |",
        "|---|---|---|",
    ]
    for _, row in agg.iterrows():
        lines.append(f"| {row['lang']} | {int(row['shots_per_class'])} | {row['mean']:.3f} ± {row['std']:.3f} |")
    lines += [
        "",
        "Overconfidence = mean confidence − accuracy (positive means the model is more sure than it should be).",
        "ECE = expected calibration error (10 bins). Accuracy @ 80% coverage: the model abstains on its "
        "20% least-confident cases.",
    ]
    return "\n".join(lines) + "\n"


def _plot(df: pd.DataFrame, zero_shot: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for lang, label in (("fr", "French"), ("ar", "Arabic")):
        g = df[df["lang"] == lang].groupby("shots_per_class")["accuracy"]
        m, s = g.mean(), g.std().fillna(0)
        ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3, label=label)
    ax.axhline(zero_shot["en"]["accuracy"], ls="--", color="grey", label="English (in-distribution)")
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xticks(sorted(df["shots_per_class"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Labelled target-language examples per class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Does adding target-language examples help?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
