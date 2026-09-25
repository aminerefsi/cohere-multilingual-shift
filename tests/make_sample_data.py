"""Write a tiny synthetic dataset for offline tests (not real reviews).

Each language has its own sentiment words; a few emojis are shared across
languages, so an English-only model transfers partially, as a real
language shift would.
"""
from pathlib import Path

import numpy as np
import pandas as pd

WORDS = {
    "en": {"positive": ["great", "love", "excellent", "perfect", "happy"],
           "negative": ["bad", "broken", "terrible", "awful", "refund"],
           "filler": ["the", "product", "delivery", "it", "was", "very", "and"]},
    "fr": {"positive": ["super", "adore", "excellent", "parfait", "content"],
           "negative": ["mauvais", "cassé", "horrible", "nul", "remboursement"],
           "filler": ["le", "produit", "livraison", "il", "était", "très", "et"]},
    "ar": {"positive": ["رائع", "أحب", "ممتاز", "مثالي", "سعيد"],
           "negative": ["سيء", "مكسور", "فظيع", "رديء", "استرجاع"],
           "filler": ["المنتج", "التوصيل", "كان", "جدا", "و", "هذا", "في"]},
}
SHARED = {"positive": ["👍", "😊"], "negative": ["👎", "😡"]}


def sentence(rng, lang, label):
    w = WORDS[lang]
    toks = list(rng.choice(w["filler"], 5)) + list(rng.choice(w[label], 2))
    if rng.random() < 0.6:
        toks.append(rng.choice(SHARED[label]))
    if rng.random() < 0.15:  # label noise
        toks.append(rng.choice(w["negative" if label == "positive" else "positive"]))
    rng.shuffle(toks)
    return " ".join(toks) + f" #{rng.integers(1_000_000)}"


def main(out_dir="tests/sample_data", n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for lang in WORDS:
        for split in ("train", "test"):
            rows = [{"text": sentence(rng, lang, lab), "label": lab}
                    for lab in ("positive", "negative") for _ in range(n)]
            pd.DataFrame(rows).to_csv(out / f"{lang}_{split}.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
