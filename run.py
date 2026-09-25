"""Run the experiment.

    python run.py                      # real run with Cohere (needs COHERE_API_KEY)
    python run.py --fake --data-dir tests/sample_data   # offline smoke test
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from shiftlab.embed import CohereEmbedder, FakeEmbedder
from shiftlab.experiment import run


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader, so the key never has to be written in code."""
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="embed-multilingual-v3.0", help="Cohere embedding model")
    ap.add_argument("--fake", action="store_true", help="offline fake embedder (no API key needed)")
    ap.add_argument("--three-class", action="store_true", help="include the 'neutral' label")
    ap.add_argument("--n-train", type=int, default=300, help="English training examples per class")
    ap.add_argument("--n-test", type=int, default=150, help="test examples per class and language")
    ap.add_argument("--data-dir", default=None, help="folder with {lang}_{split}.csv files")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    if args.fake:
        embedder = FakeEmbedder()
    else:
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise SystemExit("Set COHERE_API_KEY (in a .env file or your shell). See README.")
        embedder = CohereEmbedder(key, model=args.model)

    labels = ("positive", "neutral", "negative") if args.three_class else ("positive", "negative")
    r = run(embedder, labels=labels, n_train=args.n_train, n_test=args.n_test,
            data_dir=args.data_dir, out_dir=args.out)

    print(f"\nEmbedder: {r['embedder']} | API calls this run: {r['api_calls_this_run']}")
    for lang, m in r["zero_shot"].items():
        print(f"  {lang}: acc={m['accuracy']:.3f}  ECE={m['ece']:.3f}  "
              f"overconf={m['overconfidence']:+.3f}  acc@80%cov={m['acc_at_80pct_coverage']:.3f}")
    print(f"Full tables: {args.out}/results.md | plot: {args.out}/adaptation.png")


if __name__ == "__main__":
    main()
