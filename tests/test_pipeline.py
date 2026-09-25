import json

import numpy as np

from shiftlab.embed import FakeEmbedder
from shiftlab.experiment import run
from shiftlab.metrics import accuracy_at_coverage, expected_calibration_error
from tests.make_sample_data import main as make_data


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([1.0, 1.0, 1.0])
    correct = np.array([1.0, 1.0, 1.0])
    assert expected_calibration_error(conf, correct) == 0.0


def test_ece_overconfident_is_large():
    conf = np.full(100, 0.95)
    correct = np.r_[np.ones(50), np.zeros(50)]
    assert abs(expected_calibration_error(conf, correct) - 0.45) < 1e-9


def test_abstention_keeps_most_confident():
    conf = np.array([0.9, 0.8, 0.6, 0.55])
    correct = np.array([1, 1, 0, 0], dtype=float)
    assert accuracy_at_coverage(conf, correct, 0.5) == 1.0


def test_end_to_end_offline(tmp_path):
    data = tmp_path / "data"
    make_data(str(data), n=120)
    r = run(FakeEmbedder(cache_dir=str(tmp_path / "cache")), n_train=100, n_test=60,
            n_adapt_pool=32, shots=(0, 4, 32), seeds=(0, 1),
            data_dir=str(data), out_dir=str(tmp_path / "out"))

    zs = r["zero_shot"]
    assert zs["en"]["accuracy"] > 0.9                      # in-distribution works
    assert zs["fr"]["accuracy"] < zs["en"]["accuracy"]     # language shift hurts
    ar = [a for a in r["adaptation"] if a["lang"] == "ar"]
    acc = lambda k: np.mean([a["accuracy"] for a in ar if a["shots_per_class"] == k])  # noqa: E731
    assert acc(32) > acc(0)                                # few target examples help
    for f in ("results.json", "results.md", "adaptation.png"):
        assert (tmp_path / "out" / f).exists()
    json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
