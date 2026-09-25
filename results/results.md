# Results: `embed-multilingual-v3.0`, trained on English only

Labels: positive, negative. Train: 300/class (English). Test per class: en 150, fr 150, ar 150.

## Zero-shot transfer

| Test language | Accuracy | Macro-F1 | Mean confidence | Overconfidence | ECE | Accuracy @ 80% coverage |
|---|---|---|---|---|---|---|
| English (in-distribution) | 0.923 | 0.923 | 0.791 | -0.133 | 0.147 | 0.988 |
| French (shifted) | 0.867 | 0.867 | 0.700 | -0.167 | 0.167 | 0.917 |
| Arabic (shifted) | 0.867 | 0.866 | 0.704 | -0.162 | 0.162 | 0.925 |

## Few-shot adaptation (accuracy, mean ± std over seeds)

| Language | Target examples per class | Accuracy |
|---|---|---|
| ar | 0 | 0.867 ± 0.000 |
| ar | 4 | 0.869 ± 0.014 |
| ar | 16 | 0.859 ± 0.012 |
| ar | 64 | 0.858 ± 0.013 |
| fr | 0 | 0.867 ± 0.000 |
| fr | 4 | 0.868 ± 0.008 |
| fr | 16 | 0.878 ± 0.012 |
| fr | 64 | 0.866 ± 0.004 |

Overconfidence = mean confidence − accuracy (positive means the model is more sure than it should be).
ECE = expected calibration error (10 bins). Accuracy @ 80% coverage: the model abstains on its 20% least-confident cases.
