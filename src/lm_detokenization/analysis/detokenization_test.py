import numpy as np
from sklearn.metrics import roc_auc_score

from lm_detokenization.analysis.detokenization import suffix_auroc


def published_auroc(scores: np.ndarray, counts: np.ndarray) -> float:
    """detokenization/step1_save_scores.py: positives repeated by their count."""
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return 0.0
    y_true = np.concatenate([np.zeros((counts == 0).sum()), np.ones(nonzero.sum())])
    y_score = np.concatenate(
        [scores[counts == 0], np.repeat(scores[counts > 0], nonzero)]
    )
    return roc_auc_score(y_true, y_score)


def test_weighted_auroc_equals_repeated_positives():
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores = rng.normal(size=300).round(1)  # ties included
        counts = np.where(rng.random(300) < 0.1, rng.integers(1, 20, 300), 0)
        assert np.isclose(suffix_auroc(scores, counts), published_auroc(scores, counts))


def test_no_positive_gives_zero():
    assert suffix_auroc(np.arange(5.0), np.zeros(5, dtype=int)) == 0.0
