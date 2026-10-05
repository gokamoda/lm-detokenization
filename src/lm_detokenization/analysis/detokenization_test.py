import numpy as np
import pytest
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score

from lm_detokenization.analysis.detokenization import (
    TokenAffinity,
    batch_auroc,
    compute_auroc,
    suffix_auroc,
)


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


def accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


DEVICES = ["cpu"] + ([accelerator()] if accelerator() else [])


def random_rows(rng, n_rows=30, vocab=300):
    scores = rng.normal(size=(n_rows, vocab)).round(1).astype(np.float32)  # ties
    counts = np.where(
        rng.random((n_rows, vocab)) < 0.1, rng.integers(1, 10**6, (n_rows, vocab)), 0
    )
    counts[0] = 0  # a row without positives
    return scores, counts


@pytest.mark.parametrize("device", DEVICES)
def test_batch_auroc_equals_suffix_auroc(device):
    scores, counts = random_rows(np.random.default_rng(0))
    got = batch_auroc(
        torch.from_numpy(scores).to(device), torch.from_numpy(counts).to(device)
    )
    expected = [suffix_auroc(s, c) for s, c in zip(scores, counts)]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert got[0] == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_compute_auroc_on_a_device_equals_scikit_learn(device):
    g = torch.Generator().manual_seed(0)
    vocab, dim, heads = 200, 8, 3
    affinity = TokenAffinity(
        wte=torch.randn(vocab, dim, generator=g),
        w_qk=torch.randn(heads, dim, dim, generator=g),
    )
    rng = np.random.default_rng(1)
    counts = np.where(
        rng.random((vocab, vocab)) < 0.05, rng.integers(1, 50, (vocab, vocab)), 0
    )
    bigrams = sparse.csc_matrix(counts)
    suffix_ids = list(range(0, vocab, 3))
    expected = compute_auroc(affinity, bigrams, suffix_ids, batch_size=16)
    got = compute_auroc(affinity, bigrams, suffix_ids, batch_size=16, device=device)
    assert got.columns == expected.columns and got.dtypes == expected.dtypes
    assert got["suffix_id"].to_list() == expected["suffix_id"].to_list()
    assert got["head"].to_list() == expected["head"].to_list()
    assert (
        got["num_valid_detokenization"].to_list()
        == expected["num_valid_detokenization"].to_list()
    )
    np.testing.assert_allclose(got["auroc"], expected["auroc"], rtol=0, atol=1e-6)
