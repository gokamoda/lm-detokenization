import numpy as np
import pytest
import torch

from lm_detokenization.analysis import position_terms as pt
from lm_detokenization.layernorm import get_var_matrix
from lm_detokenization.weights import Layer0Weights, QKWeights

NUM_HEADS, HIDDEN, POS, VOCAB = 2, 8, 12, 40


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator().manual_seed(0)
    wte = torch.randn(VOCAB, HIDDEN, generator=g)
    wpe = torch.randn(POS, HIDDEN, generator=g)
    qk = QKWeights(
        w_qk=torch.randn(NUM_HEADS, HIDDEN, HIDDEN, generator=g),
        b_qk=torch.randn(NUM_HEADS, HIDDEN, generator=g),
    )
    return Layer0Weights(
        wte=wte,
        wpe=wpe,
        qk=qk,
        sigma=torch.sqrt(get_var_matrix(wpe, wte) + 1e-5).to(torch.float16),
        num_heads=NUM_HEADS,
        head_dim=HIDDEN // NUM_HEADS,
    )


def explicit_band(values: torch.Tensor) -> pt.Band:
    """What seaborn's lineplot(errorbar=("pi", 95)) computes from a
    [POS, VOCAB] grid, as in the published code."""
    v = values.float().numpy()
    low, high = np.percentile(v, [2.5, 97.5], axis=-1)
    return pt.Band(mean=v.mean(axis=-1), low=low, high=high)


def assert_band_close(actual: pt.Band, expected: pt.Band):
    for name in ("mean", "low", "high"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-4, atol=1e-5
        )


def test_tp_band(weights):
    stats = pt.inverse_sigma_stats(weights.sigma)
    raw = pt.tp_raw(weights)
    for head in range(NUM_HEADS):
        # published: self_score / var_matrix, expanded over the vocabulary
        grid = raw[head].unsqueeze(-1) / weights.sigma
        assert_band_close(pt.tp_band(weights, stats, head), explicit_band(grid))


@pytest.mark.parametrize("kind", pt.SIGMA_I_KINDS)
def test_tpp_and_sum_band(weights, kind):
    stats = pt.inverse_sigma_stats(weights.sigma)
    pos_i = POS - 2
    var_i = pt.sigma_i(weights, pos_i, kind)
    for head in range(NUM_HEADS):
        tpp = pt.tpp_raw(weights, pos_i)[head] / var_i
        grid = tpp.unsqueeze(-1) / weights.sigma[: pos_i + 1]
        assert_band_close(
            pt.tpp_band(weights, stats, head, pos_i, kind), explicit_band(grid)
        )
        grid_sum = (
            grid
            + pt.tp_raw(weights)[head, : pos_i + 1].unsqueeze(-1)
            / weights.sigma[: pos_i + 1]
        )
        assert_band_close(
            pt.tp_tpp_band(weights, stats, head, pos_i, kind), explicit_band(grid_sum)
        )
        expected_softmax = torch.softmax(
            grid_sum.float().mean(dim=-1) / (HIDDEN / NUM_HEADS) ** 0.5, dim=-1
        ).numpy()
        np.testing.assert_allclose(
            pt.tp_tpp_softmax(weights, stats, head, pos_i, kind),
            expected_softmax,
            rtol=1e-4,
            atol=1e-6,
        )
