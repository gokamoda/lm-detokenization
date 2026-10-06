"""Position-derived terms of the first-layer attention score (Section 5).

With LN, a position term for key position j is a_j / sigma_{j,v}, where
sigma_{j,v} is the std of e_v + p_j and v runs over the vocabulary. The
figures show, for each j, the mean over v and the 95% percentile interval
over v. Since a_j does not depend on v, both follow from statistics of
1 / sigma_{j,v} alone:

    mean_v(a_j / sigma_{j,v})        = a_j * mean_v(1 / sigma_{j,v})
    percentile_q(a_j / sigma_{j,v})  = a_j * percentile_q(1 / sigma_{j,v})      (a_j >= 0)
                                     = a_j * percentile_{100-q}(1 / sigma_{j,v}) (a_j < 0)

This equals seaborn's lineplot(estimator="mean", errorbar=("pi", 95)) over
the (position, vocabulary) grid that the published code built explicitly.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torchtyping import TensorType

from lm_detokenization.scores import compute_compare_score, compute_self_score
from lm_detokenization.tensor_types import HEAD, POS, VOCAB
from lm_detokenization.weights import Layer0Weights

# Which value of sigma_i (over the vocabulary at the query position) to use.
SIGMA_I_KINDS = ("mean", "max", "min")


@dataclass
class Band:
    """Mean and 95% percentile interval over the vocabulary, per position."""

    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray


@dataclass
class InverseSigmaStats:
    mean: TensorType[POS]
    q025: TensorType[POS]
    q975: TensorType[POS]


def inverse_sigma_stats(sigma: TensorType[POS, VOCAB]) -> InverseSigmaStats:
    inv = (1 / sigma.float()).numpy()
    q025, q975 = np.percentile(inv, [2.5, 97.5], axis=-1)
    return InverseSigmaStats(
        mean=torch.from_numpy(inv.mean(axis=-1)),
        q025=torch.from_numpy(q025),
        q975=torch.from_numpy(q975),
    )


def band(a: TensorType[POS], stats: InverseSigmaStats) -> Band:
    """Band of a_j / sigma_{j,v} over v, for j = 0 .. len(a) - 1."""
    n = a.shape[0]
    mean, q025, q975 = stats.mean[:n], stats.q025[:n], stats.q975[:n]
    positive = a >= 0
    low = torch.where(positive, a * q025, a * q975)
    high = torch.where(positive, a * q975, a * q025)
    return Band(mean=(a * mean).numpy(), low=low.numpy(), high=high.numpy())


def sigma_i(weights: Layer0Weights, pos_i: int, kind: str) -> float:
    row = weights.sigma[pos_i].float()
    return {"mean": row.mean, "max": row.max, "min": row.min}[kind]().item()


def tp_raw(weights: Layer0Weights) -> TensorType[HEAD, POS]:
    """b_h^QK p_j^T for every head and position (no LN)."""
    return compute_self_score(j=weights.wpe.unsqueeze(0), w=weights.qk.b_qk)[0]


def tpp_raw(weights: Layer0Weights, pos_i: int) -> TensorType[HEAD, POS]:
    """p_i W_h^QK p_j^T for j = 0 .. pos_i (no LN)."""
    return compute_compare_score(
        i=weights.wpe[pos_i : pos_i + 1].unsqueeze(0),
        j=weights.wpe[: pos_i + 1].unsqueeze(0),
        w=weights.qk.w_qk,
    )[0, :, 0]


def tp_band(weights: Layer0Weights, stats: InverseSigmaStats, head: int) -> Band:
    """T^p_{j,h} for all positions j."""
    return band(tp_raw(weights)[head], stats)


def tpp_band(
    weights: Layer0Weights,
    stats: InverseSigmaStats,
    head: int,
    pos_i: int,
    sigma_i_kind: str = "mean",
) -> Band:
    """T^pp_{i,j,h} for j = 0 .. pos_i, with sigma_i fixed to one value."""
    a = tpp_raw(weights, pos_i)[head] / sigma_i(weights, pos_i, sigma_i_kind)
    return band(a, stats)


def tp_tpp_band(
    weights: Layer0Weights,
    stats: InverseSigmaStats,
    head: int,
    pos_i: int,
    sigma_i_kind: str = "mean",
) -> Band:
    """T^pp_{i,j,h} + T^p_{j,h} for j = 0 .. pos_i."""
    a = tpp_raw(weights, pos_i)[head] / sigma_i(weights, pos_i, sigma_i_kind)
    a = a + tp_raw(weights)[head, : pos_i + 1]
    return band(a, stats)


def tp_tpp_softmax(
    weights: Layer0Weights,
    stats: InverseSigmaStats,
    head: int,
    pos_i: int,
    sigma_i_kind: str = "mean",
) -> np.ndarray:
    """softmax_j of the vocabulary mean of T^pp + T^p, scaled by 1/sqrt(d_head)."""
    mean = torch.from_numpy(tp_tpp_band(weights, stats, head, pos_i, sigma_i_kind).mean)
    return torch.softmax(mean / weights.head_dim**0.5, dim=-1).numpy()
