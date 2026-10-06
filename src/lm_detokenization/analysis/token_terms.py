"""Token-derived terms of the first-layer attention score (Sections 4 and 6)."""

from dataclasses import dataclass

import torch
from torchtyping import TensorType

from lm_detokenization.scores import compute_compare_score, compute_self_score
from lm_detokenization.tensor_types import HEAD, SAMPLE, VOCAB
from lm_detokenization.weights import Layer0Weights


@dataclass
class TeeSample:
    token_ids: TensorType[SAMPLE]
    # e_i W_h^QK e_j^T, without LN
    raw: TensorType[HEAD, SAMPLE, SAMPLE]
    # T^ee: raw / (sigma_i sigma_j), with sigma averaged over positions
    with_ln: TensorType[HEAD, SAMPLE, SAMPLE]


def tee_sample(weights: Layer0Weights, n_samples: int, seed: int = 42) -> TeeSample:
    """T^ee between n_samples tokens drawn uniformly from the vocabulary."""
    torch.manual_seed(seed)
    token_ids = torch.randint(0, weights.sigma.shape[1], (n_samples,))
    e = weights.wte[token_ids].unsqueeze(0)
    raw = compute_compare_score(i=e, j=e, w=weights.qk.w_qk)[0]
    sigma = weights.sigma[:, token_ids].mean(dim=0)
    denominator = sigma.unsqueeze(1) * sigma.unsqueeze(0)
    return TeeSample(token_ids=token_ids, raw=raw, with_ln=raw / denominator)


def te_raw(weights: Layer0Weights) -> TensorType[HEAD, VOCAB]:
    """b_h^QK e_v^T for every head and token (no LN)."""
    return compute_self_score(j=weights.wte.unsqueeze(0), w=weights.qk.b_qk)[0]


def te_with_ln(weights: Layer0Weights) -> TensorType[HEAD, VOCAB]:
    """T^e_{v,h} averaged over positions: mean_j (b_h^QK e_v^T / sigma_{j,v})."""
    inverse_sigma_mean = (1 / weights.sigma.float()).mean(dim=0)
    return te_raw(weights) * inverse_sigma_mean
