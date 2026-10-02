"""Contribution of each of the six terms to the first-layer attention weights
(Section 6.1).

For each document, query position i and head h, the contribution c of a term
is the KL divergence between the attention distribution over j <= i with all
six terms and the one without that term, as in the published code
(torch.nn.functional.kl_div with reduction="mean", i.e. divided by i + 1,
and no 1/sqrt(d_head) scaling).
"""

from pathlib import Path

import numpy as np
import torch
from torchtyping import TensorType
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from lm_detokenization.scores import compute_compare_score, compute_self_score
from lm_detokenization.tensor_types import (
    HEAD,
    HIDDEN_DIM,
    POS,
    SEQUENCE,
    TERM,
    VOCAB,
)
from lm_detokenization.weights import Layer0Weights

# Keys of compute_6terms, in the order the contributions are stored.
TERMS = ("embi_embj", "embj", "posi_posj", "posj", "embi_posj", "posi_embj")
# Names used in the paper (T^ee, T^e, ...).
TERM_LABELS = {
    "embi_embj": "ee",
    "embj": "e",
    "posi_posj": "pp",
    "posj": "p",
    "embi_posj": "ep",
    "posi_embj": "pe",
}
MAX_LENGTH = 1024


def compute_6terms(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    wte: TensorType[VOCAB, HIDDEN_DIM],
    wpe: TensorType[POS, HIDDEN_DIM],
    w_compare: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM],
    w_self: TensorType[HEAD, HIDDEN_DIM],
) -> dict[str, torch.Tensor]:
    tokens = tokenizer(prompt, return_tensors="pt")

    if tokens.input_ids.shape[1] > MAX_LENGTH:
        tokens.input_ids = tokens.input_ids[:, :MAX_LENGTH]
        tokens.attention_mask = tokens.attention_mask[:, :MAX_LENGTH]

    tok_emb: TensorType[1, SEQUENCE, HIDDEN_DIM] = wte[tokens.input_ids]
    pos_enc: TensorType[1, SEQUENCE, HIDDEN_DIM] = wpe[
        : tokens.input_ids.shape[1]
    ].unsqueeze(0)

    variances: TensorType[SEQUENCE] = (tok_emb + pos_enc).var(
        dim=-1, keepdim=True, unbiased=False
    )
    variances = torch.sqrt(variances + 1e-5)

    tok_emb = tok_emb / variances
    pos_enc = pos_enc / variances

    embi_embj = compute_compare_score(tok_emb, tok_emb, w_compare)
    embj = compute_self_score(tok_emb, w_self)
    posi_posj = compute_compare_score(pos_enc, pos_enc, w_compare)
    posj = compute_self_score(pos_enc, w_self)
    embi_posj = compute_compare_score(tok_emb, pos_enc, w_compare)
    posi_embj = compute_compare_score(pos_enc, tok_emb, w_compare)

    return {
        "posi_posj": posi_posj,
        "posj": posj,
        "embi_embj": embi_embj,
        "embj": embj,
        "embi_posj": embi_posj,
        "posi_embj": posi_embj,
    }


def _as_matrix(term: torch.Tensor, length: int) -> TensorType[HEAD, SEQUENCE, SEQUENCE]:
    """[1, H, L, L] as is; [1, H, L] (terms of j only) broadcast over i."""
    term = term[0]
    return term if term.dim() == 3 else term.unsqueeze(1).expand(-1, length, -1)


def contributions(
    scores: dict[str, torch.Tensor],
) -> TensorType[HEAD, SEQUENCE, TERM]:
    """KL contribution of each term in TERMS, for every head and query position."""
    length = scores["posj"].shape[-1]
    matrices = {name: _as_matrix(scores[name], length).float() for name in TERMS}
    total = sum(matrices.values())
    future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    neg_inf = torch.tensor(float("-inf"))

    log_p = torch.log_softmax(torch.where(future, neg_inf, total), dim=-1)
    p = log_p.exp()
    row_length = torch.arange(1, length + 1, dtype=torch.float32)
    out = []
    for name in TERMS:
        log_q = torch.log_softmax(
            torch.where(future, neg_inf, total - matrices[name]), dim=-1
        )
        # kl_div(input=log q, target=p) = sum_j p_j (log p_j - log q_j), with
        # p_j = 0 contributing 0; reduction="mean" divides by i + 1
        pointwise = torch.where(p > 0, p * (log_p - log_q), torch.zeros(()))
        out.append(pointwise.sum(dim=-1) / row_length)
    return torch.stack(out, dim=-1)


def compute_contributions(
    weights: Layer0Weights,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    output_path: Path,
) -> None:
    """Save the contributions of `texts` as a float32 array
    [document, head, query position, term], NaN beyond each document's length.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(texts), weights.num_heads, MAX_LENGTH, len(TERMS)),
    )
    out[:] = np.nan
    with torch.no_grad():
        for k, text in enumerate(tqdm(texts, desc="six terms")):
            scores = compute_6terms(
                prompt=text,
                tokenizer=tokenizer,
                wte=weights.wte,
                wpe=weights.wpe,
                w_compare=weights.qk.w_qk,
                w_self=weights.qk.b_qk,
            )
            c = contributions(scores).numpy()
            out[k, :, : c.shape[1]] = c
    out.flush()
