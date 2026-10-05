"""Static QK weights of an attention layer, read from the model with
feature-extractor.

The attention score of an absolute-position model (GPT-2) is

    score(i, j) = LN(x_i)^T W_q^T W_k LN(x_j) + b_q^T W_k LN(x_j) + (terms of i only)

The terms of i only do not change the softmax over j, so they are dropped.
LN is folded into W_q/W_k so that the inputs only need to be divided by
their standard deviation.
"""

from dataclasses import dataclass

import torch
from feature_extractor.models import (
    get_absolute_pos_embedding_module,
    get_hidden_size_per_head,
    get_model_architecture,
    get_num_attn_heads,
    get_num_kv_heads,
    get_pre_attn_norm_module,
    get_word_embedding_module,
)
from feature_extractor.models.get_modules import get_k_proj_module, get_q_proj_module
from feature_extractor.reconstruction import precompute_qk_weights
from torch import nn
from torchtyping import TensorType
from transformers import PreTrainedModel

from lm_detokenization.tensor_types import HEAD, HIDDEN_DIM, POS, VOCAB


@dataclass
class QKWeights:
    # score(i, j) = x_i^T w_qk[h] x_j + b_qk[h]^T x_j
    w_qk: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM]
    b_qk: TensorType[HEAD, HIDDEN_DIM]


def fold_layer_norm_into_linear(linear: nn.Linear, ln: nn.LayerNorm) -> nn.Linear:
    """Return a Linear equivalent to `linear(ln(x))` when applied to x / std(x).

    LN(x) = gamma * C x / std(x) + beta, where C = I - 11^T / d is the centering
    matrix. Hence linear(LN(x)) = (W diag(gamma) C)(x / std(x)) + (W beta + b).
    """
    hidden_dim = ln.weight.shape[0]
    centering = torch.eye(hidden_dim, device=ln.weight.device) - 1 / hidden_dim

    folded = nn.Linear(linear.in_features, linear.out_features, bias=True)
    folded.weight = nn.Parameter(linear.weight @ torch.diag(ln.weight) @ centering)
    bias = linear.weight @ ln.bias
    if linear.bias is not None:
        bias = bias + linear.bias
    folded.bias = nn.Parameter(bias)
    return folded


@torch.no_grad()
def get_qk_weights(model: PreTrainedModel, layer_index: int = 0) -> QKWeights:
    architecture = get_model_architecture(model)
    if architecture.attn_use_rope:
        raise NotImplementedError("QK weights of RoPE models depend on positions.")

    ln = get_pre_attn_norm_module(architecture, layer_index, model=model)
    if not isinstance(ln, nn.LayerNorm):
        raise NotImplementedError(f"Unsupported pre-attention norm: {type(ln)}")

    q_proj = fold_layer_norm_into_linear(
        get_q_proj_module(architecture, layer_index, model=model), ln
    )
    k_proj = fold_layer_norm_into_linear(
        get_k_proj_module(architecture, layer_index, model=model), ln
    )

    w_qk, bias_terms = precompute_qk_weights(
        q_proj_module=q_proj,
        k_proj_module=k_proj,
        num_attention_heads=get_num_attn_heads(model.config, architecture),
        head_dim=get_hidden_size_per_head(model.config, architecture),
        num_kv_heads=get_num_kv_heads(model.config, architecture),
    )
    return QKWeights(w_qk=w_qk.cpu(), b_qk=bias_terms.key_side.cpu())


def get_word_embedding(model: PreTrainedModel) -> TensorType[VOCAB, HIDDEN_DIM]:
    module = get_word_embedding_module(get_model_architecture(model), model=model)
    return module.weight.detach().cpu()


def get_position_embedding(model: PreTrainedModel) -> TensorType[POS, HIDDEN_DIM]:
    module = get_absolute_pos_embedding_module(
        get_model_architecture(model), model=model
    )
    return module.weight.detach().cpu()


@dataclass
class Layer0Weights:
    """Everything the weight-based analyses of the first layer read."""

    wte: TensorType[VOCAB, HIDDEN_DIM]
    wpe: TensorType[POS, HIDDEN_DIM]
    qk: QKWeights
    # std of e_v + p_j (what LN divides by), as in the published code:
    # sqrt(var + 1e-5) in float16
    sigma: TensorType[POS, VOCAB]
    num_heads: int
    head_dim: int


def load_layer0_weights(model_name: str = "gpt2") -> Layer0Weights:
    """The weights of the first layer, for the tokens of the tokenizer only:
    rows of the embedding beyond them (rinna/japanese-gpt-1b has 44928 rows
    for 44877 tokens) are padding, never given to the model."""
    # imported here so that modules using only the dataclass stay light
    from feature_extractor.models import load_causal_model, load_tokenizer

    from lm_detokenization.layernorm import get_var_matrix

    model = load_causal_model(model_name, device="cpu")
    architecture = get_model_architecture(model)
    wte = get_word_embedding(model)[: len(load_tokenizer(model_name))]
    wpe = get_position_embedding(model)
    return Layer0Weights(
        wte=wte,
        wpe=wpe,
        qk=get_qk_weights(model, layer_index=0),
        sigma=torch.sqrt(get_var_matrix(wpe, wte) + 1e-5).to(torch.float16),
        num_heads=get_num_attn_heads(model.config, architecture),
        head_dim=get_hidden_size_per_head(model.config, architecture),
    )
