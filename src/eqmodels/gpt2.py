from typing import Tuple

import torch
from torch import Tensor, nn
from torchtyping import TensorType
from transformers import GPT2LMHeadModel
from transformers.activations import ACT2FN
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP,
    GPT2Attention,
    GPT2Block,
)
from transformers.pytorch_utils import Conv1D

from utils.mytorchtyping import BATCH, HEAD, HIDDEN_DIM, SEQUENCE

from .utils.hooks import InterventionHook, ObservationHook

torch.set_printoptions(sci_mode=False)


def compute_compare_score(
    i: TensorType[BATCH, SEQUENCE, HIDDEN_DIM],
    j: TensorType[BATCH, SEQUENCE, HIDDEN_DIM],
    w: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM],
) -> TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE]:
    """Compute compare score

    Parameters
    ----------
    i : TensorType[BATCH, SEQUENCE, HIDDEN_DIM]
    j : TensorType[BATCH, SEQUENCE, HIDDEN_DIM]
    w : TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM]

    Returns
    -------
    TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE]
    """

    return torch.einsum(
        "bshd,bdt->bhst",
        torch.einsum(
            "bsd,hdi->bshi",
            i,
            w.to(i.device),
        ),
        j.transpose(-1, -2),
    )


def compute_self_score(
    j: TensorType[BATCH, SEQUENCE, HIDDEN_DIM],
    w: TensorType[HEAD, HIDDEN_DIM],
) -> TensorType[BATCH, HEAD, SEQUENCE]:
    """Compute self score

    Parameters
    ----------
    j : TensorType[BATCH, SEQUENCE, HIDDEN_DIM]
    w : TensorType[HEAD, HIDDEN_DIM]

    Returns
    -------
    TensorType[BATCH, HEAD, HIDDEN_DIM]
    """
    return torch.einsum(
        "hd,bdt->bht",
        w.to(j.device),
        j.transpose(-1, -2),
    )


def collapse_ln(
    fc: Conv1D | nn.Linear,
    ln: nn.LayerNorm,
):
    """Collapse weights and biases of ln_1.

    Parameters
    ----------
    ln_weight : TensorType[HIDDEN_DIM]
    ln_bias : TensorType[HIDDEN_DIM]
    """

    centering = torch.diag(torch.ones(ln.weight.shape[0])) - 1 / ln.weight.shape[0]
    centering = centering.to(ln.weight.device)

    if isinstance(fc, nn.Linear):
        fc_new = nn.Linear(fc.in_features, fc.out_features, bias=True)
        fc_new.bias = nn.Parameter(ln.bias.clone() @ fc.weight.T.clone())
        fc_new.weight = nn.Parameter(
            (centering @ torch.diag(ln.weight) @ fc.weight.T).T
        )
        return fc_new

    if isinstance(fc, Conv1D):
        fc.bias = nn.Parameter(ln.bias @ fc.weight + fc.bias)
        fc.weight = nn.Parameter(centering @ torch.diag(ln.weight) @ fc.weight)
        return fc


class EQGPT2LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        return hidden_states / torch.sqrt(var + self.eps)


class EQGPT2Attention(nn.Module):
    """GPT2Attention with hooks pre-computed wvo and bvo."""

    def __init__(
        self,
        config,
        attn: GPT2Attention = None,
        ln: nn.LayerNorm = None,
        is_cross_attention: bool = False,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            raise NotImplementedError

        if attn is None:
            raise NotImplementedError

        if ln is None:
            raise NotImplementedError

        c_attn = collapse_ln(attn.c_attn, ln)
        self.wvo, self.bvo = self.compute_wvo(c_attn, attn.c_proj)
        self.wqkh, self.bqwkh = self.compute_wqk(c_attn)

        self.ln = EQGPT2LayerNorm(eps=config.layer_norm_epsilon)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()
        self.attn_value_dropout = nn.Dropout(0)
        self.intervention_hook_before_attn_weight_calc = InterventionHook()
        self.observation_hook_after_intervention_before_attn_weight_calc = (
            ObservationHook()
        )
        # self.for_hook_before_qk = ForwardHook()
        # self.for_hook = ForwardHook()

    def _my_attn(
        self,
        hidden_states: TensorType[BATCH],
        value,
        attention_mask=None,
        head_mask=None,
    ) -> tuple[
        TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM],
        TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE],
    ]:
        """_summary_

        Parameters
        ----------
        query : _type_
            _description_
        key : _type_
            _description_
        value : _type_
            _description_
        attention_mask : _type_, optional
            _description_, by default None
        head_mask : _type_, optional
            _description_, by default None

        Returns
        -------
        Tuple[
            weighted_value:
                TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM],
                attn_output = values_by_head_and_seq.sum(dim=-2)
            attn_weights: TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE],
        ]
            _description_
        """
        _, s, _ = hidden_states.size()

        compare_score = compute_compare_score(
            i=hidden_states,
            j=hidden_states,
            w=self.wqkh.to(hidden_states.device),
        )

        # compare_score = torch.einsum(
        #     "bshd,bdt->bhst",
        #     torch.einsum(
        #         "bsd,hdi->bshi",
        #         hidden_states,
        #         self.wqkh.to(hidden_states.device),
        #     ),
        #     hidden_states.transpose(-1, -2),
        # )

        self_score = compute_self_score(
            j=hidden_states,
            w=self.bqwkh.to(hidden_states.device),
        )

        # self_score = torch.einsum(
        #     "hd,bdt->bht",
        #     self.bqwkh.to(hidden_states.device),
        #     hidden_states.transpose(-1, -2),
        # )

        attn_scores = compare_score + self_score.unsqueeze(-2)

        if self.scale_attn_weights:
            attn_scores = attn_scores / torch.full(
                [],
                # value.size(-1)
                self.head_dim**0.5,
                dtype=attn_scores.dtype,
                device=attn_scores.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            raise NotImplementedError

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = s, s
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ]
            mask_value = torch.finfo(attn_scores.dtype).min
            # Need to be a tensor, otherwise we get error:
            #   `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise
            #   `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full(
                [],
                mask_value,
                dtype=attn_scores.dtype,
                device=attn_scores.device,
            )

            causal_mask = causal_mask.to(attn_scores.device)

            attn_scores = torch.where(
                causal_mask, attn_scores.to(attn_scores.dtype), mask_value
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)

        # logger.info(gate_score_dim_ablate[0][0])

        # Downcast (if necessary) back to V's dtype (if in mixed-precision)
        #  -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        value = self.attn_value_dropout(value)
        weighted_value = torch.einsum("bhij,bhjd->bhijd", attn_weights, value)

        # attn_output = torch.matmul(attn_weights, value)

        return weighted_value, attn_weights, attn_scores

    def compute_wvo(self, c_attn: Conv1D, c_proj: Conv1D):
        """Pre-compute wvo and bvo."""
        wv = c_attn.weight[:, -self.embed_dim :]
        bv = c_attn.bias[-self.embed_dim :]
        wo = c_proj.weight
        bo = c_proj.bias

        wvh = wv.T.view(self.num_heads, self.head_dim, self.embed_dim).transpose(-1, -2)
        woh = wo.view(self.num_heads, self.head_dim, self.embed_dim)

        wvo = wvh @ woh  # shape = (num_heads, embed_dim, embed_dim)
        bvo = bv @ wo + bo

        return wvo, bvo

    def compute_wqk(self, c_attn: Conv1D):
        """Pre-compute wqk."""
        wq = c_attn.weight[:, : self.embed_dim]
        wk = c_attn.weight[:, self.embed_dim : self.embed_dim * 2]
        bq = c_attn.bias[: self.embed_dim]

        wqh = wq.T.view(self.num_heads, self.head_dim, self.embed_dim).transpose(-1, -2)
        wkh = wk.T.view(self.num_heads, self.head_dim, self.embed_dim)
        wqkh = wqh @ wkh  # (h, d, d)

        bqh = bq.view(self.num_heads, self.head_dim)
        bqwkh = torch.einsum("he,hed->hd", bqh, wkh)

        return wqkh, bqwkh

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor],
        layer_past: tuple[torch.Tensor] = None,
        attention_mask: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor]]:
        if encoder_hidden_states is not None:
            raise NotImplementedError

        value = torch.einsum(
            "bsd,hdi->bhsi",
            self.ln(hidden_states),
            self.wvo.to(hidden_states.device),
        )

        if layer_past is not None:
            raise NotImplementedError

        if use_cache is True:
            raise NotImplementedError

        present = None

        if self.reorder_and_upcast_attn:
            raise NotImplementedError

        hidden_states = self.intervention_hook_before_attn_weight_calc(
            before=hidden_states
        )
        self.observation_hook_after_intervention_before_attn_weight_calc(
            hidden_states=hidden_states
        )

        (weighted_value, attn_weights, attn_scores) = self._my_attn(
            self.ln(hidden_states), value, attention_mask, head_mask
        )

        # self.for_hook(
        #     attn_weights=attn_weights.detach().to("cpu"),
        #     weighted_value=weighted_value.detach().to("cpu"),
        #     attn_scores=attn_scores.detach().to("cpu"),
        # )

        attn_output: TensorType[BATCH, SEQUENCE, HIDDEN_DIM] = (
            weighted_value.sum(dim=-2).permute(0, 2, 1, 3)
        ).sum(dim=2)
        attn_output = attn_output + self.bvo.to(hidden_states.device)
        # attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class EQGPT2MLP(nn.Module):
    """GPT2MLP with hooks."""

    def __init__(self, intermediate_size, config, ln: nn.LayerNorm, mlp: GPT2MLP):
        if mlp is None:
            raise NotImplementedError
        if ln is None:
            raise NotImplementedError

        super().__init__()

        self.c_fc = collapse_ln(mlp.c_fc, ln)
        self.c_proj = mlp.c_proj
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.intervention_hook_before_mlp = InterventionHook()
        self.observatoin_hook_after_intervention_before_mlp = ObservationHook()
        # self.for_hook = ForwardHook()

    def forward(self, hidden_states: tuple[torch.FloatTensor]) -> torch.FloatTensor:
        hidden_states = self.intervention_hook_before_mlp(before=hidden_states)
        self.observatoin_hook_after_intervention_before_mlp(hidden_states=hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        # self.for_hook(activation=hidden_states.detach().to("cpu"))
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


GPT2_ATTENTION_CLASSES = {"eager": GPT2Attention}


class EQGPT2Block(nn.Module):
    attn: EQGPT2Attention
    mlp: EQGPT2MLP
    ln_1: EQGPT2LayerNorm
    ln_2: EQGPT2LayerNorm

    def __init__(self, config, layer_idx=None, gpt2_block: GPT2Block = None):
        if gpt2_block is None:
            raise NotImplementedError
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.attn = EQGPT2Attention(
            config=config,
            attn=gpt2_block.attn,
            ln=gpt2_block.ln_1,
        )

        if config.add_cross_attention:
            raise NotImplementedError

        self.ln_2 = EQGPT2LayerNorm(eps=config.layer_norm_epsilon)
        self.mlp = EQGPT2MLP(
            intermediate_size=inner_dim,
            config=config,
            ln=gpt2_block.ln_2,
            mlp=gpt2_block.mlp,
        )

        self.intervention_hook_before_block = InterventionHook()
        self.observatoin_hook_after_intervention_before_block = ObservationHook()

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor],
        layer_past: tuple[torch.Tensor] = None,
        attention_mask: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> (
        tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.FloatTensor, ...]] | None
    ):
        hidden_states = self.intervention_hook_before_block(before=hidden_states)
        self.observatoin_hook_after_intervention_before_block(
            hidden_states=hidden_states
        )

        residual = hidden_states

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            raise NotImplementedError

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            raise NotImplementedError
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class EQGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "EQGPT2LMHeadModel":
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        model.transformer.h = nn.ModuleList(
            [
                EQGPT2Block(
                    config=model.config,
                    gpt2_block=block,
                )
                for block in model.transformer.h
            ]
        )
        cls.lm_head = collapse_ln(model.lm_head, model.transformer.ln_f)
        model.transformer.ln_f = EQGPT2LayerNorm(eps=model.config.layer_norm_epsilon)
        return model
