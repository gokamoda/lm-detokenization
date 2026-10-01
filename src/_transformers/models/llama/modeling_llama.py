from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torchtyping import TensorType
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg

from _transformers.utils.hooks import ObservationHook
from utils.mylogger import init_logging
from utils.mytorchtyping import (
    BATCH,
    HEAD,
    HEAD_DIM,
    HEAD_KV,
    HEAD_Q,
    HIDDEN_DIM,
    SEQUENCE,
    NHEADxDHEAD,
)

log_path = "test.log"
logger = init_logging(__name__, log_path=log_path, clear=False)


def collapse_rmsnorm(linear: nn.Linear, rmsnorm: LlamaRMSNorm):
    """
    Collapse rmsnorm weights into lienar layer.
    After collapsing, use EQLlamaRMSNorm instead of LlamaRMSNorm.
    """
    assert linear.bias is None
    linear_new = nn.Linear(linear.in_features, linear.out_features, bias=False)
    linear_new.weight = nn.Parameter((torch.diag(rmsnorm.weight) @ linear.weight.T).T)

    return linear_new


class EQLlamaRMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        """
        LlamaRMSNorm without weight.
        """
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"eps={self.variance_epsilon}"


class EQLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """

    logger.info(f"hidden_states.shape: {hidden_states.shape}")
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv_weights(
    proj: nn.Linear, num_attention_heads: int, num_key_value_heads: int
) -> nn.Linear:
    assert proj.bias is None

    hidden_size = proj.in_features
    n_repeat = num_attention_heads // num_key_value_heads
    weight: TensorType[HEAD_KV, HEAD_DIM, HIDDEN_DIM] = proj.weight.view(
        (num_key_value_heads, hidden_size // num_attention_heads, proj.in_features)
    )

    new_proj = nn.Linear(
        in_features=proj.in_features,
        out_features=proj.out_features * n_repeat,
        bias=proj.bias is not None,
    )
    new_proj.weight = nn.Parameter(
        weight.repeat_interleave(n_repeat, dim=0).reshape(
            (proj.out_features * n_repeat, proj.in_features)
        )
    )

    return new_proj


def eager_attention_forward(
    module: nn.Module,
    query: TensorType[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    key: TensorType[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    value: TensorType[BATCH, HEAD, SEQUENCE, HIDDEN_DIM],
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # logger.info(f"key.shape: {key.shape}")
    # logger.info(f"{module.num_key_value_groups=}")
    # key_states: TensorType[BATCH,] = repeat_kv(key, module.num_key_value_groups)
    # logger.info(f"key_states.shape: {key_states.shape}")
    # value_states = repeat_kv(value, module.num_key_value_groups)
    # logger.info(f"value_states.shape: {value_states.shape}")

    attn_scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_scores = attn_scores + causal_mask

    attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights: TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE] = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    weighted_value = torch.einsum("bhij,bhjd->bhijd", attn_weights, value)
    # attn_output = torch.matmul(attn_weights, value_states)
    # attn_output: TensorType[BATCH, SEQUENCE, HEAD, HEAD_DIM] = attn_output.transpose(
    #     1, 2
    # ).contiguous()

    return weighted_value, attn_weights, attn_scores


class EQLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        attn: LlamaAttention,
        rms_norm: LlamaRMSNorm,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = collapse_rmsnorm(
            attn.q_proj,
            rms_norm,
        )

        self.k_proj = repeat_kv_weights(
            collapse_rmsnorm(
                attn.k_proj,
                rms_norm,
            ),
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )

        self.v_proj = repeat_kv_weights(
            collapse_rmsnorm(
                attn.v_proj,
                rms_norm,
            ),
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )

        self.o_proj = attn.o_proj

        num_heads = config.hidden_size // config.head_dim
        wvh: TensorType[HEAD, HEAD_DIM, HIDDEN_DIM] = self.v_proj.weight.view(
            (num_heads, config.head_dim, config.hidden_size)
        )
        woh: TensorType[HEAD, HIDDEN_DIM, HEAD_DIM] = self.o_proj.weight.T.view(
            (num_heads, config.head_dim, config.hidden_size)
        ).transpose(-1, -2)
        self.wvoh = woh @ wvh

        self.attn_observation_hook = ObservationHook()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states: TensorType[BATCH, HEAD, SEQUENCE, HIDDEN_DIM] = torch.einsum(
            "bsd,hed->bhse", hidden_states, self.wvoh
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        self.config._attn_implementation = "eager"
        if self.config._attn_implementation != "eager":
            raise NotImplementedError

        weighted_value: TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM]
        weighted_value, attn_weights, attn_scores = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        self.attn_observation_hook(
            attn_scores=attn_scores,
            weighted_value=weighted_value,
        )
        attn_output: TensorType[BATCH, SEQUENCE, HIDDEN_DIM] = (
            weighted_value.sum(dim=-2).permute(0, 2, 1, 3)
        ).sum(dim=2)
        return attn_output, attn_weights


class EQLlamaDecoderLayer(nn.Module):
    def __init__(
        self, config: LlamaConfig, layer_idx: int, decoder_layer: LlamaDecoderLayer
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = EQLlamaAttention(
            config=config,
            layer_idx=layer_idx,
            attn=decoder_layer.self_attn,
            rms_norm=decoder_layer.input_layernorm,
        )

        self.mlp = decoder_layer.mlp
        self.input_layernorm = EQLlamaRMSNorm(eps=config.rms_norm_eps)
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class EQLlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class EQLlamaModel(EQLlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def wte(self):
        return self.embed_tokens


class EQLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = EQLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None = None,
        *model_args,
        **kwargs,
    ) -> "EQLlamaForCausalLM":
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.config._attn_implementation = "eager"

        model.model.layers = nn.ModuleList(
            [
                EQLlamaDecoderLayer(
                    model.config, layer_idx=layer_idx, decoder_layer=decoder_layer
                )
                for layer_idx, decoder_layer in enumerate(model.model.layers)
            ]
        )

        return model
    
    @property
    def transformer(self):
        return self.model
    

