import torch
from torch import nn
from torchtyping import TensorType
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from _transformers.models.llama.modeling_llama import (
    EQLlamaAttention,
    EQLlamaForCausalLM,
    EQLlamaRMSNorm,
    collapse_rmsnorm,
    repeat_kv,
    repeat_kv_weights,
)
from _transformers.utils.hooks import BatchAttentionObservationResult, Hook
from utils.mylogger import init_logging
from utils.mytorchtyping import BATCH, HEAD, HEAD_DIM, HIDDEN_DIM, SEQUENCE

LOG_PATH = "test.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)


def test_collapse_ln():
    hidden_dim = 5
    _hidden_states = torch.randn(1, 1, hidden_dim)

    # original
    rmsnorm = LlamaRMSNorm(hidden_size=hidden_dim, eps=1e-5)
    linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
    hidden_states_original = rmsnorm(_hidden_states)
    hidden_states_original = linear(hidden_states_original)

    # redefined
    linear_new = collapse_rmsnorm(linear, rmsnorm)
    rmsnorm = EQLlamaRMSNorm(eps=1e-5)
    hidden_states_redefined = rmsnorm(_hidden_states)
    hidden_states_redefined = linear_new(hidden_states_redefined)

    assert torch.allclose(
        hidden_states_original, hidden_states_redefined, atol=1e-5, rtol=1e-5
    )


def test_wov():
    hidden_size = 10
    head_dim = 5

    _hidden_states = torch.randn(1, 3, hidden_size)
    input_shape = _hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    v_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
    o_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
    value_states: TensorType[BATCH, HEAD, SEQUENCE, HEAD_DIM] = v_proj(
        _hidden_states
    ).view(hidden_shape)
    value_states: TensorType[BATCH, SEQUENCE, HIDDEN_DIM] = value_states.reshape(
        *input_shape, -1
    ).contiguous()
    value_states: TensorType[BATCH, SEQUENCE, HIDDEN_DIM] = o_proj(value_states)

    num_heads = hidden_size // head_dim
    wvh: TensorType[HEAD, HEAD_DIM, HIDDEN_DIM] = v_proj.weight.view(
        (num_heads, head_dim, hidden_size)
    )
    woh: TensorType[HEAD, HIDDEN_DIM, HEAD_DIM] = o_proj.weight.T.view(
        (num_heads, head_dim, hidden_size)
    ).transpose(-1, -2)
    wvo: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM] = woh @ wvh
    value_states_redefined = torch.einsum("bsd,hed->bhse", _hidden_states, wvo)
    value_states_redefined = value_states_redefined.sum(dim=1)

    assert torch.allclose(
        value_states,
        value_states_redefined,
        atol=1e-5,
        rtol=1e-5,
    )


def test_attn():
    config = LlamaConfig()
    logger.info(config)
    rotary_emb = LlamaRotaryEmbedding(config)
    layer_idx = 1

    # original
    rmsnorm_original = LlamaRMSNorm(
        hidden_size=config.hidden_size, eps=config.rms_norm_eps
    )
    rmsnorm_original.weight = nn.Parameter(torch.randn(config.hidden_size))
    attn_original = LlamaAttention(config, layer_idx=layer_idx)
    rmsnorm_original.eval()
    attn_original.eval()

    # redefined
    ln_redefined = EQLlamaRMSNorm(eps=config.rms_norm_eps)
    attn_redefined = EQLlamaAttention(
        config, rms_norm=rmsnorm_original, attn=attn_original, layer_idx=layer_idx
    )
    ln_redefined.eval()
    attn_redefined.eval()

    # test
    hidden_states = torch.randn(1, 10, config.hidden_size)
    position_ids = torch.arange(10).unsqueeze(0)
    position_embeddings = rotary_emb(hidden_states, position_ids)
    attention_mask = torch.ones(1, 1, 10, 10)

    assert torch.allclose(
        attn_original(
            rmsnorm_original(hidden_states), position_embeddings, attention_mask
        )[0],
        attn_redefined(
            ln_redefined(hidden_states), position_embeddings, attention_mask
        )[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_repeat_kv():
    batch_size = 1
    num_heads_pre = 3
    sequence_length = 5
    dim = 7
    groups = 2

    key = torch.randn(batch_size, num_heads_pre, sequence_length, dim)
    logger.info(key.shape)
    key_states = repeat_kv(key, groups)
    logger.info(key_states.shape)

    assert torch.allclose(
        key_states[0, 0],
        key[0, 0],
        atol=1e-5,
        rtol=1e-5,
    ), f"key_states[0,0]: {key_states[0, 0]} key[0,0]: {key[0, 0]}"
    assert torch.allclose(
        key_states[0, 0],
        key_states[0, 1],
        atol=1e-5,
        rtol=1e-5,
    ), (
        f"key_states[0,0]: {key_states[0, 0]} key[0,num_heads_pre]: {key_states[0, num_heads_pre]}"
    )


def test_repeat_kv_weights():
    hidden_size = 20
    num_attention_heads = 4
    num_key_value_heads = 2

    proj = nn.Linear(
        in_features=hidden_size,
        out_features=hidden_size // num_attention_heads * num_key_value_heads,
        bias=False,
    )
    proj_new= repeat_kv_weights(
        proj=proj,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads
    )

    assert proj_new.weight.shape == (hidden_size, hidden_size)

    head_dim = hidden_size // num_attention_heads
    assert torch.allclose(
        proj.weight[0:head_dim],
        proj_new.weight[0:head_dim],
        atol=1e-5,
        rtol=1e-5,
    ), (
        f"key_proj.weight[0:head_dim]: {proj.weight[0:head_dim]} key_proj_new.weight[0:head_dim]: {proj_new.weight[0:head_dim]}"
    )

    assert torch.allclose(
        proj_new.weight[0:head_dim],
        proj_new.weight[head_dim : 2 * head_dim],
        atol=1e-5,
        rtol=1e-5,
    ), (
        f"key_proj_new.weight[0:head_dim]: {proj_new.weight[0:head_dim]} key_proj_new.weight[head_dim:2*head_dim]: {proj_new.weight[head_dim : 2 * head_dim]}"
    )


def test_causal_lm():
    model_name = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = " Tokyo is the capital of"
    inputs = tokenizer(prompt, return_tensors="pt")

    # original
    causal_lm_original = LlamaForCausalLM.from_pretrained(model_name).eval()
    logger.info(causal_lm_original.config)
    logger.info(causal_lm_original)

    # redefined
    causal_lm_redefined = EQLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).eval()

    generation_args = {
        "max_new_tokens": 1,
        "do_sample": False,
        "use_cache": False,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }

    with torch.no_grad():
        original_outputs = causal_lm_original.generate(
            **inputs, **generation_args, output_logits=True
        )
        redefined_outputs = causal_lm_redefined.generate(
            **inputs, **generation_args, output_logits=True
        )

    logger.info(original_outputs.logits[0][-1])
    logger.info(redefined_outputs.logits[0][-1])

    assert torch.allclose(
        original_outputs.logits[0][-1],
        redefined_outputs.logits[0][-1],
        atol=1e-4,
        rtol=1e-4,
    )


def test_attn_hook():
    model_name = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = " Tokyo is the capital of"
    inputs = tokenizer(prompt, return_tensors="pt")

    # redefined
    model = EQLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).eval()
    logger.info(model)

    hooks: list[Hook] = [
        Hook(
            model.model.layers[i].self_attn.attn_observation_hook,
            result_class=BatchAttentionObservationResult,
        )
        for i in range(model.config.num_hidden_layers)
    ]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    for hook in hooks:
        hook.remove()

    logger.info(outputs.attentions)
    logger.info(hooks[0].result.attn_scores.shape)
