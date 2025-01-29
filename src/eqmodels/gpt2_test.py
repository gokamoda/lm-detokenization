import torch
from torch.nn import LayerNorm
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP,
    GPT2Attention,
    GPT2Block,
    GPT2Config,
    GPT2LMHeadModel,
)

from .gpt2 import (
    EQGPT2MLP,
    EQGPT2Attention,
    EQGPT2Block,
    EQGPT2LayerNorm,
    EQGPT2LMHeadModel,
)


def test_attn():
    config = GPT2Config()

    # original
    ln_original = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    attn_original = GPT2Attention(config)
    ln_original.eval()
    attn_original.eval()

    # redefined
    ln_redefined = EQGPT2LayerNorm(eps=config.layer_norm_epsilon)
    attn_redefined = EQGPT2Attention(config, ln=ln_original, attn=attn_original)
    ln_redefined.eval()
    attn_redefined.eval()

    # test
    hidden_states = torch.randn(1, 10, config.n_embd)

    assert torch.allclose(
        attn_original(ln_original(hidden_states))[0],
        attn_redefined(ln_redefined(hidden_states))[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_mlp():
    config = GPT2Config()

    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd

    # original
    ln_original = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    mlp_original = GPT2MLP(intermediate_size=inner_dim, config=config)
    ln_original.eval()
    mlp_original.eval()

    # redefined
    ln_redefined = EQGPT2LayerNorm(eps=config.layer_norm_epsilon)
    mlp_redefined = EQGPT2MLP(
        intermediate_size=inner_dim,
        config=config,
        ln=ln_original,
        mlp=mlp_original,
    )
    ln_redefined.eval()
    mlp_redefined.eval()

    # test
    hidden_states = torch.randn(1, 10, config.n_embd)

    assert torch.allclose(
        mlp_original(ln_original(hidden_states))[0],
        mlp_redefined(ln_redefined(hidden_states))[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_block():
    config = GPT2Config()

    # original
    block_original = GPT2Block(config)
    block_original.eval()

    # redefined
    block_redefined = EQGPT2Block(config, gpt2_block=block_original)
    block_redefined.eval()

    # test
    hidden_states = torch.randn(1, 10, config.n_embd)

    assert torch.allclose(
        block_original(hidden_states)[0],
        block_redefined(hidden_states)[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_gpt2lmhead():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "<|endoftext|> Tokyo is the capital of"
    inputs = tokenizer(prompt, return_tensors="pt")

    # original
    lmhead_original = GPT2LMHeadModel.from_pretrained("gpt2")
    lmhead_original.eval()

    # redefined
    lmhead_redefined = EQGPT2LMHeadModel.from_pretrained(
        pretrained_model_name_or_path="gpt2"
    )
    lmhead_redefined.eval()

    generation_args = {
        "max_new_tokens": 1,
        "do_sample": False,
        "use_cache": False,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }

    original_outputs = lmhead_original.generate(
        **inputs, **generation_args, output_logits=True
    )

    redefined_outputs = lmhead_redefined.generate(
        **inputs, **generation_args, output_logits=True
    )

    print(tokenizer.decode(original_outputs.sequences[0]))
    print(tokenizer.decode(redefined_outputs.sequences[0]))

    assert torch.allclose(
        original_outputs.logits[0][-1],
        redefined_outputs.logits[0][-1],
        atol=1e-5,
        rtol=1e-5,
    )


def test_gpt2_last_hidden():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = EQGPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    lm_head = model.lm_head

    prompt = "Tokyo is the capital of"
    inputs = tokenizer(prompt, return_tensors="pt")

    generation_args = {
        "max_new_tokens": 1,
        "do_sample": False,
        "use_cache": False,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_args,
            output_hidden_states=True,
            output_logits=True,
        )
        logits = lm_head(outputs.hidden_states[0][-1])

    assert torch.allclose(
        outputs.logits[0][-1],
        logits[0, -1],
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == "__main__":
    test_gpt2_last_hidden()
