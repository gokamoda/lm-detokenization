from argparse import ArgumentParser

import torch
from torchtyping import TensorType
from transformers import AutoModelForCausalLM

from utils.ln import get_var_matrix
from utils.mytorchtyping import HIDDEN_DIM, POS, VOCAB
from visualize import (
    l0_te_main,
    l0_tee_main,
    l0_tp_main,
    l0_tp_tpp_main,
    l0_tpp_main,
    l0_tpp_undertrained_main,
    var_wpe_main,
    var_wte_main,
)

scripts = {
    "l0_tp": {
        "function": l0_tp_main,
        "required_args": ["heads"],
    },
    "l0_tpp": {
        "function": l0_tpp_main,
        "required_args": ["heads", "pos_i"],
    },
    "l0_tp_tpp": {
        "function": l0_tp_tpp_main,
        "required_args": ["heads", "pos_i"],
    },
    "l0_tpp_undertrained": {
        "function": l0_tpp_undertrained_main,
        "required_args": ["heads"],
    },
    "l0_tee": {
        "function": l0_tee_main,
        "required_args": ["heads", "n_samples"],
    },
    "l0_te": {
        "function": l0_te_main,
        "required_args": ["heads"],
    },
    "var_wpe": {
        "function": var_wpe_main,
        "required_args": [],
    },
    "var_wte": {
        "function": var_wte_main,
        "required_args": [],
    },
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="self", choices=scripts.keys())
    parser.add_argument("--heads", type=int, nargs="+", default=[0])
    parser.add_argument("--pos-i", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    if args.mode in scripts:
        script = scripts[args.mode]
        for arg in script["required_args"]:
            if not hasattr(args, arg):
                raise ValueError(f"Missing required argument: {arg}")
        kwargs = {arg: getattr(args, arg) for arg in script["required_args"]}

    # prepare data
    model_name = "gpt2"
    wpe: TensorType[POS, HIDDEN_DIM] = (
        AutoModelForCausalLM.from_pretrained(model_name)
        .transformer.wpe.weight.detach()
        .cpu()
    )
    wte: TensorType[VOCAB, HIDDEN_DIM] = (
        AutoModelForCausalLM.from_pretrained(model_name)
        .transformer.wte.weight.detach()
        .cpu()
    )
    var_matrix = torch.sqrt(get_var_matrix(wpe, wte) + 1e-5).to(torch.float16)

    kwargs["wpe"] = wpe
    kwargs["wte"] = wte
    kwargs["var_matrix"] = var_matrix
    kwargs["model_name"] = model_name

    script["function"](**kwargs)
