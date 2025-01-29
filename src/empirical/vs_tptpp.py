from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
from datasets import load_dataset
from torchtyping import TensorType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer

from eqmodels import EQGPT2LMHeadModel
from eqmodels.gpt2 import compute_compare_score, compute_self_score
from utils.ln import get_var_matrix, ln_pos
from utils.mylogger import init_logging
from utils.mytorchtyping import HEAD, HIDDEN_DIM, POS, VOCAB
from utils.path import WORK_DIR
from visualize.subplots import plot_lineplot_ln

from .utils import get_data

plt.rcParams["axes.xmargin"] = 0.00

LOG_PATH = "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)


linewidth = 10
save_dir = WORK_DIR.joinpath("attn_l0")
save_dir.mkdir(exist_ok=True, parents=True)

num_instances = 100


def main():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = model.to(device)
    model.eval()

    for i, row in tqdm(enumerate(get_data())):
        if i == num_instances:
            break

        save_path = save_dir.joinpath(f"{i}.pt")
        if save_path.exists():
            # print(f"Skipping {i}")
            # continue
            pass

        # prompt = "<|endoftext|>" + row["text"]
        prompt = row["text"]
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)

        if input_ids["input_ids"].shape[1] > 1024:
            input_ids["input_ids"] = input_ids["input_ids"][:, :1024]
            input_ids["attention_mask"] = input_ids["attention_mask"][:, :1024]

        with torch.no_grad():
            output = model.generate(
                **input_ids,
                num_return_sequences=1,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_attentions=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            attn = output.attentions[0][0][0].cpu().detach()
        attn = attn.to(torch.float16)
        torch.save(attn, save_path)


def vis_empirical(axes, tokenizer, target_length=500, head=0, begin_offsets=0):
    logger.info("Collecting empirical attns")
    attns = []
    for i, row in tqdm(enumerate(get_data())):
        if i == num_instances:
            break
        prompt = "<|endoftext|>" + row["text"]
        if len(tokenizer.encode(prompt)) < target_length:
            continue

        attn = torch.load(save_dir.joinpath(f"{i}.pt"), weights_only=True)

        if attn.shape[1] < target_length:
            continue
        attns.append(attn[head, target_length, : target_length + 1])
    attns: TensorType[num_instances, target_length] = torch.stack(attns)

    logger.info("Plotting theoretical approximations")
    for i, begin_offset in enumerate([0, begin_offsets]):
        plot_lineplot_ln(axes[i], attns.T)
        axes[i].xaxis.set_tick_params(
            width=linewidth, length=linewidth * 2, direction="out"
        )
        axes[i].xaxis.set_tick_params(
            width=linewidth, length=linewidth * 2, direction="out"
        )
        axes[i].set_xlim(left=begin_offset, right=target_length)
        axes[i].set_ylabel(None)
        axes[i].set_xlabel("Past token position ($j$)")
        axes[i].set_title(
            "$\\alpha_{" + str(target_length) + ", j, " + str(head) + "}$",
            y=0.80,
            x=0.01,
            loc="left",
        )

        if head == 7:
            top = 0.5
            axes[i].set_ylim(bottom=-top * 0.01, top=top)
        else:
            top = 1.0
            axes[i].set_ylim(bottom=-top * 0.01, top=top)


def vis_theoretical(axes, target_length=500, head=0, begin_offset=0):
    # add te + tee plot
    model_name = "gpt2"

    wpe = (
        AutoModelForCausalLM.from_pretrained(model_name)
        .transformer.wpe.weight.detach()
        .cpu()
    )
    wte = (
        AutoModelForCausalLM.from_pretrained(model_name)
        .transformer.wte.weight.detach()
        .cpu()
    )
    var_matrix: TensorType[POS, VOCAB] = get_var_matrix(wpe=wpe, wte=wte)
    var_matrix = torch.sqrt(var_matrix + 1e-5)
    var_matrix = var_matrix.to(torch.float16)

    self_score: TensorType[1, HEAD, POS, POS] = compute_self_score(
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.bqwkh.detach()
        .cpu(),
    )
    self_score: TensorType[1, HEAD, POS, VOCAB] = ln_pos(
        x=self_score, var_matrix=var_matrix
    )

    compare_score: TensorType[1, HEAD, POS, POS] = compute_compare_score(
        i=wpe.unsqueeze(0),
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu(),
    )

    var_mean_i = var_matrix[target_length].mean(dim=-1)

    # divide by var_i
    compare_score_i: TensorType[target_length] = (
        compare_score[0, head, target_length, : target_length + 1] / var_mean_i
    )

    # divide by var_j
    var_j: TensorType[target_length, VOCAB] = var_matrix[: target_length + 1]
    compare_score_i: TensorType[target_length, VOCAB] = (
        compare_score_i.unsqueeze(-1).expand(-1, var_matrix.shape[1]) / var_j
    )

    scores_i = compare_score_i + self_score[0, head, : target_length + 1]

    # after softmax
    hidden_dim = wpe.shape[1]
    num_heads = self_score.shape[1]
    scores_i = scores_i.mean(dim=-1) / ((hidden_dim / num_heads) ** 0.5)
    scores_i = torch.nn.functional.softmax(scores_i, dim=-1)

    axes[0].plot(scores_i, color="red", linewidth=linewidth / 1.5)
    axes[0].set_xlim(left=0, right=target_length)
    axes[1].plot(
        scores_i, color="red", linewidth=linewidth / 1.5, marker="o", markersize=20
    )
    axes[1].set_xlim(left=begin_offset, right=target_length)


def vis():
    fig, axes = plt.subplots(figsize=(40, 20), nrows=2, ncols=2, sharex="col")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    heads = [1, 7]
    heads_str = "-".join(map(str, heads))

    target_length = 500
    save_path = Path(
        f"figures/empirical/empirical_vstptpp_head-{heads_str}_length-{target_length}.pdf"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for ax, head in enumerate(heads):
        vis_empirical(
            axes[ax],
            tokenizer=tokenizer,
            target_length=target_length,
            head=head,
            begin_offsets=485,
        )

        vis_theoretical(
            axes[ax], target_length=target_length, head=head, begin_offset=485
        )
        axes[ax][0].set_ylabel(f"Head {head}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
