from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pl
import seaborn as sns
import torch
from torchtyping import TensorType
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from eqmodels import EQGPT2LMHeadModel
from eqmodels.gpt2 import compute_compare_score
from utils.mytorchtyping import HEAD, HIDDEN_DIM, POS, VOCAB

plt.rc("font", size=70)
plt.rcParams["axes.xmargin"] = 0.001

linewidth = 10
save_dir = Path("figures/l0_tpp")
save_dir.mkdir(exist_ok=True, parents=True)


def main(
    wpe: TensorType[POS, HIDDEN_DIM],
    model_name: str,
    heads: list[int],
    **kwargs,
) -> None:
    del kwargs
    head = heads[0]
    fig, ax = plt.subplots(1, 1, figsize=(32, 10))

    compare_score: TensorType[1, HEAD, POS, POS] = compute_compare_score(
        i=wpe.unsqueeze(0),
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu(),
    )

    n = 5

    dfs = []
    for i in range(n):
        length = wpe.shape[0] - i
        compare_score_i = compare_score[0, head, length - 1, :length]

        df = pl.DataFrame(
            {
                "i": length - 1,
                "j": np.arange(length),
                "value": compare_score_i.numpy(),
            }
        )

        dfs.append(df)

    df = pl.concat(dfs)

    sns.lineplot(
        data=df,
        x="j",
        y="value",
        hue="i",
        ax=ax,
        linewidth=linewidth,
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(
        "$\\mathbf{p}_i\\mathbf{W}_0^{QK}\\mathbf{p}_j^\\top$",
        y=0.80,
        loc="left",
        x=0.01,
    )
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    ax.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    ax.set_xlabel("Past token position ($j$)")
    fig.savefig(save_dir.joinpath(f"undertrained_head-{head}.pdf"), bbox_inches="tight")
    print("saved at:", save_dir.joinpath(f"undertrained_head-{head}.pdf"))
