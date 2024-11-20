from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM

from eqmodels import EQGPT2LMHeadModel
from eqmodels.gpt2 import compute_compare_score, compute_self_score

plt.rc("font", size=70)
plt.rcParams["axes.xmargin"] = 0.01


def main(**kwargs):
    wte: torch.Tensor = AutoModelForCausalLM.from_pretrained(
        "gpt2"
    ).transformer.wte.weight.detach()
    var = wte.var(dim=-1, unbiased=False)

    counts = torch.load("freqs/token/openwebtext_gpt2.pt")

    df = pl.DataFrame({"idx": list(range(var.shape[0])), "variance": var.numpy()})
    df = df.with_columns(
        pl.col("idx")
        .map_elements(lambda x: counts.get(x, -1), return_dtype=pl.Int64)
        .alias("count"),
    )

    df = df.filter(pl.col("count") >= 0)

    fig, ax = plt.subplots(1, 1, figsize=(40, 20))

    ax.plot(df["count"], df["variance"], ".", markersize=5)
    ax.set_xscale("log")

    linewidth = 10
    ax.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    ax.yaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    ax.set_xlabel("Token Count")
    ax.set_title("Var($e_\\text{ID}$)", y=0.9, loc="left", x=0.02)

    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True, parents=True)

    fig.savefig(save_dir.joinpath("wte_variance.png"), bbox_inches="tight")
    # fig.savefig("../wte_variance.pdf", bbox_inches="tight")
    # fig.savefig("wte_variance.png", bbox_inches="tight")
