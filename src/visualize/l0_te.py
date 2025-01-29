from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
from scipy.stats import spearmanr
from torchtyping import TensorType
from tqdm import tqdm

from eqmodels import EQGPT2LMHeadModel
from eqmodels.gpt2 import compute_compare_score, compute_self_score
from utils.ln import ln_pos
from utils.mytorchtyping import HEAD, HIDDEN_DIM, POS, VOCAB

from .subplots import get_fig_axes_nrows2, plot_lineplot_ln, plot_lineplot_noln

plt.rc("font", size=80)
plt.rcParams["axes.xmargin"] = 0.001

linewidth = 10
save_dir = Path("figures/l0_te")
save_dir.mkdir(exist_ok=True, parents=True)


def self_score_vis_all_heads(
    wte: TensorType[VOCAB, HIDDEN_DIM],
    model_name: str,
    max_pos: int,
    var_matrix: TensorType[VOCAB, VOCAB],
    counter: Counter[int],
):
    self_score = compute_self_score(
        j=wte.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.bqwkh.detach()
        .cpu(),
    )

    n_heads = self_score.shape[1]
    if n_heads == 12:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=4,
            figsize=(80, 35),
            sharex=True,
        )
    else:
        raise NotImplementedError

    self_score_ln: TensorType[1, HEAD, VOCAB, POS] = self_score.unsqueeze(-1).expand(
        -1, -1, -1, max_pos
    ) / var_matrix.T.unsqueeze(0).unsqueeze(0)

    for head in tqdm(range(n_heads)):
        ax = axes[head // 4, head % 4]

        df = pl.DataFrame(
            {
                "idx": list(range(self_score.shape[-1])),
                "score": self_score[0, head].cpu().numpy(),
                "score_ln": self_score_ln[0, head].mean(dim=-1).cpu().numpy(),
            }
        )

        df = df.with_columns(
            pl.col("idx")
            .map_elements(lambda x: counter[x], return_dtype=pl.Int64)
            .alias("count"),
        ).filter(pl.col("count") >= 0)

        ax.scatter(data=df, x="count", y="score_ln", alpha=0.1, marker=".")
        ax.set_xscale("log")
        ax.set_title(
            "$T_{j, " + str(head) + "}^{e}$",
            y=0.82,
            x=0.006,
            loc="left",
        )
        ax.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
        ax.axhline(0, color="black", linewidth=linewidth / 3, linestyle="--")

        print("------head", head)
        print("count vs score", spearmanr(df["count"], df["score"]))
        print("count vs score_ln", spearmanr(df["count"], df["score_ln"]))
        print("score_ln vs score", spearmanr(df["score_ln"], df["score"]))

    fig.supxlabel("Token ($j$) Count", y=-0.0)

    fig.savefig(save_dir.joinpath("l0te_head-all.png"), bbox_inches="tight")


def self_score_vis(
    wte: TensorType[VOCAB, HIDDEN_DIM],
    model_name: str,
    heads: list[int],
    max_pos: int,
    var_matrix: TensorType[VOCAB, VOCAB],
    counter: Counter[int],
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(heads),
        figsize=(40, 20),
        sharex=True,
    )

    self_score = compute_self_score(
        j=wte.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.bqwkh.detach()
        .cpu(),
    )

    self_score_ln: TensorType[1, HEAD, VOCAB, POS] = self_score.unsqueeze(-1).expand(
        -1, -1, -1, max_pos
    ) / var_matrix.T.unsqueeze(0).unsqueeze(0)

    for h, head in enumerate(heads):
        ax = axes[h, 0]
        df = pl.DataFrame(
            {
                "idx": list(range(self_score.shape[-1])),
                "score": self_score[0, head].cpu().numpy(),
                "score_ln": self_score_ln[0, head].mean(dim=-1).cpu().numpy(),
            }
        )
        df = df.with_columns(
            pl.col("idx")
            .map_elements(lambda x: counter[x], return_dtype=pl.Int64)
            .alias("count"),
        ).filter(pl.col("count") >= 0)

        ax.scatter(data=df, x="count", y="score", alpha=0.1, marker=".")
        ax.set_xscale("log")
        ax.set_title(
            "$\\mathbf{b}_{" + str(head) + "}^{QK}\\mathbf{e}_{\\text{ID}_j}^\\top$",
            y=0.80,
            x=0.006,
            loc="left",
        )
        ax.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
        ax.axhline(0, color="black", linewidth=linewidth / 3, linestyle="--")

        ax = axes[h, 1]
        ax.scatter(data=df, x="count", y="score_ln", alpha=0.1, marker=".")
        ax.set_xscale("log")
        ax.set_title(
            "$T_{j, " + str(head) + "}^{e}$",
            y=0.80,
            x=0.006,
            loc="left",
        )
        ax.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
        ax.axhline(0, color="black", linewidth=linewidth / 3, linestyle="--")

        print("------head", head)
        print("count vs score", spearmanr(df["count"], df["score"]))
        print("count vs score_ln", spearmanr(df["count"], df["score_ln"]))
        print("score_ln vs score", spearmanr(df["score_ln"], df["score"]))

        if h == 1:
            axes[h, 0].set_xlabel("Token ($j$) Count")
            axes[h, 1].set_xlabel("Token ($j$) Count")

        axes[h, 0].set_ylabel(f"Head {head}")

    head_str = "-".join(map(str, heads))
    fig.savefig(save_dir.joinpath(f"l0te_head-{head_str}.png"), bbox_inches="tight")
    print("saved at:", save_dir.joinpath(f"l0te_head-{head_str}.png"))


def main(
    model_name: str,
    wpe: TensorType[POS, HIDDEN_DIM],
    wte: TensorType[VOCAB, HIDDEN_DIM],
    var_matrix: TensorType[POS, VOCAB],
    heads: list[int],
    **kwargs,
) -> None:
    counter = torch.load("freqs/token/openwebtext_gpt2.pt", weights_only=True)

    if heads == [-1]:
        self_score_vis_all_heads(
            wte=wte,
            model_name=model_name,
            max_pos=wpe.shape[0],
            var_matrix=var_matrix,
            counter=counter,
        )
    else:
        self_score_vis(
            wte=wte,
            model_name=model_name,
            heads=heads,
            max_pos=wpe.shape[0],
            var_matrix=var_matrix,
            counter=counter,
        )
