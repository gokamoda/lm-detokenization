from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchtyping import TensorType
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from eqmodels import EQGPT2LMHeadModel
from eqmodels.gpt2 import compute_compare_score
from utils.mytorchtyping import HEAD, HIDDEN_DIM, POS, VOCAB

from .subplots import get_fig_axes_nrows2, plot_lineplot_ln, plot_lineplot_noln

plt.rc("font", size=80)
plt.rcParams["axes.xmargin"] = 0.001

linewidth = 10
save_dir = Path("figures/l0_tpp")
save_dir.mkdir(exist_ok=True, parents=True)


def compare_score_vis(
    wpe: TensorType[POS, HIDDEN_DIM],
    model_name: str,
    head: int,
    var_matrix: TensorType[POS, VOCAB],
    pos_i: int,
) -> None:
    fig, axes = get_fig_axes_nrows2()

    compare_score: TensorType[1, HEAD, POS, POS] = compute_compare_score(
        i=wpe.unsqueeze(0),
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu(),
    )

    # Without LN
    plot_lineplot_noln(
        axes[0], compare_score[0, head, pos_i, :pos_i], linewidth=linewidth
    )

    var_mean_i = var_matrix[pos_i].mean(dim=-1)
    var_max_i = var_matrix[pos_i].max(dim=-1).values
    var_min_i = var_matrix[pos_i].min(dim=-1).values

    for var_i in (var_mean_i, var_max_i, var_min_i):
        # divide by var_i
        compare_score_i: TensorType[pos_i] = (
            compare_score[0, head, pos_i, : pos_i + 1] / var_i
        )

        # divide by var_j
        var_j: TensorType[pos_i, VOCAB] = var_matrix[: pos_i + 1]
        compare_score_i: TensorType[pos_i, VOCAB] = (
            compare_score_i.unsqueeze(-1).expand(-1, var_matrix.shape[1]) / var_j
        )

        plot_lineplot_ln(
            axes[1],
            compare_score_i,
        )

    # Adjustments
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # Titles
    axes[0].set_title(
        r"$\mathbf{p}_{"
        + str(pos_i)
        + r"}\mathbf{W}_{"
        + str(head)
        + "}^{QK}\\mathbf{p}_j^\\top$",
        y=0.80,
        x=0.006,
        loc="left",
    )
    axes[1].set_title(
        "$T^{\\text{pp}}_{" + str(pos_i) + ", j, " + str(head) + "}$",
        y=0.80,
        x=0.006,
        loc="left",
    )

    # Labels
    axes[1].set_xlabel("Past token position ($j$)")
    axes[1].set_ylabel("")

    # Ticks
    axes[0].xaxis.set_tick_params(
        width=linewidth, length=linewidth * 2, direction="out"
    )
    axes[1].xaxis.set_tick_params(
        width=linewidth, length=linewidth * 2, direction="out"
    )
    axes[0].yaxis.set_tick_params(
        width=linewidth, length=linewidth * 2, direction="out"
    )
    axes[1].yaxis.set_tick_params(
        width=linewidth, length=linewidth * 2, direction="out"
    )

    # Save
    fig.savefig(
        save_dir.joinpath(f"l0tpp_head-{head}_posi-{pos_i}.pdf"), bbox_inches="tight"
    )


def compare_score_vis_all_heads(
    wpe: TensorType[POS, HIDDEN_DIM],
    model_name: str,
    var_matrix: TensorType[POS, VOCAB],
) -> None:
    print("Visualizing TPP for all heads")

    compare_score: TensorType[1, HEAD, POS, POS] = compute_compare_score(
        i=wpe.unsqueeze(0),
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu(),
    )

    # Prepare figure
    num_heads = compare_score.shape[1]
    if num_heads == 12:
        fig, axes = plt.subplots(nrows=12, ncols=3, figsize=(80, 100), sharex="col")
    else:
        raise NotImplementedError

    # Plot
    for head in tqdm(range(num_heads), desc="Plotting heads", leave=False):
        for pos_i_idx, pos_i in tqdm(
            enumerate([50, 500, 1000]), desc="Plotting lengths", total=3, leave=False
        ):
            ax = axes[head, pos_i_idx]

            var_mean_i = var_matrix[pos_i].mean(dim=-1)
            var_max_i = var_matrix[pos_i].max(dim=-1).values
            var_min_i = var_matrix[pos_i].min(dim=-1).values

            for var_i in (var_mean_i, var_max_i, var_min_i):
                # divide by var_i
                compare_score_i: TensorType[pos_i] = (
                    compare_score[0, head, pos_i, : pos_i + 1] / var_i
                )

                # divide by var_j
                var_j: TensorType[pos_i, VOCAB] = var_matrix[: pos_i + 1]
                compare_score_i: TensorType[pos_i, VOCAB] = (
                    compare_score_i.unsqueeze(-1).expand(-1, var_matrix.shape[1])
                    / var_j
                )

                plot_lineplot_ln(ax, compare_score_i)

            # Adjustments

            # Titles
            ax.set_title(
                "$T_{" + str(pos_i) + ", j, " + str(head) + "}^{pp}$",
                y=0.75,
                x=0.01,
                loc="left",
            )

            # Labels
            ax.set_ylabel("")
            if head == 11:
                ax.set_xlabel(
                    "Past token position ($0\\leq j \\leq" + str(pos_i) + "$)"
                )

            # Ticks
            ax.xaxis.set_tick_params(
                width=linewidth, length=linewidth * 2, direction="out"
            )

    # Save
    fig.savefig(save_dir.joinpath("l0tpp_head-all.pdf"), bbox_inches="tight")


def main(
    model_name: str,
    wpe: TensorType[POS, HIDDEN_DIM],
    var_matrix: TensorType[POS, VOCAB],
    head: int,
    **kwargs,
) -> None:
    if head == -1:
        compare_score_vis_all_heads(
            wpe=wpe,
            model_name=model_name,
            var_matrix=var_matrix,
        )
    else:
        compare_score_vis(
            wpe=wpe,
            model_name=model_name,
            var_matrix=var_matrix,
            head=head,
            pos_i=kwargs["pos_i"],
        )
