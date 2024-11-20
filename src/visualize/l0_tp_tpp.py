from pathlib import Path

import matplotlib.pyplot as plt
import torch
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
save_dir = Path("figures/l0_tp_tpp")
save_dir.mkdir(exist_ok=True, parents=True)


def both_score_vis(wpe, model_name, head, var_matrix, pos_i) -> None:
    fig, axes = get_fig_axes_nrows2()

    compare_score: TensorType[1, HEAD, POS, POS] = compute_compare_score(
        i=wpe.unsqueeze(0),
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu(),
    )

    self_score: TensorType[1, HEAD, POS] = compute_self_score(
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.bqwkh.detach()
        .cpu(),
    )
    self_score: TensorType[1, HEAD, POS, VOCAB] = ln_pos(
        x=self_score, var_matrix=var_matrix
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

        scores_i = compare_score_i + self_score[0, head, : pos_i + 1]

        # before softmax
        plot_lineplot_ln(axes[0], scores_i)

        # after softmax
        hidden_dim = wpe.shape[1]
        num_heads = self_score.shape[1]
        scores_i = scores_i.mean(dim=-1) / ((hidden_dim / num_heads) ** 0.5)
        scores_i = torch.nn.functional.softmax(scores_i, dim=-1)
        axes[1].plot(scores_i, linewidth=linewidth)

    # Adjustments

    # Titles
    axes[0].set_title(
        "$T^{\\text{pp}}_{"
        + str(pos_i)
        + ", j, "
        + str(head)
        + "}"
        + "+ T^{\\text{p}}_{j, "
        + str(head)
        + "}$",
        y=0.80,
        x=0.006,
        loc="left",
    )
    axes[1].set_title(
        "$\\text{softmax}_j\\frac{T^{\\text{pp}}_{"
        + str(pos_i)
        + ", j, "
        + str(head)
        + "}"
        + " + T^{\\text{p}}_{j, "
        + str(head)
        + "}}{\\sqrt{d'}}$",
        y=0.7,
        x=0.006,
        fontdict={"fontsize": 100},
        loc="left",
    )

    # Labels
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Past token position ($j$)")

    # Ticks
    axes[0].xaxis.set_tick_params(
        width=linewidth, length=linewidth * 2, direction="out"
    )
    axes[0].yaxis.set_tick_params(
        width=linewidth, length=linewidth * 2, direction="out"
    )
    axes[1].xaxis.set_tick_params(
        width=linewidth, length=linewidth * 2, direction="out"
    )
    axes[1].yaxis.set_tick_params(
        width=linewidth, length=linewidth * 2, direction="out"
    )

    # Save
    fig.savefig(
        save_dir.joinpath(f"l0tptpp_head-{head}_posi-{pos_i}.pdf"), bbox_inches="tight"
    )


def both_score_vis_all_heads(wpe, model_name, var_matrix):
    self_score = compute_self_score(
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.bqwkh.detach()
        .cpu(),
    )
    vocab_size = var_matrix.shape[-1]
    self_score: TensorType[1, HEAD, POS, VOCAB] = self_score.unsqueeze(-1).expand(
        -1, -1, -1, vocab_size
    ) / var_matrix.unsqueeze(0).unsqueeze(0)

    compare_score = compute_compare_score(
        i=wpe.unsqueeze(0),
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu(),
    )

    num_heads = self_score.shape[1]

    if num_heads == 12:
        fig, axes = plt.subplots(nrows=12, ncols=3, figsize=(80, 100), sharex="col")
    else:
        raise NotImplementedError

    for head in tqdm(range(num_heads)):
        for pos_i_idx, pos_i in enumerate([50, 500, 1000]):
            ax = axes[head, pos_i_idx]

            var_mean_i = var_matrix[pos_i].mean(dim=-1)
            var_max_i = var_matrix[pos_i].max(dim=-1).values
            var_min_i = var_matrix[pos_i].min(dim=-1).values

            for var_i in (var_mean_i, var_max_i, var_min_i):
                compare_score_i = (compare_score[0, head, pos_i] / var_i).unsqueeze(
                    -1
                ).expand(-1, var_matrix.shape[1]) / var_matrix

                compare_score_i = compare_score_i[: pos_i + 1]
                self_score_i = self_score[0, head, : pos_i + 1]

                scores_i = compare_score_i + self_score_i
                scores_i = (
                    scores_i.mean(dim=-1) / (wpe.shape[1] / self_score.shape[1]) ** 0.5
                )
                scores_i = torch.nn.functional.softmax(scores_i, dim=-1)

                ax.plot(scores_i, linewidth=linewidth)

                ax.xaxis.set_tick_params(
                    width=linewidth, length=linewidth * 2, direction="out"
                )
                ax.yaxis.set_tick_params(
                    width=linewidth, length=linewidth * 2, direction="out"
                )
                ax.set_ylabel("")
                ax.set_title(
                    "$\\text{softmax}_j\\frac{T_{"
                    + str(pos_i)
                    + ", j, "
                    + str(head)
                    + "}^{pp}+T_{j, "
                    + str(head)
                    + "}^{p}}{\\sqrt{d'}}$",
                    y=0.67,
                    loc="left",
                )

                if head == 11:
                    ax.set_xlabel(
                        "Past token position ($0\\leq j \\leq" + str(pos_i) + "$)"
                    )

    fig.savefig(save_dir.joinpath("l0tptpp_head-all.pdf"), bbox_inches="tight")


def main(
    model_name: str,
    wpe: TensorType[POS, HIDDEN_DIM],
    var_matrix: TensorType[POS, VOCAB],
    head: int,
    **kwargs,
) -> None:
    if head == -1:
        both_score_vis_all_heads(
            wpe=wpe,
            model_name=model_name,
            var_matrix=var_matrix,
        )
    else:
        both_score_vis(
            wpe=wpe,
            model_name=model_name,
            var_matrix=var_matrix,
            head=head,
            pos_i=kwargs["pos_i"],
        )
