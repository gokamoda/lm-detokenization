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


def both_score_vis(
    wpe: TensorType[POS, HIDDEN_DIM],
    model_name: str,
    heads: list[int],
    var_matrix: TensorType[POS, VOCAB],
    pos_i: int,
) -> None:
    fig, axes_ = get_fig_axes_nrows2()

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

    for head_iterator_idx, head in enumerate(heads):
        axes = axes_[head_iterator_idx]

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
        axes[0].annotate(
            "$T^{\\text{pp}}_{"
            + str(pos_i)
            + ", j, "
            + str(head)
            + "}"
            + "+ T^{\\text{p}}_{j, "
            + str(head)
            + "}$",
            xy=(0.006, 0.82),
            xycoords="axes fraction",
            fontsize=80,
        )
        axes[1].annotate(
            "$\\text{softmax}_j\\frac{T^{\\text{pp}}_{"
            + str(pos_i)
            + ", j, "
            + str(head)
            + "}"
            + " + T^{\\text{p}}_{j, "
            + str(head)
            + "}}{\\sqrt{d'}}$",
            xy=(0.006, 0.75),
            xycoords="axes fraction",
            fontsize=80,
        )

        if head == 7:
            bottom, top = axes[0].get_ylim()
            axes[0].set_ylim((bottom, top + 0.1 * (top - bottom)))

        # Labels
        if head_iterator_idx == 1:
            axes[0].set_xlabel("Past token position ($j$)")
            axes[1].set_xlabel("Past token position ($j$)")
        if head_iterator_idx == 0:
            axes[0].set_title("Before softmax")
            axes[1].set_title("After softmax")

        axes[0].set_ylabel(f"Head {head}")
        axes[1].set_ylabel("")

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

    fig.align_ylabels()
    # Save
    head_str = "-".join(map(str, heads))
    fig.savefig(
        save_dir.joinpath(f"l0tptpp_head-{head_str}_posi-{pos_i}.pdf"),
        bbox_inches="tight",
    )
    print("saved at:", save_dir.joinpath(f"l0tptpp_head-{head_str}_posi-{pos_i}.pdf"))


def both_score_vis_all_heads(
    wpe: TensorType[POS, HIDDEN_DIM],
    model_name: str,
    var_matrix: TensorType[POS, VOCAB],
):
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
    heads: list[int],
    **kwargs,
) -> None:
    if heads == [-1]:
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
            heads=heads,
            pos_i=kwargs["pos_i"],
        )
