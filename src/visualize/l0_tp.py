from pathlib import Path

import matplotlib.pyplot as plt
from torchtyping import TensorType
from tqdm import tqdm

from eqmodels import EQGPT2LMHeadModel
from eqmodels.gpt2 import compute_self_score
from utils.ln import ln_pos
from utils.mytorchtyping import HEAD, HIDDEN_DIM, POS, VOCAB

from .subplots import get_fig_axes_nrows2, plot_lineplot_ln, plot_lineplot_noln

plt.rc("font", size=80)
plt.rcParams["axes.xmargin"] = 0.001

linewidth = 10
save_dir = Path("figures/l0_tp")
save_dir.mkdir(exist_ok=True, parents=True)


def self_score_vis(
    wpe: TensorType[POS, HIDDEN_DIM],
    model_name: str,
    heads: list[int],
    var_matrix: TensorType[POS, VOCAB],
) -> None:
    self_score: TensorType[1, HEAD, POS] = compute_self_score(
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.bqwkh.detach()
        .cpu(),
    )

    assert len(heads) <= 2
    fig, _axes = get_fig_axes_nrows2()

    for head_iterator_idx, head in enumerate(heads):
        # print(f"Plotting head {head}")
        axes = _axes[head_iterator_idx]
        # # Without LN
        print("Plotting without LN")
        plot_lineplot_noln(axes[0], self_score[0, head], linewidth=linewidth)

        # With LN
        print("Plotting with LN")
        self_score_: TensorType[1, HEAD, POS, VOCAB] = ln_pos(
            x=self_score, var_matrix=var_matrix
        )
        plot_lineplot_ln(axes[1], self_score_[0, head], linewidth=linewidth)

        # Adjustments

        # Space between subplots
        # fig.subplots_adjust(wspace=0.1, hspace=0.1)

        # Titles
        axes[0].annotate(
            "$\\mathbf{b}_{" + str(head) + "}^{QK}\\mathbf{p}_j^\\top$",
            xy=(0.006, 0.82),
            xycoords="axes fraction",
            fontsize=80,
        )
        axes[1].annotate(
            "$T^{\\text{p}}_{j, " + str(head) + "}$",
            xy=(0.006, 0.82),
            xycoords="axes fraction",
            fontsize=80,
        )
        # axes[0].set_title(
        #     "$\\mathbf{b}_{" + str(head) + "}^{QK}\\mathbf{p}_j^\\top$",
        #     y=0.82,
        #     x=0.006,
        #     loc="left",
        # )
        # axes[1].set_title(
        #     "$T^{\\text{p}}_{j, " + str(head) + "}$",
        #     y=0.80,
        #     x=0.006,
        #     loc="left",
        # )

        # Labels
        if head_iterator_idx == 1:
            axes[0].set_xlabel("Past token position ($j$)")
            axes[1].set_xlabel("Past token position ($j$)")
        if head_iterator_idx == 0:
            axes[0].set_title("Without LN")
            axes[1].set_title("With LN")

        axes[0].set_ylabel(f"Head {head}")
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
    fig.subplots_adjust(wspace=0.25)

    # Save
    print("Saving")
    head_str = "-".join(map(str, heads))
    fig.savefig(save_dir.joinpath(f"l0tp_head-{head_str}.pdf"), bbox_inches="tight")
    print("saved at:", save_dir.joinpath(f"l0tp_head-{head_str}.pdf"))


def self_score_vis_all_heads(
    wpe: TensorType[POS, HIDDEN_DIM],
    model_name: str,
    var_matrix: TensorType[POS, VOCAB],
) -> None:
    print("Visualizing TP for all heads")

    self_score: TensorType[1, HEAD, POS] = compute_self_score(
        j=wpe.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.bqwkh.detach()
        .cpu(),
    )

    n_heads = self_score.shape[1]
    if n_heads == 12:
        fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(80, 60), sharex="col")
    else:
        raise NotImplementedError

    # With LN
    self_score: TensorType[1, HEAD, POS, VOCAB] = ln_pos(
        x=self_score, var_matrix=var_matrix
    )
    for head in tqdm(range(n_heads), desc="Plotting heads"):
        ax = axes[head // 2, head % 2]

        plot_lineplot_ln(
            ax,
            self_score[0, head],
            linewidth=linewidth,
        )

        # Adjustments

        # Titles
        ax.set_title(
            "$T_{j, " + str(head) + "}^{p}$",
            y=0.80,
            x=0.006,
            loc="left",
        )

        # Labels
        ax.set_ylabel("")
        if head // 2 == 5:
            ax.set_xlabel("Past token position ($j$)")

        # Ticks
        ax.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
        ax.yaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    # Save
    fig.savefig(save_dir.joinpath("l0tp_head-all.pdf"), bbox_inches="tight")


def main(
    model_name: str,
    wpe: TensorType[POS, HIDDEN_DIM],
    var_matrix: TensorType[POS, VOCAB],
    heads: list[int],
    **kwargs,
) -> None:
    del kwargs
    if heads == [-1]:
        self_score_vis_all_heads(
            wpe=wpe,
            model_name=model_name,
            var_matrix=var_matrix,
        )
    else:
        self_score_vis(
            wpe=wpe,
            model_name=model_name,
            heads=heads,
            var_matrix=var_matrix,
        )
