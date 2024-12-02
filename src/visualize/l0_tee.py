from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import spearmanr
from torchtyping import TensorType
from tqdm import tqdm

from eqmodels import EQGPT2LMHeadModel
from eqmodels.gpt2 import compute_compare_score
from utils.mytorchtyping import HEAD, HIDDEN_DIM, POS, VOCAB

plt.rc("font", size=70)
plt.rcParams["axes.xmargin"] = 0.001

linewidth = 10
save_dir = Path("figures/l0_tee")
save_dir.mkdir(exist_ok=True, parents=True)


def compare_score_vis_all_heads(
    wte: TensorType[VOCAB, HIDDEN_DIM],
    n_samples: int,
    model_name: str,
    var_matrix: TensorType[VOCAB, VOCAB],
):
    torch.manual_seed(42)
    sampled_tokens = torch.randint(0, var_matrix.shape[1], (n_samples,))
    wte_sampled = wte[sampled_tokens]

    compare_score = compute_compare_score(
        i=wte_sampled.unsqueeze(0),
        j=wte_sampled.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu(),
    )

    var_matrix = var_matrix[:, sampled_tokens]
    var_matrix_for_compare_score: TensorType[VOCAB, VOCAB] = var_matrix.mean(
        dim=0
    ).unsqueeze(0).expand(var_matrix.shape[1], -1) * var_matrix.mean(dim=0).unsqueeze(
        1
    ).expand(var_matrix.shape[1], -1)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(80, 60), sharex=True)

    for h, head in enumerate(range(12)):
        ax_row = h // 4
        ax_col = h % 4

        ax = axes[ax_row, ax_col]

        # sns.heatmap(
        #     compare_score[0, head].cpu().numpy(),
        #     ax=ax
        #     center=0,
        #     cbar_kws={"location": "top"},
        #     cmap="RdBu",
        # )
        # axes[h * 2 + 0].xaxis.set_tick_params(
        #     width=linewidth, length=linewidth * 2, direction="out"
        # )
        # axes[h * 2 + 0].yaxis.set_tick_params(
        #     width=linewidth, length=linewidth * 2, direction="out"
        # )
        # axes[h * 2 + 0].set_title(
        #     "$\mathbf{e}_{\\text{ID}_i}\mathbf{W}_{"
        #     + str(head)
        #     + "}^{QK}\mathbf{e}_{\\text{ID}_j}^\\top$",
        #     y=1.4,
        # )
        # if h == 0:
        #     axes[h * 2 + 0].set_ylabel("Token $\\text{ID}_i$")
        # else:
        #     axes[h * 2 + 0].set_yticklabels([])

        sns.heatmap(
            compare_score[0, head].cpu().numpy() / var_matrix_for_compare_score,
            ax=ax,
            center=0,
            cbar_kws={"location": "top"},
            cmap="RdBu",
        )
        ax.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
        ax.yaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
        ax.set_title(
            "$T_{i, j, " + str(head) + "}^{ee}$",
            y=1.4,
        )
        if ax_col == 0:
            ax.set_ylabel("Token $\\text{ID}_i$")
        else:
            ax.set_yticklabels([])

        if ax_row == 1:
            ax.set_xlabel("Token $\\text{ID}_j$")
        else:
            ax.set_xticklabels([])

        print("------head", head)
        print(
            "score vs score_ln",
            spearmanr(
                compare_score[0, head].cpu().numpy().flatten(),
                (compare_score[0, head] / var_matrix_for_compare_score)
                .cpu()
                .numpy()
                .flatten(),
            ),
        )
        ax.set_xlabel("")

    fig.supxlabel("Token $\\text{ID}_j$")
    fig.subplots_adjust(hspace=0.4, wspace=0.1)

    fig.savefig(save_dir.joinpath("l0tee_head-all.png"), bbox_inches="tight")


def compare_score_vis(
    wte: TensorType[VOCAB, HIDDEN_DIM],
    n_samples: int,
    model_name: str,
    heads: list[int],
    var_matrix: TensorType[VOCAB, VOCAB],
):
    print(var_matrix.shape)
    torch.manual_seed(42)
    sampled_tokens = torch.randint(0, var_matrix.shape[1], (n_samples,))
    wte_sampled = wte[sampled_tokens]

    compare_score = compute_compare_score(
        i=wte_sampled.unsqueeze(0),
        j=wte_sampled.unsqueeze(0),
        w=EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu(),
    )

    var_matrix = var_matrix[:, sampled_tokens]

    var_matrix_for_compare_score: TensorType[VOCAB, VOCAB] = var_matrix.mean(
        dim=0
    ).unsqueeze(0).expand(var_matrix.shape[1], -1) * var_matrix.mean(dim=0).unsqueeze(
        1
    ).expand(var_matrix.shape[1], -1)

    fig, axes = plt.subplots(
        nrows=1, ncols=len(heads) * 2, figsize=(40, 10), sharex=True
    )

    for h, head in enumerate(heads):
        sns.heatmap(
            compare_score[0, head].cpu().numpy(),
            ax=axes[h * 2 + 0],
            center=0,
            cbar_kws={"location": "top"},
            cmap="RdBu",
        )
        axes[h * 2 + 0].xaxis.set_tick_params(
            width=linewidth, length=linewidth * 2, direction="out"
        )
        axes[h * 2 + 0].yaxis.set_tick_params(
            width=linewidth, length=linewidth * 2, direction="out"
        )
        axes[h * 2 + 0].set_title(
            "$\\mathbf{e}_{\\text{ID}_i}\\mathbf{W}_{"
            + str(head)
            + "}^{QK}\\mathbf{e}_{\\text{ID}_j}^\\top$",
            y=1.4,
        )
        if h == 0:
            axes[h * 2 + 0].set_ylabel("Token $\\text{ID}_i$")
        else:
            axes[h * 2 + 0].set_yticklabels([])

        sns.heatmap(
            compare_score[0, head].cpu().numpy() / var_matrix_for_compare_score,
            ax=axes[h * 2 + 1],
            center=0,
            cbar_kws={"location": "top"},
            cmap="RdBu",
        )
        axes[h * 2 + 1].xaxis.set_tick_params(
            width=linewidth, length=linewidth * 2, direction="out"
        )
        axes[h * 2 + 1].yaxis.set_tick_params(
            width=linewidth, length=linewidth * 2, direction="out"
        )
        axes[h * 2 + 1].set_title(
            "$T_{i, j, " + str(head) + "}^{ee}$",
            y=1.4,
        )
        axes[h * 2 + 1].set_yticklabels([])

        print("------head", head)
        print(
            "score vs score_ln",
            spearmanr(
                compare_score[0, head].cpu().numpy().flatten(),
                (compare_score[0, head] / var_matrix_for_compare_score)
                .cpu()
                .numpy()
                .flatten(),
            ),
        )

    fig.supxlabel("Token $\\text{ID}_j$", y=-0.2)
    fig.subplots_adjust(hspace=0.6, wspace=0.1)

    head_str = "-".join(map(str, heads))

    fig.savefig(save_dir.joinpath(f"l0tee_head-{head_str}.png"), bbox_inches="tight")
    print("saved at: ", save_dir.joinpath(f"l0tee_head-{head_str}.png"))
    # fig.savefig("compare_tok_score.png", bbox_inches="tight")
    # fig.savefig("compare_tok_score.pdf", bbox_inches="tight")


def main(
    model_name: str,
    wte: TensorType[VOCAB, HIDDEN_DIM],
    var_matrix: TensorType[POS, VOCAB],
    heads: list[int],
    n_samples: int,
    **kwargs,
) -> None:
    if heads == [-1]:
        compare_score_vis_all_heads(
            wte=wte,
            model_name=model_name,
            var_matrix=var_matrix,
            n_samples=n_samples,
        )
    else:
        compare_score_vis(
            wte=wte,
            model_name=model_name,
            var_matrix=var_matrix,
            heads=heads,
            n_samples=n_samples,
        )
