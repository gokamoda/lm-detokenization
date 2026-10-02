"""ROC curves of T^ee for one current token (Section 4.2, Figure 2 C)."""

from matplotlib.figure import Figure
from scipy import sparse

from lm_detokenization.analysis.detokenization import TokenAffinity, roc
from lm_detokenization.plots.common import new_figure, target_style
from lm_detokenization.plots.targets import FigureTarget


def plot_roc(
    affinity: TokenAffinity,
    bigram_counts: sparse.csc_matrix,
    panels: list[tuple[int, str, int]],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
) -> Figure:
    """One ROC curve per (suffix id, suffix text, head): how well T^ee ranks
    the prefixes that form an occurring bigram with the suffix."""
    with target_style(target):
        fig, axes = new_figure(
            target, width=width, aspect=aspect, ncols=len(panels), sharey=True
        )
        for ax, (suffix_id, suffix_text, head) in zip(axes.flat, panels):
            fpr, tpr, auc = roc(affinity, bigram_counts, suffix_id, head)
            (line,) = ax.plot(fpr, tpr)
            ax.fill_between(fpr, tpr, color=line.get_color(), alpha=0.2, linewidth=0)
            ax.plot([0, 1], [0, 1], ls="--", c=".3", linewidth=0.4)
            ax.text(
                0.1,
                0.1,
                f"AUC: {auc:.2f}",
                bbox={"boxstyle": "round", "fc": "w", "linewidth": 0.4},
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'"{suffix_text}"\nHead: {head}', pad=2)
            ax.set_xlabel("False Positive Rate")
        axes.flat[0].set_ylabel("True Positive Rate")
    return fig
