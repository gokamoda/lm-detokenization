"""Plot of the contribution of each of the six terms (Section 6.1, Figure 4)."""

import numpy as np
from matplotlib.figure import Figure

from lm_detokenization.analysis.position_terms import Band
from lm_detokenization.analysis.six_terms import TERM_LABELS, TERMS
from lm_detokenization.plots.common import (
    new_figure,
    panel_title,
    plot_band,
    target_style,
)
from lm_detokenization.plots.targets import FigureTarget

# Head value meaning "averaged over heads".
MEAN_OVER_HEADS = -1


def contribution_band(contributions: np.ndarray, head: int, term: int) -> Band:
    """Mean over documents, per query position i, with a normal-approximation
    95% confidence interval (mean +- 1.96 SE). As in the published plot,
    i = 0 and i = 1023 are dropped and values <= 1e-10 are left out; with
    MEAN_OVER_HEADS, values are first averaged over heads per document."""
    if head == MEAN_OVER_HEADS:
        values = np.nanmean(contributions[..., term], axis=1)  # [document, i]
    else:
        values = np.asarray(contributions[:, head, :, term])
    values = np.where(values > 1e-10, values, np.nan)
    values[:, 0] = np.nan
    values[:, -1] = np.nan
    n = np.sum(~np.isnan(values), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(values, axis=0)
        se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(n)
    return Band(mean=mean, low=mean - 1.96 * se, high=mean + 1.96 * se)


def plot_six_terms(
    contributions: np.ndarray,
    heads: list[int],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    ncols: int = 2,
    legend: str = "top",
) -> Figure:
    """C_{i,h} of the six terms on a log axis, one panel per head
    (MEAN_OVER_HEADS for the average over heads). legend: top, right or none."""
    ncols = min(ncols, len(heads))
    nrows = -(-len(heads) // ncols)
    with target_style(target):
        fig, axes = new_figure(
            target, width=width, aspect=aspect, nrows=nrows, ncols=ncols, sharex=True
        )
        for ax, head in zip(axes.flat, heads):
            lowest_mean = np.inf
            for t, name in enumerate(TERMS):
                band = contribution_band(contributions, head, t)
                plot_band(ax, band, label=rf"$C^{{{TERM_LABELS[name]}}}$")
                lowest_mean = min(lowest_mean, np.nanmin(band.mean))
            ax.set_yscale("log")
            # The lower end of a band can reach 0 (large SE); keep the range
            # to the mean lines instead of following it.
            ax.set_ylim(bottom=lowest_mean / 10)
            panel_title(
                ax, r"$C_{i}$" if head == MEAN_OVER_HEADS else rf"$C_{{i,{head}}}$"
            )
        for ax in axes.flat[len(heads) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("Present token position ($i$)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if legend == "top":
            fig.legend(
                handles, labels, loc="outside upper center", ncol=3, frameon=False
            )
        elif legend == "right":
            fig.legend(handles, labels, loc="outside right center", frameon=False)
    return fig
