"""Plots of the token-derived terms and of the embedding variances
(Sections 4 and 6, Appendix D)."""

import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from lm_detokenization.analysis.token_terms import TeeSample, te_raw, te_with_ln
from lm_detokenization.plots.common import new_figure, panel_title, target_style
from lm_detokenization.plots.targets import FigureTarget
from lm_detokenization.weights import Layer0Weights


def _grid(n: int, ncols: int) -> tuple[int, int]:
    ncols = min(ncols, n)
    return -(-n // ncols), ncols


def plot_tee(
    sample: TeeSample,
    heads: list[int],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    ncols: int = 4,
    without_ln: bool = False,
    tick_every: int = 17,
) -> Figure:
    """Heatmaps of T^ee_{i,j,h} between the sampled tokens, one per head. With
    without_ln, e_i W^QK_h e_j^T is shown next to each."""
    panels = [
        (h, raw) for h in heads for raw in ((True, False) if without_ln else (False,))
    ]
    nrows, ncols = _grid(len(panels), ncols * (2 if without_ln else 1))
    with target_style(target):
        fig, axes = new_figure(
            target, width=width, aspect=aspect, nrows=nrows, ncols=ncols
        )
        for ax, (h, raw) in zip(axes.flat, panels):
            values = (sample.raw if raw else sample.with_ln)[h].numpy()
            sns.heatmap(
                values,
                ax=ax,
                center=0,
                cmap="RdBu",
                square=False,
                xticklabels=tick_every,
                yticklabels=tick_every,
                rasterized=True,
                cbar_kws={"location": "top", "pad": 0.02, "aspect": 30},
            )
            ax.set_title(
                rf"$\mathbf{{e}}_i\mathbf{{W}}^{{QK}}_{{{h}}}\mathbf{{e}}_j^\top$"
                if raw
                else rf"$T^{{ee}}_{{i,j,{h}}}$",
                pad=2,
            )
            ax.tick_params(labelsize=ax.xaxis.label.get_fontsize() * 0.8, length=1)
        for ax in axes.flat[len(panels) :]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"Token $\mathrm{ID}_i$")
        for ax in axes[-1]:
            ax.set_xlabel(r"Token $\mathrm{ID}_j$")
    return fig


def plot_te(
    weights: Layer0Weights,
    token_counts: np.ndarray,
    heads: list[int],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    ncols: int = 4,
    without_ln: bool = False,
) -> Figure:
    """T^e_{j,h} (averaged over positions) of every token against its corpus
    count, one panel per head. Tokens that never occur are not shown (log
    axis). With without_ln, b^QK_h e_j^T is plotted instead."""
    values = (te_raw(weights) if without_ln else te_with_ln(weights)).numpy()
    occurs = token_counts > 0
    nrows, ncols = _grid(len(heads), ncols)
    with target_style(target):
        fig, axes = new_figure(
            target, width=width, aspect=aspect, nrows=nrows, ncols=ncols, sharex=True
        )
        for ax, h in zip(axes.flat, heads):
            ax.scatter(
                token_counts[occurs],
                values[h][occurs],
                s=1.0,
                alpha=0.3,
                marker=".",
                linewidths=0,
                rasterized=True,
            )
            ax.set_xscale("log")
            ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
            panel_title(
                ax,
                rf"$\mathbf{{b}}^{{QK}}_{{{h}}}\mathbf{{e}}_j^\top$"
                if without_ln
                else rf"$T^{{e}}_{{j,{h}}}$",
            )
        for ax in axes.flat[len(heads) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("Token ($j$) count")
    return fig


VARIANCE_PANELS = ("wte", "wpe", "wpe_ends")


def plot_variance(
    weights: Layer0Weights,
    token_counts: np.ndarray | None,
    panels: tuple[str, ...],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    num_ends: int = 11,
) -> Figure:
    """Variance of the embeddings (Section 6.3-6.4):
    wte: Var(e_ID) against the token count; wpe: Var(p_i) over all positions;
    wpe_ends: Var(p_i) at the first and the last num_ends positions (stacked)."""
    wte_var = weights.wte.var(dim=-1, unbiased=False).numpy()
    wpe_var = weights.wpe.var(dim=-1, unbiased=False).numpy()
    num_positions = len(wpe_var)
    with target_style(target):
        fig = new_figure(target, width=width, aspect=aspect)[0]
        fig.delaxes(fig.axes[0])
        subfigs = fig.subfigures(1, len(panels), squeeze=False)[0]
        for subfig, panel in zip(subfigs, panels):
            if panel == "wte":
                if token_counts is None:
                    raise ValueError("the wte panel needs token counts")
                ax = subfig.subplots()
                occurs = token_counts > 0
                ax.scatter(
                    token_counts[occurs],
                    wte_var[occurs],
                    s=1.0,
                    alpha=0.5,
                    marker=".",
                    linewidths=0,
                    rasterized=True,
                )
                ax.set_xscale("log")
                ax.set_xlabel("Token count")
                panel_title(ax, r"$\mathrm{Var}(\mathbf{e}_{\mathrm{ID}})$")
            elif panel == "wpe":
                ax = subfig.subplots()
                ax.plot(wpe_var)
                ax.set_xlabel("Position ($i$)")
                panel_title(ax, r"$\mathrm{Var}(\mathbf{p}_i)$")
            elif panel == "wpe_ends":
                top, bottom = subfig.subplots(2, 1)
                head = np.arange(num_ends)
                tail = np.arange(num_positions - num_ends, num_positions)
                top.plot(head, wpe_var[head], marker="o", markersize=1.5)
                bottom.plot(tail, wpe_var[tail], marker="o", markersize=1.5)
                # integer positions, both ends included (0, 5, 10 / 1013, 1018, 1023)
                top.set_xticks(head[::5])
                bottom.set_xticks(tail[::5])
                bottom.set_xlabel("Position ($i$)")
                for ax in (top, bottom):
                    panel_title(ax, r"$\mathrm{Var}(\mathbf{p}_i)$")
            else:
                raise ValueError(
                    f"unknown panel {panel!r}; choose from {VARIANCE_PANELS}"
                )
    return fig
