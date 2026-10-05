"""Plots of the position-derived terms (Section 5, Figure 3, Appendix D)."""

import numpy as np
from matplotlib.figure import Figure

from lm_detokenization.analysis import position_terms as pt
from lm_detokenization.analysis.empirical_attention import AttentionRows
from lm_detokenization.plots.common import (
    new_figure,
    panel_letter,
    panel_title,
    plot_band,
    target_style,
    top_legend,
)
from lm_detokenization.plots.targets import FigureTarget
from lm_detokenization.weights import Layer0Weights

POSITION_LABEL = "Past token position ($j$)"
SIGMA_I_LABELS = {
    "mean": r"mean $\sigma_i$",
    "max": r"max $\sigma_i$",
    "min": r"min $\sigma_i$",
}


def _tp(h) -> str:
    return rf"$T^{{\mathrm{{p}}}}_{{j,{h}}}$"


def _tpp(i, h) -> str:
    return rf"$T^{{\mathrm{{pp}}}}_{{{i},j,{h}}}$"


def _tp_tpp(i, h) -> str:
    return rf"$T^{{\mathrm{{pp}}}}_{{{i},j,{h}}}+T^{{\mathrm{{p}}}}_{{j,{h}}}$"


def _softmax(i, h) -> str:
    return (
        rf"$\mathrm{{softmax}}_j\,(T^{{\mathrm{{pp}}}}_{{{i},j,{h}}}"
        rf"+T^{{\mathrm{{p}}}}_{{j,{h}}})/\sqrt{{d'}}$"
    )


def _head_grid(n_panels: int, ncols: int) -> tuple[int, int]:
    ncols = min(ncols, n_panels)
    return -(-n_panels // ncols), ncols


def _finish_grid(axes: np.ndarray, n_panels: int) -> None:
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(POSITION_LABEL)


def plot_tp(
    weights: Layer0Weights,
    stats: pt.InverseSigmaStats,
    heads: list[int],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    ncols: int = 1,
    without_ln: bool = False,
) -> Figure:
    """T^p_{j,h} (mean and 95% interval over the vocabulary) for each head.
    With without_ln, each head gets a second column b^QK_h p_j^T first."""
    raw = pt.tp_raw(weights)
    with target_style(target):
        if without_ln:
            fig, axes = new_figure(
                target,
                width=width,
                aspect=aspect,
                nrows=len(heads),
                ncols=2,
                sharex=True,
            )
            for r, h in enumerate(heads):
                axes[r, 0].plot(raw[h].numpy())
                panel_title(
                    axes[r, 0], rf"$\mathbf{{b}}^{{QK}}_{{{h}}}\mathbf{{p}}_j^\top$"
                )
                plot_band(axes[r, 1], pt.tp_band(weights, stats, h))
                panel_title(axes[r, 1], _tp(h))
            axes[0, 0].set_title("Without LN")
            axes[0, 1].set_title("With LN")
            _finish_grid(axes, axes.size)
        else:
            nrows, ncols = _head_grid(len(heads), ncols)
            fig, axes = new_figure(
                target,
                width=width,
                aspect=aspect,
                nrows=nrows,
                ncols=ncols,
                sharex=True,
            )
            for ax, h in zip(axes.flat, heads):
                plot_band(ax, pt.tp_band(weights, stats, h))
                panel_title(ax, _tp(h))
            _finish_grid(axes, len(heads))
    return fig


def _plot_tpp_panel(ax, weights, stats, h, pos_i) -> None:
    for kind in pt.SIGMA_I_KINDS:
        plot_band(
            ax, pt.tpp_band(weights, stats, h, pos_i, kind), label=SIGMA_I_LABELS[kind]
        )
    panel_title(ax, _tpp(pos_i, h))


def plot_tpp(
    weights: Layer0Weights,
    stats: pt.InverseSigmaStats,
    heads: list[int],
    pos_is: list[int],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    without_ln: bool = False,
    legend: bool = False,
) -> Figure:
    """T^pp_{i,j,h} for j <= i, one row per head and one column per i. The
    three bands use the mean, max and min over the vocabulary of sigma_i.
    With without_ln, each i gets a column p_i W^QK_h p_j^T first."""
    per_i = 2 if without_ln else 1
    with target_style(target):
        fig, axes = new_figure(
            target,
            width=width,
            aspect=aspect,
            nrows=len(heads),
            ncols=len(pos_is) * per_i,
        )
        for r, h in enumerate(heads):
            for c, pos_i in enumerate(pos_is):
                if without_ln:
                    ax = axes[r, c * 2]
                    ax.plot(pt.tpp_raw(weights, pos_i)[h].numpy())
                    panel_title(
                        ax,
                        rf"$\mathbf{{p}}_{{{pos_i}}}\mathbf{{W}}^{{QK}}_{{{h}}}\mathbf{{p}}_j^\top$",
                    )
                _plot_tpp_panel(
                    axes[r, c * per_i + per_i - 1], weights, stats, h, pos_i
                )
        if without_ln:
            for c in range(len(pos_is)):
                axes[0, 2 * c].set_title("Without LN")
                axes[0, 2 * c + 1].set_title("With LN")
        if legend:
            top_legend(fig, axes[0, per_i - 1])
        _finish_grid(axes, axes.size)
    return fig


def plot_tp_tpp(
    weights: Layer0Weights,
    stats: pt.InverseSigmaStats,
    heads: list[int],
    pos_is: list[int],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    views: tuple[str, ...] = ("before", "after"),
    legend: bool = False,
) -> Figure:
    """T^pp_{i,j,h} + T^p_{j,h} before softmax (bands) and/or after softmax with
    temperature sqrt(d_head) (vocabulary mean), one row per head and one column
    per (i, view)."""
    with target_style(target):
        fig, axes = new_figure(
            target,
            width=width,
            aspect=aspect,
            nrows=len(heads),
            ncols=len(pos_is) * len(views),
        )
        for r, h in enumerate(heads):
            for c, (pos_i, view) in enumerate([(i, v) for i in pos_is for v in views]):
                ax = axes[r, c]
                for kind in pt.SIGMA_I_KINDS:
                    if view == "before":
                        plot_band(
                            ax,
                            pt.tp_tpp_band(weights, stats, h, pos_i, kind),
                            label=SIGMA_I_LABELS[kind],
                        )
                    else:
                        ax.plot(
                            pt.tp_tpp_softmax(weights, stats, h, pos_i, kind),
                            label=SIGMA_I_LABELS[kind],
                        )
                panel_title(
                    ax, _tp_tpp(pos_i, h) if view == "before" else _softmax(pos_i, h)
                )
        if len(views) == 2 and len(pos_is) == 1:
            axes[0, 0].set_title("Before softmax")
            axes[0, 1].set_title("After softmax")
        if legend:
            top_legend(fig, axes[0, 0])
        _finish_grid(axes, axes.size)
    return fig


def plot_undertrained(
    weights: Layer0Weights,
    head: int,
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    num_last: int = 5,
) -> Figure:
    """p_i W^QK_h p_j^T (no LN) for the last num_last query positions i."""
    num_positions = weights.wpe.shape[0]
    with target_style(target):
        fig, axes = new_figure(target, width=width, aspect=aspect)
        ax = axes[0, 0]
        for pos_i in range(num_positions - num_last, num_positions):
            ax.plot(pt.tpp_raw(weights, pos_i)[head].numpy(), label=f"$i={pos_i}$")
        panel_title(
            ax, rf"$\mathbf{{p}}_i\mathbf{{W}}^{{QK}}_{{{head}}}\mathbf{{p}}_j^\top$"
        )
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        ax.set_xlabel(POSITION_LABEL)
    return fig


def _empirical_band(rows: AttentionRows, head: int) -> pt.Band:
    w = rows.weights[:, head].float().numpy()  # [document, j]
    low, high = np.percentile(w, [2.5, 97.5], axis=0)
    return pt.Band(mean=w.mean(axis=0), low=low, high=high)


OBSERVED_COLOR = "red"
PREDICTION_COLOR = "C0"


def _alpha(i, h) -> str:
    return rf"$\alpha_{{{i},j,{h}}}$"


def _plot_empirical(ax, weights, stats, rows: AttentionRows, h, begin: int) -> None:
    """Observed attention (mean and 95% interval over documents) with the
    prediction from T^pp + T^p (dots), for j = begin .. i."""
    i = rows.position
    x = np.arange(i + 1)
    # Colors as in Figure 3 E of the paper: observed in red, prediction in blue.
    plot_band(ax, _empirical_band(rows, h), x=x, color=OBSERVED_COLOR, label="observed")
    prediction = pt.tp_tpp_softmax(weights, stats, h, i, "mean")
    ax.plot(
        x[begin:],
        prediction[begin:],
        color=PREDICTION_COLOR,
        marker="o",
        markersize=1.5,
        label=r"from $T^{\mathrm{pp}}+T^{\mathrm{p}}$",
    )
    ax.set_xlim(begin, i)
    ax.set_ylim(bottom=0)
    panel_title(ax, _alpha(i, h))


def plot_vs_tptpp(
    weights: Layer0Weights,
    stats: pt.InverseSigmaStats,
    rows: AttentionRows,
    heads: list[int],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    begin_offset: int,
    zoom_only: bool = False,
    legend: bool = False,
) -> Figure:
    """Observed layer-0 attention from position i vs. the prediction from
    T^pp + T^p: one row per head, the whole range and j >= begin_offset
    (only the latter with zoom_only, as in Figure 3 E)."""
    begins = [begin_offset] if zoom_only else [0, begin_offset]
    with target_style(target):
        fig, axes = new_figure(
            target, width=width, aspect=aspect, nrows=len(heads), ncols=len(begins)
        )
        for r, h in enumerate(heads):
            for c, begin in enumerate(begins):
                _plot_empirical(axes[r, c], weights, stats, rows, h, begin)
            axes[r, 0].set_ylabel(f"Head {h}")
        if legend:
            top_legend(fig, axes[0, 0])
        _finish_grid(axes, axes.size)
    return fig


def plot_position_bias(
    weights: Layer0Weights,
    stats: pt.InverseSigmaStats,
    rows: AttentionRows | None,
    heads: list[int],
    *,
    target: FigureTarget,
    width: float,
    aspect: float,
    pos_i: int = 500,
    begin_offset: int = 485,
) -> Figure:
    """Figure 3: A T^p, B T^pp, C T^pp + T^p, D its softmax, E the observed
    attention (only with rows), one row per head."""
    ncols = 5 if rows is not None else 4
    with target_style(target):
        fig, axes = new_figure(
            target, width=width, aspect=aspect, nrows=len(heads), ncols=ncols
        )
        for r, h in enumerate(heads):
            plot_band(axes[r, 0], pt.tp_band(weights, stats, h))
            panel_title(axes[r, 0], _tp(h))
            _plot_tpp_panel(axes[r, 1], weights, stats, h, pos_i)
            for kind in pt.SIGMA_I_KINDS:
                plot_band(axes[r, 2], pt.tp_tpp_band(weights, stats, h, pos_i, kind))
                axes[r, 3].plot(pt.tp_tpp_softmax(weights, stats, h, pos_i, kind))
            panel_title(axes[r, 2], _tp_tpp(pos_i, h))
            panel_title(axes[r, 3], _softmax(pos_i, h))
            if rows is not None:
                if rows.position != pos_i:
                    raise ValueError(
                        f"attention rows are for i={rows.position}, not {pos_i}"
                    )
                _plot_empirical(axes[r, 4], weights, stats, rows, h, begin_offset)
            axes[r, 0].set_ylabel(f"Head {h}")
        for c, letter in enumerate("ABCDE"[:ncols]):
            panel_letter(axes[0, c], letter)
        _finish_grid(axes, axes.size)
    return fig
