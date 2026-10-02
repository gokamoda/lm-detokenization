"""Helpers shared by the plots: figures at a target's printed size, bands and
panel labels."""

import contextlib
from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lm_detokenization.analysis.position_terms import Band
from lm_detokenization.plots.targets import FigureTarget, figure_size, target_rc

LINE_WIDTH = 0.8
BAND_ALPHA = 0.25


@contextlib.contextmanager
def target_style(target: FigureTarget) -> Iterator[None]:
    """Text created inside takes its size from the target (see targets.py)."""
    with plt.rc_context(
        {
            **target_rc(target),
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "axes.xmargin": 0.0,
            "lines.linewidth": LINE_WIDTH,
        }
    ):
        yield


def new_figure(
    target: FigureTarget,
    *,
    width: float,
    aspect: float,
    nrows: int = 1,
    ncols: int = 1,
    **subplots_kw,
) -> tuple[Figure, np.ndarray]:
    """A figure `width` x textwidth wide and `aspect` times as tall, with a 2D
    array of axes. Call inside target_style(target)."""
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figure_size(target, width=width, aspect=aspect),
        squeeze=False,
        layout="constrained",
        **subplots_kw,
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    return fig, axes


def plot_band(
    ax: Axes, band: Band, *, x: np.ndarray | None = None, color=None, label=None
) -> None:
    x = np.arange(len(band.mean)) if x is None else x
    (line,) = ax.plot(x, band.mean, color=color, label=label)
    ax.fill_between(
        x, band.low, band.high, color=line.get_color(), alpha=BAND_ALPHA, linewidth=0
    )


def panel_title(ax: Axes, text: str) -> None:
    """A label in the upper left corner inside the axes, as in the paper."""
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5},
    )


def panel_letter(ax: Axes, letter: str) -> None:
    """A bold panel letter (A, B, ...) above the upper left corner."""
    ax.text(
        0.0,
        1.02,
        letter,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontweight="bold",
    )


def top_legend(fig: Figure, ax: Axes) -> None:
    """The legend of `ax` in one row above all panels."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncols=len(labels), frameon=False
    )
