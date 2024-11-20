import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from torchtyping import TensorType

from utils.mytorchtyping import POS, VOCAB


def get_fig_axes_nrows2() -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(nrows=2, figsize=(40, 20), sharex=True)
    return fig, axes


def plot_lineplot_ln(
    ax: plt.Axes, scores: TensorType[POS, VOCAB], zorder: int = 1, linewidth: int = 10
) -> None:
    """_summary_

    Parameters
    ----------
    ax : plt.Axes
    scores : TensorType[POS, VOCAB]
        x-axis: POS (takes mean by VOCAB)
    zorder : int, optional
    """

    arr = scores.numpy()

    df = pl.DataFrame(
        {
            "pos": np.repeat(np.arange(arr.shape[0]), arr.shape[1]),
            "tok": np.tile(np.arange(arr.shape[1]), arr.shape[0]),
            "value": arr.flatten(),
        }
    )

    sns.lineplot(
        data=df.select(["pos", "value"]),
        x="pos",
        y="value",
        legend=False,
        errorbar=("pi", 95),
        ax=ax,
        linewidth=linewidth,
        zorder=zorder,
    )


def plot_lineplot_noln(
    ax: plt.Axes, scores: TensorType[POS], linewidth: int = 10
) -> None:
    """_summary_

    Parameters
    ----------
    ax : plt.Axes
    scores : TensorType[POS]
    """
    sns.lineplot(
        scores.numpy(),
        ax=ax,
        linewidth=linewidth,
    )
