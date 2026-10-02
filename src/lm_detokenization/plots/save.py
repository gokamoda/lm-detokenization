"""Saving figures at their printed size."""

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from lm_detokenization.plots.targets import FigureTarget, target_rc


def save_figure(fig: Figure, path: Path, *, target: FigureTarget | None) -> None:
    """Save ``fig`` at its exact size (no tight bbox) and close it.

    The file is written under a hidden temporary name and renamed into place,
    so a folder watcher (e.g. CloudLaTeX syncing ../thesis) never uploads a
    half-written PDF.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=f"{path.suffix}.tmp"
    )
    os.close(fd)
    try:
        with plt.rc_context(target_rc(target) if target else {}):
            fig.savefig(tmp_name, format=path.suffix.lstrip(".") or None)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        plt.close(fig)
