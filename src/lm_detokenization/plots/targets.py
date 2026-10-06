"""Page geometry of the documents that figures are made for (copied from
inseg-attention, with the naacl2025 paper added).

Figures for a document are drawn at their printed size, so LaTeX includes them
without scaling and the text in a figure keeps its LaTeX font size on the page.
"""

from dataclasses import dataclass

MM_PER_INCH = 25.4

# LaTeX font sizes (pt) for each class option, from size10.clo / size11.clo /
# size12.clo. They are a table per class size, not a fixed ratio.
LATEX_FONT_SIZES: dict[int, dict[str, float]] = {
    10: {
        "tiny": 5,
        "scriptsize": 7,
        "footnotesize": 8,
        "small": 9,
        "normalsize": 10,
        "large": 12,
        "Large": 14.4,
    },
    11: {
        "tiny": 6,
        "scriptsize": 8,
        "footnotesize": 9,
        "small": 10,
        "normalsize": 10.95,
        "large": 12,
        "Large": 14.4,
    },
    12: {
        "tiny": 6,
        "scriptsize": 8,
        "footnotesize": 10,
        "small": 10.95,
        "normalsize": 12,
        "large": 14.4,
        "Large": 17.28,
    },
}

# LaTeX size used for each text element of a figure. Two steps below what the
# names suggest: matplotlib's DejaVu Sans looks larger than the thesis body
# font (Times) at the same pt size, and figure text should read slightly
# smaller than the body.
ELEMENT_SIZES: dict[str, str] = {
    "font.size": "tiny",
    "axes.labelsize": "tiny",
    "xtick.labelsize": "tiny",
    "ytick.labelsize": "tiny",
    "legend.fontsize": "tiny",
    "axes.titlesize": "tiny",
    "figure.titlesize": "tiny",
}


@dataclass(frozen=True)
class FigureTarget:
    name: str
    textwidth_in: float
    class_fontsize: int  # the document class option: 10, 11 or 12 (pt)

    def fontsize(self, latex_size: str) -> float:
        """Return the pt size of ``latex_size`` (e.g. "footnotesize")."""
        return LATEX_FONT_SIZES[self.class_fontsize][latex_size]


TARGETS = {
    # conferences/naacl2025: \documentclass[11pt]{article} with acl.sty,
    # textwidth 16.0cm in two columns separated by 0.6cm (a column is
    # (16.0 - 0.6) / 2 = 7.7cm, i.e. width 0.48125). The appendix is one column.
    "naacl2025": FigureTarget(
        name="naacl2025", textwidth_in=160 / MM_PER_INCH, class_fontsize=11
    ),
    # ../thesis: \documentclass[11pt,a4paper]{report}; preamble.tex sets
    # A4 (210mm) with inner 30mm + outer 25mm margins. Same as inseg-attention.
    "thesis": FigureTarget(
        name="thesis", textwidth_in=(210 - 30 - 25) / MM_PER_INCH, class_fontsize=11
    ),
}
# Width of one column of the naacl2025 two-column layout, as a fraction of
# its textwidth.
NAACL2025_COLUMN = (16.0 - 0.6) / 2 / 16.0


def figure_size(
    target: FigureTarget, *, width: float, aspect: float
) -> tuple[float, float]:
    """Return (width, height) in inches for a figure ``width`` x textwidth wide."""
    width_in = target.textwidth_in * width
    return (width_in, width_in * aspect)


def target_rc(target: FigureTarget) -> dict[str, object]:
    """Matplotlib rcParams giving each text element its size in ELEMENT_SIZES."""
    rc: dict[str, object] = {
        key: target.fontsize(latex_size) for key, latex_size in ELEMENT_SIZES.items()
    }
    # TrueType instead of Type 3 fonts: several matplotlib PDFs included in one
    # LaTeX document otherwise corrupt each other under pdfTeX.
    rc["pdf.fonttype"] = 42
    # Resolution of rasterized parts (e.g. large heatmaps) in a vector PDF.
    rc["savefig.dpi"] = 600
    return rc
