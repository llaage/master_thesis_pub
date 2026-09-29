import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- design-system tokens (light surface) -------------------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
SERIES    = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE      = "#2a78d6"
ORANGE    = "#eb6834"
# sequential blue ramp 100 -> 700
SEQ_BLUE  = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5",
             "#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"]
CMAP_SEQ  = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
# diverging blue <-> red with neutral gray midpoint
CMAP_DIV  = LinearSegmentedColormap.from_list(
    "divbr", ["#0d366b","#2a78d6","#9ec5f4","#f0efec","#f0a8a8","#e34948","#8f1f1f"])

# SAP_FONT picks the typeface; it falls through to DejaVu Sans for any glyph the
# chosen face lacks. Aptos is not installed on this machine.
FONT = os.environ.get("SAP_FONT", "DejaVu Sans")

def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans", "Arial"],
        "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 8, "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })

def ygrid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)

def place_labels(xs, ys, min_sep):
    """greedy non-overlapping label selection, strongest peaks first"""
    keep, taken = [], []
    for x, y in sorted(zip(xs, ys), key=lambda t: -t[1]):
        if all(abs(x - t) >= min_sep for t in taken):
            keep.append((x, y)); taken.append(x)
    return keep

def caption(fig, text, y=None, fontsize=8):
    """Footnote under a figure, hard-wrapped to the figure width so the saved
    bbox never stretches sideways to fit one long line."""
    import textwrap
    w = fig.get_size_inches()[0]
    ncol = max(60, int(w * 13.6 * (8.0 / fontsize)))
    lines = textwrap.wrap(text, ncol)
    if y is None:
        y = -0.018 * len(lines) - 0.015
    return fig.text(0.005, y, "\n".join(lines), fontsize=fontsize,
                    color=MUTED, ha="left", va="top")

def keep_in_layout(ann):
    """Opt an annotation out of detach_annotations() so it is not clipped."""
    ann._keep_layout = True
    return ann

def detach_annotations(fig):
    """Annotations anchored in axes-fraction coordinates (headers, sub-titles,
    threshold captions) sit OUTSIDE the axes but are still children of it, so
    tight_layout counts them in the axes' bounding box and shrinks the plot to
    make room. Exclude them from the layout pass; bbox_inches='tight' still
    keeps them in the saved canvas."""
    from matplotlib.text import Annotation
    n = 0
    for ax in fig.axes:
        for t in ax.texts:
            if not isinstance(t, Annotation):
                continue
            if getattr(t, "_keep_layout", False):
                continue          # small edge labels that must stay in the canvas
            xc = t.xycoords
            frac = "axes fraction" in (xc if isinstance(xc, tuple) else (xc,))
            if frac:
                t.set_in_layout(False); n += 1
    return n

def header(fig, title, subtitle=None, title_size=14, sub_size=8.2, sub_frac=1.0):
    """Figure header drawn in FIGURE coordinates, with matching space reserved in
    the layout pass. Anchoring a header to an axes instead makes tight_layout
    shrink that axes to fit the text; excluding it from layout then clips it.

    sub_frac < 1 narrows the subtitle's wrap width, to keep it clear of a
    figure legend sitting in the same band."""
    import textwrap
    w, h = fig.get_size_inches()
    hpt = h * 72
    sub_lines = textwrap.wrap(subtitle, max(40, int(w * sub_frac * 13.6 * (8.0 / sub_size)))) if subtitle else []
    need = title_size * 1.6 + sub_size * 1.55 * len(sub_lines) + 12
    detach_annotations(fig)
    fig.tight_layout(rect=[0, 0, 1, 1 - need / hpt])
    fig.text(0.005, 1 - 4 / hpt, title, fontsize=title_size, fontweight="bold",
             color=INK, ha="left", va="top")
    if sub_lines:
        fig.text(0.005, 1 - (4 + title_size * 1.55) / hpt, "\n".join(sub_lines),
                 fontsize=sub_size, color=MUTED, ha="left", va="top",
                 linespacing=1.45)
