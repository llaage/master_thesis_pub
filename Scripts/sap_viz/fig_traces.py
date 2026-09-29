"""Per-backbone SAP traces, the aligned overlay, and the aligned heat map.

Hotspots that occur at the same aligned position in every backbone are named
once, in RH5.1 numbering, and carried across all panels by a guide line.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
import style as S
from conserved import conserved_hotspots, stagger

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = f"{HERE}/work"
OUTDIR = os.environ.get("SAP_OUTDIR", "/home/llaage/thesis/sap_score_viz")
os.makedirs(OUTDIR, exist_ok=True)
S.apply()

df = pd.read_csv(f"{WORK}/tidy_sap.csv")
meta = json.load(open(f"{WORK}/meta.json"))
ORDER, FILES, KIND = meta["order"], meta["files"], meta["kind"]
REF, IDENT, NCOL = meta["ref"], meta["identity"], meta["ncol"]
ref_num = {int(k): v for k, v in meta["ref_num_of_col"].items()}
ticks = [c for c in sorted(ref_num) if ref_num[c] % 25 == 0]
THRESH, NLAB = 2.0, 12

CONS, CTAG = conserved_hotspots(df, meta, THRESH)
CONS_SET = set(CONS)
GUIDE = S.ORANGE
SPECIFIC = "#8c8a84"          # backbone-specific hotspot: present but de-emphasised

def save(fig, name):
    fn = f"{OUTDIR}/{name}.png"
    fig.savefig(fn); fig.savefig(f"{OUTDIR}/{name}.pdf"); plt.close(fig)
    print("wrote", fn)

def conserved_ruler(ax, xlim, min_sep, levels=3, fontsize=7.5, fs=1.0):
    """Name each conserved hotspot once, above the panels, in REF numbering."""
    fontsize *= fs
    lvl = stagger(CONS, min_sep, levels)
    # at 1x keep the original rows; at larger fs pull them into the lower part of
    # the band so they stay clear of the figure header above
    ys = ([0.10, 0.42, 0.74][:levels] if fs < 1.5
          else [0.06 + i * (0.66 / levels) for i in range(levels)])
    for c in CONS:
        y = ys[lvl[c]]
        ax.vlines(c, 0, y - 0.04, color=GUIDE, alpha=0.55, lw=0.9)
        ax.text(c, y, CTAG[c], ha="center", va="bottom", fontsize=fontsize,
                fontweight="bold", color=S.INK)
    ax.set_xlim(*xlim); ax.set_ylim(0, 1.06)
    ax.set_yticks([])
    ax.tick_params(bottom=False, labelbottom=False)
    for sp in ax.spines.values(): sp.set_visible(False)

# ------------------------------------------------------- 1. one per backbone
for lab in ORDER:
    sub = df[df.label == lab].sort_values("resnum")
    x, y, aa = sub.resnum.values, sub.sap.values, sub.aa.values
    col = sub.aln_col.values
    cons = np.isin(col, CONS) & (y >= THRESH)
    spec = (y >= THRESH) & ~cons

    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.bar(x, y, width=1.0, color=S.BLUE, linewidth=0, zorder=3)
    ax.plot(x[spec], y[spec], "o", ms=4.5, color=S.SURFACE, mec=SPECIFIC, mew=1.3, zorder=5)
    ax.plot(x[cons], y[cons], "o", ms=5.0, color=S.ORANGE, mec=S.SURFACE, mew=1.0, zorder=6)
    ax.axhline(THRESH, color=S.BASELINE, linewidth=0.9, zorder=2)
    S.keep_in_layout(ax.annotate(
        f"SAP {THRESH:g}", xy=(1.0, THRESH), xycoords=("axes fraction", "data"),
        xytext=(4, -3), textcoords="offset points", ha="left", va="center",
        fontsize=7.5, color=S.MUTED, annotation_clip=False, zorder=6))
    ax.set_ylim(0, max(y.max() * 1.34, THRESH * 1.6))
    span = x.max() - x.min()
    for px, py in S.place_labels(x[y >= THRESH], y[y >= THRESH], span / 22.0)[:NLAB]:
        i = int(np.where(x == px)[0][0])
        ax.annotate(f"{aa[i]}{px}", (px, py), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color=S.INK, zorder=7,
                    fontweight="bold" if cons[i] else "normal")
        # the same site's name in the reference backbone, when it differs
        if cons[i] and lab != REF:
            ax.annotate(f"≡ {CTAG[int(col[i])]}", (px, py), textcoords="offset points",
                        xytext=(0, 17), ha="center", fontsize=6.8, color=S.ORANGE, zorder=7)
    ax.set_xlim(x.min() - 4, x.max() + 4)
    S.ygrid(ax)
    ax.set_ylabel("SAP score", fontsize=9)
    ax.set_xlabel("residue number", fontsize=9)
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=5, color=S.ORANGE, mec=S.SURFACE,
               label=f"hotspot conserved in all three backbones"),
        Line2D([], [], ls="", marker="o", ms=4.5, color=S.SURFACE, mec=SPECIFIC, mew=1.3,
               label=f"hotspot specific to {lab}")],
        loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"Per-residue SAP score — {lab}", fontsize=13, pad=26)
    ax.annotate(f"{FILES[lab]} · chain A · {KIND[lab]}   ·   {len(sub)} residues · "
                f"total SAP {y.sum():.1f} · max {y.max():.2f} · "
                f"{int((y >= THRESH).sum())} residues at SAP ≥ {THRESH:g} "
                f"({int(cons.sum())} of them conserved)",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", fontsize=8.5, color=S.INK2)
    cap = ("Bars: SAP score of each residue. Markers: surface-hydrophobicity hotspots "
           f"(SAP ≥ {THRESH:g}); up to {NLAB} of the best-separated peaks are labelled in this "
           "backbone's own numbering.")
    if lab != REF:
        cap += (f" For conserved hotspots the equivalent {REF} name is given above the label "
                "(≡), so the same physical site can be matched across figures.")
    S.caption(fig, cap)
    S.detach_annotations(fig)
    fig.tight_layout()
    save(fig, f"01_sap_per_residue_{lab}")

# ------------------------------------- 2. all backbones on a shared alignment
def _place_legend(fig, ax_last, handles, fs):
    """Put the key just below the axis label of the bottom panel.

    The offset is measured from the rendered geometry rather than guessed as an
    axes fraction: the panels change height with fs, so a fixed fraction leaves a
    growing band of dead space underneath."""
    h_in = ax_last.get_position().height * fig.get_size_inches()[1]
    below_in = (9 * fs + 9 * fs + 14) / 72.0      # tick labels + x label + gap
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(1.0, -below_in / max(h_in, 0.1)),
               bbox_transform=ax_last.transAxes,
               fontsize=8.5 * fs, ncol=2, handletextpad=0.6)

def _reassert_hspace(fig, gs, fs):
    """tight_layout recomputes hspace and does not know about the detached stats
    note, so the inter-panel gap is set again afterwards. It must clear the title
    (at its pad) plus the note line above each axes.

    Note this updates the GridSpec, not fig.subplots_adjust: axes built from an
    explicit GridSpec take their spacing from it, so subplots_adjust is ignored.
    """
    if fs < 1.5:
        return
    gap_pt = (8 + 12.5 * fs) + 11 * fs * 1.4 + 8 * fs * 1.4 + 10
    ax_h_in = fig.axes[1].get_position().height * fig.get_size_inches()[1]
    gs.update(hspace=(gap_pt / 72.0) / max(ax_h_in, 0.1))

def aligned_figure(minimal=False, fs=1.0, suffix=""):
    """minimal=True: no title/subtitle and a one-line caption, for a thesis
    figure whose caption is set by the document rather than baked into the PNG."""
    ymax = df.sap.max() * 1.22
    XLIM = (-3, NCOL + 3)
    PANEL_H = 2.15 * (1.0 if fs < 1.5 else 1.3)
    # the ruler is the one band that must grow with the type: its labels stack
    # The ruler is an annotation layer, not the data: hold it a little below the
    # body size so it reads as a label track. At 0.85 the names still fit in three
    # stagger rows, which keeps the band short.
    RS = 1.0 if fs < 1.5 else 0.85
    LEVELS = 3
    _line_in = 7.5 * fs * RS * 1.5 / 72.0
    top = max(0.55 if minimal else 0.95,
              LEVELS * _line_in / (0.70 if fs >= 1.5 else 0.9) + 0.30 * fs)
    fig = plt.figure(figsize=(14, top + PANEL_H * len(ORDER)))
    gs = fig.add_gridspec(len(ORDER) + 1, 1,
                          height_ratios=[top] + [PANEL_H] * len(ORDER),
                          hspace=0.34 * (1 if fs < 1.5 else 1.55))
    rul = fig.add_subplot(gs[0])
    axes = [fig.add_subplot(gs[i + 1], sharex=rul) for i in range(len(ORDER))]
    conserved_ruler(rul, XLIM, min_sep=NCOL / 37.0 * fs * RS,
                    levels=LEVELS, fontsize=7.5 * RS, fs=fs)

    for ax, lab in zip(axes, ORDER):
        sub = df[df.label == lab].sort_values("aln_col")
        for c in CONS:                  # carry each conserved site down every panel
            ax.axvline(c, color=GUIDE, alpha=0.32, lw=0.9, zorder=1)
        ax.bar(sub.aln_col, sub.sap, width=1.0 if fs < 1.5 else 1.6,
               color=S.BLUE, linewidth=0, zorder=3)
        hot = sub[sub.sap >= THRESH]
        cons = hot[hot.aln_col.isin(CONS_SET)]
        spec = hot[~hot.aln_col.isin(CONS_SET)]
        ax.plot(spec.aln_col, spec.sap, "o", ms=4.0 * fs, color=S.SURFACE,
                mec=SPECIFIC, mew=1.2 * fs, zorder=5)
        ax.plot(cons.aln_col, cons.sap, "o", ms=4.4 * fs, color=S.ORANGE,
                mec=S.SURFACE, mew=0.8 * fs, zorder=6)
        ax.axhline(THRESH, color=S.BASELINE, linewidth=0.8, zorder=2)
        ax.set_ylim(0, ymax); ax.set_xlim(*XLIM)
        S.ygrid(ax)
        # label the middle panel only; three repeats is pure ink at this size
        ax.set_ylabel("SAP score" if (fs < 1.5 or lab == ORDER[len(ORDER) // 2])
                      else "", fontsize=8.5 * fs)
        ax.set_title(lab, fontsize=11 * fs, fontweight="bold", color=S.INK, pad=4 * fs)
        # (per-panel residue/total-SAP/hotspot counts removed: at this type size
        #  they crowded out the data. The numbers live in the README.)
        # only backbone-specific peaks need naming inside the panel; the conserved
        # ones are named once in the ruler above
        _nlab = 5 if fs < 1.5 else 3
        for px, py in S.place_labels(spec.aln_col.values, spec.sap.values,
                                     NCOL / 20.0 * fs)[:_nlab]:
            r = spec[spec.aln_col == px].iloc[0]
            ax.annotate(f"{r.aa}{r.resnum}", (px, py), textcoords="offset points",
                        xytext=(0, 4 * fs), ha="center",
                        fontsize=7.2 * fs * (1.0 if fs < 1.5 else 0.85),
                        color=S.INK2, zorder=7)

    for a_ in axes[:-1]:
        a_.tick_params(bottom=False, labelbottom=False)
    step = 25 if fs < 1.5 else 50
    _cand = [c for c in sorted(ref_num) if ref_num[c] % step == 0]
    _gap = NCOL / 30.0 * fs          # min column separation for a tick label
    _ticks, _last = [], -1e9
    for c in _cand:
        if c - _last >= _gap:
            _ticks.append(c); _last = c
    axes[-1].set_xticks(_ticks)
    axes[-1].set_xticklabels([str(ref_num[c]) for c in _ticks])
    rul.tick_params(bottom=False, labelbottom=False)
    axes[-1].set_xlabel(f"aligned position  ({REF})", fontsize=9 * fs)
    for a_ in axes + [rul]:          # tick labels come from rcParams, not a kwarg
        a_.tick_params(labelsize=plt.rcParams["font.size"] * fs)

    handles = [
        Line2D([], [], ls="", marker="o", ms=5, color=S.ORANGE, mec=S.SURFACE,
               label="conserved hotspot" if minimal
                     else f"conserved hotspot (named above, in {REF} numbering)"),
        Line2D([], [], ls="", marker="o", ms=5, color=S.SURFACE, mec=SPECIFIC, mew=1.4,
               label="backbone-specific hotspot" if minimal
                     else "backbone-specific hotspot (own numbering)")]

    if minimal:
        if fs < 1.5:
            rul.legend(handles=handles, loc="upper right", fontsize=8.5 * fs,
                       ncol=1, borderpad=0.0, handletextpad=0.6)
        else:
            # the ruler's right side is occupied by its own labels at this size
            fig.legend(handles=handles, loc="upper right",
                       bbox_to_anchor=(1.0, -0.62), bbox_transform=axes[-1].transAxes,
                       fontsize=8.5 * fs, ncol=2, handletextpad=0.6)
        S.detach_annotations(fig)
        fig.tight_layout()
            # single short line: sit it just under the axis label, not a block below
        S.caption(fig, f"hotspot is SAP ≥ {THRESH:g}",
                  y=-0.004 if fs < 1.5 else -0.075, fontsize=8 * fs)
        save(fig, "02_sap_all_backbones_aligned_properlabel" + suffix)
        return

    S.header(fig, "Per-residue SAP across the three backbones, sequence-aligned",
             "Columns are structurally equivalent positions (global Needleman–Wunsch, "
             f"BLOSUM62, centre = {REF}). Orange guides mark the {len(CONS)} sites that are a "
             f"hotspot in every backbone; each is named once, in {REF} numbering, and is the "
             "same physical position in all three panels.",
             sub_frac=0.68 if fs < 1.5 else 0.97,
             title_size=14 * fs, sub_size=8.2 * fs)
    # figure-level. At 1x it fits beside the header; at 2x the title spans the
    # full width, so it moves to its own row under the axis label.
    if fs < 1.5:
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.999, 1.0),
                   fontsize=8.5 * fs, ncol=1)
    else:
        _place_legend(fig, axes[-1], handles, fs)
    S.caption(fig, f"A hotspot is SAP ≥ {THRESH:g}. Every conserved site carries the same residue "
                   "in all three backbones. Peaks that are a hotspot in only one or two backbones "
                   "are labelled inside their own panel, in that backbone's numbering. Tick "
                   f"numbers are {REF} residue numbers and skip residues disordered in the "
                   "crystal, so the axis is continuous in alignment columns, not in residue "
                   "numbers.", fontsize=8 * fs)
    save(fig, "02_sap_all_backbones_aligned" + suffix)

FS = float(os.environ.get("SAP_FIG2_FONT_SCALE", "2.0"))
aligned_figure(minimal=False, fs=FS)
aligned_figure(minimal=True,  fs=FS)

# ------------------------------------------------- 3. aligned heatmap version
mat = np.full((len(ORDER), NCOL), np.nan)
for i, lab in enumerate(ORDER):
    sub = df[df.label == lab]
    mat[i, sub.aln_col.values.astype(int)] = sub.sap.values

cmap = S.CMAP_SEQ.copy()
cmap.set_bad("#e6e5e0")           # alignment gap — must not read as "low SAP"
VMAX = 4.0
fig = plt.figure(figsize=(14, 3.5))
gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.9], hspace=0.16)
rul = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])
conserved_ruler(rul, (-0.5, NCOL - 0.5), min_sep=NCOL / 37.0)
im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap,
               norm=PowerNorm(gamma=0.6, vmin=0, vmax=VMAX), interpolation="nearest")
for k in range(1, len(ORDER)):
    ax.axhline(k - 0.5, color=S.SURFACE, lw=2.0, zorder=5)
# bracket each conserved column at the edges and in the row gaps rather than
# drawing across it — a line over the cell would hide the hotspot it points at
for y0, y1 in [(-0.5, -0.33), (0.40, 0.60), (1.40, 1.60), (2.33, 2.5)]:
    ax.vlines(CONS, y0, y1, color=GUIDE, lw=1.2, zorder=7)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels(ORDER, fontsize=10, color=S.INK)
ax.set_xticks(ticks); ax.set_xticklabels([str(ref_num[c]) for c in ticks])
ax.set_xlim(-0.5, NCOL - 0.5)
ax.set_xlabel(f"aligned position  ({REF} / 4wat chain A numbering)", fontsize=9)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=[rul, ax], pad=0.012, fraction=0.030)
cb.set_label(f"SAP score  (clipped at {VMAX:g})", fontsize=8.5, color=S.INK2)
cb.set_ticks([0, 0.5, 1, 2, 3, 4])
cb.outline.set_visible(False); cb.ax.tick_params(length=2, color=S.MUTED)
fig.text(0.005, 0.985, "Same data as a heat map — conserved vs. backbone-specific patches",
         fontsize=14, fontweight="bold", color=S.INK, ha="left", va="top")
fig.text(0.005, 0.925, "Grey cells = position absent from that backbone (alignment gap). "
                       f"Colour scale is gamma-compressed and clipped at {VMAX:g}. Orange "
                       f"guides mark the {len(CONS)} conserved hotspots.",
         fontsize=8.2, color=S.MUTED, ha="left", va="top")
S.detach_annotations(fig)
save(fig, "03_sap_aligned_heatmap")
