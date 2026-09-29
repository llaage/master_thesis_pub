"""Totals, top hotspots and SAP distribution across the three backbones."""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S
from conserved import conserved_hotspots

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = f"{HERE}/work"
OUTDIR = os.environ.get("SAP_OUTDIR", "/home/llaage/thesis/sap_score_viz")
os.makedirs(OUTDIR, exist_ok=True)
S.apply()

df = pd.read_csv(f"{WORK}/tidy_sap.csv")
meta = json.load(open(f"{WORK}/meta.json"))
ORDER, FILES, KIND, REF = meta["order"], meta["files"], meta["kind"], meta["ref"]
THRESH = 2.0

def save(fig, name):
    fn = f"{OUTDIR}/{name}.png"
    fig.savefig(fn); fig.savefig(f"{OUTDIR}/{name}.pdf"); plt.close(fig)
    print("wrote", fn)

# ============================================ 4. total vs. size-normalised SAP
agg = (df.groupby("label")
         .agg(total=("sap", "sum"), n=("sap", "size"),
              hot=("sap", lambda s: (s >= THRESH).sum()))
         .reindex(ORDER))
agg["per_res"] = agg.total / agg.n
agg["name"] = agg.index

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
panels = [("total",   "Total SAP score",  "sum over all scored residues",          "{:.1f}"),
          ("per_res", "SAP per residue",  "total ÷ number of residues — size-independent", "{:.3f}"),
          ("hot",     f"Hotspot residues (SAP ≥ {THRESH:g})", "count of high-SAP residues", "{:.0f}")]
for ax, (col, title, sub, fmt) in zip(axes, panels):
    o = agg.sort_values(col, ascending=True)
    ypos = np.arange(len(o))
    ax.barh(ypos, o[col], height=0.42, color=S.BLUE, linewidth=0, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels(o.name, fontsize=9.5, color=S.INK)
    for yy, v in zip(ypos, o[col]):
        ax.annotate(fmt.format(v), (v, yy), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=S.INK2)
    ax.set_xlim(0, o[col].max() * 1.22)
    ax.set_axisbelow(True); ax.grid(axis="x", color=S.GRID, linewidth=0.7)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=10.5, pad=22)
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
                textcoords="offset points", fontsize=8, color=S.MUTED)
best = agg.per_res.idxmin()
S.caption(fig, f"All three are single chains, so the three views largely agree. {best} has both "
               f"the lowest total SAP ({agg.total[best]:.1f}) and the lowest SAP per residue "
               f"({agg.per_res[best]:.3f}) — the least hydrophobic surface of the three. "
               "SAP per residue is the size-independent comparison; the totals also reflect "
               f"chain length ({', '.join(f'{k} {int(v)} res' for k, v in agg.n.items())}).")
S.header(fig, "Surface hydrophobicity per backbone")
save(fig, "04_total_sap_summary")

# =================================================== 5. top hotspots, faceted
TOPN = 10
CONS, CTAG = conserved_hotspots(df, meta, THRESH)
CONS_SET = set(CONS)
SPECIFIC = "#8c8a84"
hs = (df.sort_values("sap", ascending=False).groupby("label").head(TOPN)
        .assign(tag=lambda d: d.aa + d.resnum.astype(str)))
hs["conserved"] = hs.aln_col.isin(CONS_SET)
# a conserved site carries three different residue numbers; show the shared
# REF name next to the local one so the same site is recognisable across figures
hs["ylab"] = [t if (not c) or lab == REF else f"{t}  ≡ {CTAG[int(col)]}"
              for t, c, lab, col in zip(hs.tag, hs.conserved, hs.label, hs.aln_col)]
xmax = hs.sap.max() * 1.20
fig, axes = plt.subplots(1, len(ORDER), figsize=(10.5, 4.2), sharex=True)
for ax, lab in zip(axes, ORDER):
    o = hs[hs.label == lab].sort_values("sap")
    y = np.arange(len(o))
    ax.hlines(y, 0, o.sap, color=S.BLUE, linewidth=1.6, zorder=3)
    ax.plot(o.sap[~o.conserved], y[~o.conserved.values], "o", ms=8, color=S.SURFACE,
            mec=SPECIFIC, mew=1.6, zorder=4)
    ax.plot(o.sap[o.conserved], y[o.conserved.values], "o", ms=8, color=S.ORANGE,
            mec=S.SURFACE, mew=1.4, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(o.ylab, fontsize=9.5, color=S.INK)
    for yy, v in zip(y, o.sap):
        ax.annotate(f"{v:.2f}", (v, yy), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=S.MUTED)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.7, len(o) - 0.3)
    ax.set_axisbelow(True); ax.grid(axis="x", color=S.GRID, linewidth=0.7)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title(lab, fontsize=11, color=S.INK, pad=8)
    ax.set_xlabel("SAP score", fontsize=8.5)
    ax.set_xticks([0, 2, 4, 6])
S.header(fig, f"Top-{TOPN} surface-hydrophobicity hotspots per backbone",
         "Labelled in each backbone's own numbering; where the site is a hotspot in all three "
         f"backbones the shared {REF} name follows (≡). These are the primary redesign targets.")
# figure-level, in the header band: the panels have no reliably empty corner
fig.legend(handles=[
    Line2D([], [], ls="", marker="o", ms=7, color=S.ORANGE, mec=S.SURFACE,
           label="conserved in all three"),
    Line2D([], [], ls="", marker="o", ms=7, color=S.SURFACE, mec=SPECIFIC, mew=1.6,
           label="backbone-specific")],
    loc="upper right", bbox_to_anchor=(0.999, 1.0), fontsize=8.5, ncol=1)
save(fig, "05_top_hotspots")

# ============================== 6. distribution + cumulative SAP contribution
ranked = {lab: np.sort(df[df.label == lab].sap.values)[::-1] for lab in ORDER}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
for i, lab in enumerate(ORDER):
    v = ranked[lab]
    ax.plot(np.arange(1, len(v) + 1), v, lw=2, color=S.SERIES[i], zorder=3,
            solid_capstyle="round", label=lab)
ax.axhline(THRESH, color=S.BASELINE, lw=0.9, zorder=2)
ax.annotate(f"SAP {THRESH:g}", xy=(1.0, THRESH), xycoords=("axes fraction", "data"),
            xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=S.MUTED,
            annotation_clip=False, va="center")
for i, lab in enumerate(ORDER):          # staggered direct labels, never stacked
    v = ranked[lab]; k = 16 + 15 * i
    ax.annotate(lab, (k, v[k]), xytext=(12, 10 - 3 * i), textcoords="offset points",
                fontsize=8, color=S.INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=S.BASELINE, lw=0.7,
                                shrinkA=0, shrinkB=2))
ax.set_xlabel("residues, ranked by SAP score", fontsize=9)
ax.set_ylabel("SAP score", fontsize=9)
ax.set_xlim(0, 340); S.ygrid(ax)
ax.set_title("Rank profile — how steep is the hydrophobicity?", fontsize=10.5, pad=10)
ax.legend(loc="upper right", fontsize=8.5)

ax = axes[1]
n50 = {}
for i, lab in enumerate(ORDER):
    v = ranked[lab]
    cum = 100 * np.cumsum(v) / v.sum()
    ax.plot(np.arange(1, len(v) + 1), cum, lw=2, color=S.SERIES[i], zorder=3,
            solid_capstyle="round", label=lab)
    n50[lab] = int(np.searchsorted(cum, 50) + 1)
ax.axhline(50, color=S.BASELINE, lw=0.9, zorder=2)
lo, hi = min(n50.values()), max(n50.values())
ax.plot([lo, hi], [50, 50], color=S.ORANGE, lw=3, solid_capstyle="butt", zorder=5)
ax.annotate(f"half of all SAP sits on the top {lo}–{hi} residues" if lo != hi
            else f"half of all SAP sits on the top {lo} residues",
            (hi, 50), xytext=(10, -12), textcoords="offset points",
            fontsize=8, color=S.INK, va="center")
ax.set_xlim(0, 120); ax.set_ylim(0, 100)
ax.set_xlabel("number of top-ranked residues", fontsize=9)
ax.set_ylabel("cumulative share of total SAP  (%)", fontsize=9)
S.ygrid(ax)
ax.set_title("Concentration — few residues carry most of the score", fontsize=10.5, pad=10)
ax.legend(loc="lower right", fontsize=8.5)
S.caption(fig, "Left: SAP of every residue, sorted high→low. Right: cumulative share of the "
               "total. Residues needed to reach 50 % of total SAP — "
               + ",  ".join(f"{k} {v}" for k, v in n50.items()) + ". The three curves are "
               "nearly superimposed, so the backbones distribute their hydrophobicity the same "
               f"way; {min(n50, key=lambda k: ranked[k].sum())} differs by having uniformly "
               "lower SAP, not a different shape.")
S.header(fig, "SAP is concentrated in a small set of residues on every backbone")
save(fig, "06_sap_distribution")
