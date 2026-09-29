#!/usr/bin/env python3
"""Step 6 - complex I vs complex III ddG.

Unlike complex II, the R5.008 epitope partially overlaps the complex-I epitope
(F137 and the 226-233 helix), so complex III does respond to these mutations - in
both directions. The plot is scaled and annotated for that."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, "/home/llaage/projects/rh5_folding/260513_ddg_interface_scoring/com_II/scripts")
from _numbering import relabel

COM2 = Path("/home/llaage/projects/rh5_folding/260513_ddg_interface_scoring/com_III")
OUT = COM2 / "output" / "comparison_plots"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(COM2 / "output" / "comparison_comI_vs_comIII.csv")
df = df.dropna(subset=["ddg_comI_vs_wt", "ddg_comIII_vs_wt"])
# labels shown in 4WAT numbering (wild-type PfRH5); same mutations, renumbered
df["label"] = df["mutation_comI_8pwv"].map(relabel)

fig, ax = plt.subplots(figsize=(11, 8.5))
for setname, colour, marker in [("point_mutant", "#4299e7", "o"),
                                ("double_mutant", "#e4764a", "s")]:
    s = df[df["set"] == setname]
    ax.scatter(s["ddg_comI_vs_wt"], s["ddg_comIII_vs_wt"], s=70, c=colour, marker=marker,
               edgecolor="white", linewidth=0.8, alpha=0.9,
               label=f"{setname.replace('_', ' ')} (n={len(s)})")

ax.axhline(0, color="grey", lw=1)
ax.axvline(0, color="grey", lw=1)
ax.axhspan(-1, 1, color="#868e96", alpha=0.10, zorder=0)
ax.axvspan(-1, 1, color="grey", alpha=0.06, zorder=0)
ax.text(0.99, 0.02, "grey band: |ΔΔG| < 1 REU, within run-to-run noise",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555",
        style="italic")
ax.text(0.985, 0.975, "complex III binding TIGHTENED", transform=ax.transAxes,
        ha="right", va="top", fontsize=9.5, color="#2f9e44", fontweight="bold")
ax.text(0.015, 0.025, "complex III binding WEAKENED", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=9.5, color="#c92a2a", fontweight="bold")

top = df.reindex(df["selectivity_ddg_comI_minus_comIII"].abs().sort_values(ascending=False).index).head(12)
for _, r in top.iterrows():
    ax.annotate(r["label"], (r["ddg_comI_vs_wt"], r["ddg_comIII_vs_wt"]),
                textcoords="offset points", xytext=(6, 5), fontsize=7.5, color="#333")

lim = max(abs(df["ddg_comI_vs_wt"]).max(), abs(df["ddg_comIII_vs_wt"]).max()) * 1.08
ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="#bbb", zorder=0, label="equal effect")
ax.set_xlim(-2, lim)
ylim = max(abs(df["ddg_comIII_vs_wt"])) * 1.25
ax.set_ylim(-ylim, ylim)
ax.set_xlabel("ΔΔG vs WT, complex I  (8pwv, RH5 · scFv)   [REU]", fontsize=12, fontweight="bold")
ax.set_ylabel("ΔΔG vs WT, complex III  (8pwx, RH5 · R5.008)   [REU]", fontsize=12, fontweight="bold")
ax.set_title("Same RH5 interface mutations scored against R5.008, whose epitope "
             "partly overlaps complex I\n"
             "right = complex I binding lost · below 0 = complex III binding weakened too",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "01_ddg_comI_vs_comIII.png", dpi=300, bbox_inches="tight")
print("saved", OUT / "01_ddg_comI_vs_comIII.png")

# ranked selectivity bar chart
d = df.sort_values("selectivity_ddg_comI_minus_comIII", ascending=True)
fig, ax = plt.subplots(figsize=(9, 14))
colours = ["#e4764a" if s == "double_mutant" else "#4299e7" for s in d["set"]]
ax.barh(range(len(d)), d["selectivity_ddg_comI_minus_comIII"], color=colours)
ax.set_yticks(range(len(d)))
ax.set_yticklabels(d["label"], fontsize=8)
ax.set_ylim(-0.6, len(d) - 0.4)
ax.axvline(0, color="grey", lw=1)
ax.set_xlabel("ΔΔG(complex I) − ΔΔG(complex III)   [REU]", fontsize=12, fontweight="bold")
ax.set_title("Epitope selectivity of each mutation\n(large positive = kills complex I, spares complex III)",
             fontsize=13, fontweight="bold", pad=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "02_selectivity_ranking.png", dpi=300, bbox_inches="tight")
print("saved", OUT / "02_selectivity_ranking.png")
