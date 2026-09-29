#!/usr/bin/env python
"""
Step 12 -- figures for the duration analysis (rung 2), in 4WAT wild-type
numbering, using the same patch vocabulary as Figure 8.

(a) residue x residue map coloured by tau_half instead of contact frequency
(b) longest-lived pairs, ranked by tau_half
(c) per-residue persistence along the sequence

tau_half is the lag at which the baseline-corrected contact correlation falls to
0.5: "half the contacts present at a given moment are still present this much
later, above what returning by chance would give". It is NOT the mean episode
duration, which is unusable here (a 2 ns change in flicker tolerance moves it
tenfold; see the docstring of step 11).

Usage: python steps/12_duration_figure.py [--outdir figures/time_dependent]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "lib"))
CFG = json.load(open(os.path.join(ROOT, "config.json")))

import matplotlib.pyplot as plt         # noqa: E402
import plotting as P                    # noqa: E402
import patches as PT                    # noqa: E402
import bubblekey as BK                  # noqa: E402
import cifio                            # noqa: E402
import align as _al                     # noqa: E402

P.setup()
NRES = CFG["domains"]["RH5"][1]


def sequence(rung):
    src = os.path.join(CFG["base"], "inputs", "repose", f"{rung}_r1.pdb")
    if not os.path.isfile(src):
        src = os.path.join(CFG["base"], "inputs", "97p04_fusion_trimer.pdb")
    first = {}
    for L in open(src):
        if L.startswith(("ATOM", "HETATM")) and L[21] == "A":
            r = int(L[22:26]); first.setdefault(r, L[17:20].strip())
    return {r: cifio.AA3TO1.get(first.get(r, "XXX"), "X") for r in range(1, NRES + 1)}


def ref_map(cif, ours):
    atoms = cifio.read_atoms(cif); ents = cifio.read_entities(cif)
    rh5 = [k for k, v in ents.items() if "rh5" in v.lower()]
    best = None
    for ch in sorted(set(atoms["chain"])):
        m = (atoms["chain"] == ch) & np.isin(atoms["entity"], rh5)
        if m.sum() == 0:
            continue
        first = {}
        for r, n in zip(atoms["resid"][m], atoms["resname"][m]):
            first.setdefault(int(r), n)
        rid = sorted(first)
        if best is None or len(rid) > len(best[0]):
            best = (rid, first)
    rid, first = best
    ref = "".join(cifio.AA3TO1.get(first[r], "X") for r in rid)
    pr = _al.align(ours, ref)
    return {i + 1: int(rid[j]) for i, j in pr}


def marginal_freq(rung):
    """Contact-frequency marginal, exactly as Figure 8 computes it."""
    tot = np.zeros(NRES)
    tags = CFG["rungs"][rung]["replicates"]
    for tag in tags:
        z = np.load(os.path.join(ROOT, "results", "matrices", f"{tag}.npz"),
                    allow_pickle=True)
        acc = np.zeros(NRES); m = 0
        for k in [x for x in z.files if x.endswith("__M")]:
            key = k[:-3]; a, b = key.split("__")
            if not (a.startswith("rh5") and b.startswith("rh5")):
                continue
            for side in "ab":
                for l, v in zip(z[f"{key}__labels_{side}"], z[f"{key}__freq_{side}"]):
                    r = int(str(l).split(":")[1])
                    if 1 <= r <= NRES:
                        acc[r - 1] += v
            m += 1
        tot += acc / max(1, 2 * m)
    return tot / len(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="rung2")
    ap.add_argument("--dur", default=None)
    ap.add_argument("--cif", default="/home/llaage/PyMOL/rh5/4wat.cif")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "figures", "time_dependent"))
    ap.add_argument("--min-tau", type=float, default=400.0,
                    help="panel (b) shows pairs at or above this tau_half, in ns")
    ap.add_argument("--top", type=int, default=25,
                    help="hard cap on panel (b) rows; the cutoff normally governs")
    ap.add_argument("--clean", action="store_true",
                    help="drop the explanatory annotations and the censoring "
                         "arrows; axis labels lose the numbering note")
    ap.add_argument("--font", default=None,
                    help="font family for the whole figure, e.g. Calibri")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--panel-a", default="map", choices=["map", "bubble"],
                    help="'map' = per-cell heatmap; 'bubble' = one uniform "
                         "square per quotable pair. Only 203 of ~73000 cells "
                         "carry a value, and at print size a cell is ~0.01 inch "
                         "-- visible as a speck but too small to read a colour "
                         "from. Markers decouple symbol size from matrix "
                         "resolution; position stays exact.")
    ap.add_argument("--fontscale", type=float, default=1.0,
                    help="multiply every font size; use with a taller --figsize")
    ap.add_argument("--figsize", default="7.2,5.4",
                    help="inches, w,h. Width should stay near the printable "
                         "width of the target page: pasting at that width fixes "
                         "the scale, so only font size RELATIVE to width matters.")
    ap.add_argument("--a4", action="store_true",
                    help="preset for pasting into A4 and printing: larger type "
                         "on a taller canvas of the same width")
    a = ap.parse_args()
    dur = json.load(open(a.dur or os.path.join(ROOT, "results",
                                               f"duration_{a.rung}.json")))
    seq = sequence(a.rung)
    NUM = ref_map(a.cif, "".join(seq[r] for r in range(1, NRES + 1)))
    freq = marginal_freq(a.rung)
    pats = PT.derive(freq, floor_frac=0.06, gap=6, max_patches=6)

    def show(r):
        if r not in NUM:
            return f"{seq[r]}{r}?"
        return f"{seq[r]}{NUM[r]}"

    if a.a4:
        a.fontscale = max(a.fontscale, 1.75)
        a.figsize = "7.2,8.6"
    F = a.fontscale
    W, H = (float(x) for x in a.figsize.split(","))

    if a.font:
        import matplotlib as mpl
        mpl.rcParams["font.family"] = a.font
        # Deliberately NOT mathtext.fontset="custom": pointing mathtext at
        # Calibri drops glyphs it has no math coverage for. That silently ate
        # the subscript in every tau_1/2 label AND the value labels on the bars,
        # and left a stray glyph in the corner. Plain Unicode below instead.

    CLEAN = a.clean
    # U+03C4 + U+00BD: both present in Calibri and in the default face, and
    # they need no mathtext at all.
    TAU = "\u03c4\u00bd" if a.font else r"$\tau_{1/2}$"
    axlab = "RH5 residue" if CLEAN else "RH5 residue (4WAT numbering)"

    quot = [p for p in dur["pairs"] if p["quotable"]]
    WIN = max(dur["lags_ns"])

    # per-residue tau_half, only where there is enough evidence
    MIN_OBS_RES = 300
    th = np.full(NRES, np.nan); cen = np.zeros(NRES, bool)
    for r in dur["per_residue"]:
        if r["observations"] >= MIN_OBS_RES and np.isfinite(r["tau_half_ns"]):
            th[r["resid"] - 1] = r["tau_half_ns"]
            cen[r["resid"] - 1] = r["tau_half_censored"]

    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], width_ratios=[1, 1])
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # ---- (a) map coloured by tau_half
    M = np.full((NRES, NRES), np.nan)
    for p in quot:
        M[p["i"] - 1, p["j"] - 1] = p["tau_half_ns"]
        M[p["j"] - 1, p["i"] - 1] = p["tau_half_ns"]
    rows = [p["i"] for p in quot] + [p["j"] for p in quot]
    lo = max(1, min(rows) - 8); hi = min(NRES, max(rows) + 8)
    if pats:
        lo = max(1, min(lo, min(q["start"] for q in pats) - 4))
        hi = min(NRES, max(hi, max(q["end"] for q in pats) + 4))
    sub = M[lo - 1:hi, lo - 1:hi]
    # cividis, matching Figure 8's map, so the two can be read side by side
    # without recalibrating what "bright" means.
    cmap = plt.get_cmap("cividis").copy(); cmap.set_bad("#f2f2f2")
    if a.panel_a == "bubble":
        # Area scales with tau_half as well as colour, so the persistent pairs
        # are findable at a glance; the key names what a big bubble is.
        ii, jj = np.where(np.isfinite(M))
        vv = M[ii, jj]
        o = np.argsort(vv)                       # longest-lived drawn last
        ii, jj, vv = ii[o], jj[o], vv[o]
        axA.set_facecolor("#f2f2f2")
        im = axA.scatter(jj + 1, ii + 1, s=BK.sizes_for(vv, WIN, F), c=vv,
                         cmap="cividis", vmin=0, vmax=WIN, linewidths=0, zorder=3)
        _kx, _ky = jj + 1, ii + 1
    else:
        im = axA.imshow(np.ma.masked_invalid(sub), origin="lower", cmap=cmap,
                        aspect="auto", extent=[lo, hi, lo, hi], vmin=0, vmax=WIN)
    cb = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.03)
    cb.set_label(f"{TAU} (ns)", fontsize=6.5 * F); cb.ax.tick_params(labelsize=5.5 * F)
    PT.draw_axis_strips(axA, pats, axis="both")
    # Patch names below the tick numbers. Offsets are in POINTS, anchored to the
    # axis line, so they stay clear of the tick labels at any font scale --
    # axes-fraction offsets drifted into the numbers as soon as the type grew.
    tick_pt = 6.0 * F
    row0 = -(tick_pt + 9.0)          # below the tick labels
    rowh = 5.0 * F + 6.0             # one extra row
    # Stagger only on a genuine overlap: width of a ~2-character label converted
    # through the axes width. A flat 9 % of the range pushed P5 onto a second row
    # although P4 and P5 are 20 residues apart and never touched.
    _ax_pt = 0.42 * W * 72.0
    _lab_w = (2.4 * 0.62 * 5.0 * F) / max(_ax_pt, 1.0) * (hi - lo)
    _prev, _row, _staggered = None, 0, False
    for q in pats:
        if q["end"] < lo or q["start"] > hi:
            continue
        mid = (max(q["start"], lo) + min(q["end"], hi)) / 2
        if _prev is not None and (mid - _prev) < _lab_w:
            _row = 1 - _row
        else:
            _row = 0
        if _row:
            _staggered = True
        axA.annotate(q["label"], xy=(mid, 0), xycoords=axA.get_xaxis_transform(),
                     xytext=(0, row0 - _row * rowh), textcoords="offset points",
                     ha="center", va="top", fontsize=5.0 * F,
                     color=q["color"], fontweight="bold", annotation_clip=False)
        _prev = mid

    ticks = [r for r in sorted(NUM) if lo <= r <= hi][::max(1, (hi - lo) // 6)]
    axA.set_xticks(ticks); axA.set_xticklabels([str(NUM[r]) for r in ticks])
    axA.set_yticks(ticks); axA.set_yticklabels([str(NUM[r]) for r in ticks])
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
    if a.panel_a == "bubble":
        BK.draw_size_key(axA, [250, 1000, 2500], WIN, F,
                         BK.emptiest_corner(_kx, _ky, lo, hi),
                         f"{TAU} (ns)", fmt="{:.0f}")
    axA.tick_params(labelsize=6 * F)
    # the axis title has to clear the patch names, including a second row
    # clear whichever patch-name row sits lowest
    _lowest = abs(row0 - (rowh if _staggered else 0)) + 5.0 * F
    axA.set_xlabel(axlab, fontsize=7 * F,
                   labelpad=max(4.0, _lowest - tick_pt - 6.0))
    axA.set_ylabel(axlab, fontsize=7 * F)
    axA.set_title(f"{TAU} of RH5 interaction", fontsize=7.5 * F, pad=8)
    if not CLEAN:
        axA.text(0.99, 0.02, "grey: too few\ncontacts to quote",
                 transform=axA.transAxes, fontsize=4.8 * F, color="0.35",
                 ha="right", va="bottom")
    P.panel_label(axA, "a", dx=-0.17, dy=1.13, fontsize=10 * F)

    # ---- (b) ranked pairs by tau_half
    # Drop pairs whose residues have no wild-type counterpart: on a 4WAT-numbered
    # figure they can only be shown as "K4?" in no patch, which tells a reader
    # nothing. Filtering BEFORE the cut keeps the list at a.top entries.
    rankable = [p for p in quot
                if p["i"] in NUM and p["j"] in NUM
                and p["tau_half_ns"] >= a.min_tau]
    top = sorted(rankable,
                 key=lambda p: (-p["tau_half_ns"], -p["observations"]))[:a.top]
    lab, val, cols, cflag = [], [], [], []
    for p in top:
        li = PT.label_of(p["i"], pats); lj = PT.label_of(p["j"], pats)
        lab.append(f"{li}×{lj}  {show(p['i'])}–{show(p['j'])}")
        val.append(p["tau_half_ns"]); cflag.append(p["tau_half_censored"])
        cols.append(PT.color_of(p["i"], pats))
    y = np.arange(len(lab))[::-1]
    axB.barh(y, val, color=cols, height=0.74)
    for yy, v, c in zip(y, val, cflag):
        # Explicit hex and a slightly larger size: at 4.6 pt Calibri's finer
        # strokes rendered these to nothing in the saved PNG, while the same
        # call was fine in DejaVu.
        axB.text(v + WIN * 0.015, yy, (f"> {v:.0f}" if c else f"{v:.0f}"),
                 va="center", fontsize=5.4 * F, color="#333333", zorder=6)
        if c and not CLEAN:
            axB.annotate("", xy=(v + WIN * 0.10, yy), xytext=(v, yy),
                         arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.7))
    axB.set_yticks(y); axB.set_yticklabels(lab, fontsize=4.8 * F)
    axB.set_xlabel(f"{TAU} (ns)", fontsize=7 * F); axB.tick_params(labelsize=6 * F)
    axB.set_xlim(0, WIN * 1.30)
    axB.set_title("longest-lived pairs", fontsize=7.5 * F)
    if not CLEAN:
        axB.text(0.98, 0.02, "arrow: no decay\nwithin the window",
                 transform=axB.transAxes, fontsize=4.6 * F, color="0.35",
                 ha="right", va="bottom")
    P.panel_label(axB, "b", dx=-0.42, dy=1.13, fontsize=10 * F)

    # ---- (c) per-residue persistence
    xs = np.arange(1, NRES + 1)
    ok = np.isfinite(th)
    axC.bar(xs[ok & ~cen], th[ok & ~cen], width=1.6, color="0.35", lw=0, zorder=2)
    axC.bar(xs[ok & cen], th[ok & cen], width=1.6, color="0.35", lw=0,
            alpha=0.45, zorder=2)
    PT.draw_bands(axC, pats, fontsize=5.4 * F)
    if not CLEAN:
        for r in xs[ok & cen]:
            axC.annotate("", xy=(r, th[r - 1] * 1.14), xytext=(r, th[r - 1]),
                         arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.6))
    order = np.argsort(np.where(ok, th, -1))[::-1][:11]
    for k, i in enumerate(sorted(order)):
        if ok[i]:
            axC.annotate(show(i + 1), (i + 1, th[i]), textcoords="offset points",
                         xytext=(0, 5 + 7 * (k % 2)), fontsize=5.0 * F, ha="center",
                         color="#333333")
    axC.set_xlim(1, NRES); axC.set_ylim(0, WIN * 1.32)
    tk = [r for r in sorted(NUM)][::max(1, len(NUM) // 8)]
    axC.set_xticks(tk); axC.set_xticklabels([str(NUM[r]) for r in tk])
    axC.tick_params(labelsize=6 * F)
    axC.set_xlabel(axlab, fontsize=7 * F)
    axC.set_ylabel(f"{TAU} of engagement (ns)", fontsize=7 * F)
    if not CLEAN:
        axC.text(0.995, 0.96,
                 "faded + arrow: no decay within the observation window",
                 transform=axC.transAxes, fontsize=5.0 * F, color="0.35",
                 va="top", ha="right")
    P.panel_label(axC, "c", dx=-0.062, dy=1.16, fontsize=10 * F)

    fig.text(0.5, 0.008,
             f"{TAU} = lag at which the baseline-corrected contact correlation "
             f"falls to 0.5, pooled over 3 replicates \u00d7 3 chain pairs; "
             f"observation window {WIN:.0f} ns; (b) shows pairs with "
             f"{TAU} \u2265 {a.min_tau:.0f} ns.",
             fontsize=5.0 * F, color="0.35", ha="center", va="center")
    try:
        fig.get_layout_engine().set(rect=(0, 0.035, 1, 1))
    except Exception:
        pass

    os.makedirs(a.outdir, exist_ok=True)
    P.save(fig, os.path.join(a.outdir,
                             f"fig10_duration_{a.rung}_rh5_4wat{a.suffix}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
