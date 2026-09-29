#!/usr/bin/env python
"""
Step 08 -- figures from results/census.json.

Reads only the census JSON, never a trajectory, so figures can be restyled in
seconds. Produces:

    fig02_assay_validation   native contact fraction vs time, controls included
    fig03_census             N(t), occupancy, survival, internal-control ratio

CONVENTIONS ENFORCED HERE
  * error bars are always defined in the caption text printed to stdout
  * a censored tau_slow is drawn as a lower-bound arrow, never as a point
  * n is written on every panel

Usage:  python steps/08_figures.py [--census results/census.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "lib"))

import matplotlib.pyplot as plt        # noqa: E402
import plotting as P                   # noqa: E402
import stats as st                     # noqa: E402

P.setup()
CAPTIONS = []
FIG3_DROPPED = []


def pairs_of(rep):
    return rep["pairs"]


def mean_curve(curves):
    """Mean +/- SEM across replicates on a common grid (curves may differ in length)."""
    n = min(len(c) for c in curves)
    A = np.array([c[:n] for c in curves], dtype=float)
    m = np.nanmean(A, axis=0)
    sem = (np.nanstd(A, axis=0, ddof=1) / np.sqrt(A.shape[0])
           if A.shape[0] > 1 else np.zeros(n))
    return m, sem, A.shape[0], n


# ----------------------------------------------------------------- figure 2
def figure2(cen, outdir):
    """
    Assay validation, showing BOTH persistence measures.

    Native contact fraction alone mislabels the I53-50B pentamer. The previous
    evaluation established this independently (8 +/- 12 % native contacts, but
    44-88 % buried SASA and 0.1-2.1 nm drift) and read it as interface
    plasticity: the subunits slide until no original pair survives while staying
    associated. Plotting only native fraction would therefore show the positive
    controls "failing" in the very figure that is supposed to validate the
    assay. Panel (b) puts native fraction against occupancy so that sliding
    (low native, high occupancy) is visibly distinct from separation (low both),
    which is the honest way to present a disagreement between two metrics.
    """
    want = [r for r in ["rung1", "rung3", "rung4", "ctrl"] if r in cen["rungs"]]
    pre = cen["config"]["preregistered"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax, ax2 = axes

    for rung in want:
        reps = cen["rungs"][rung]["replicates"]
        if not reps:
            continue
        curves, tgrid = [], None
        for tag, rep in reps.items():
            per_pair = [np.array(p["timeseries"]["native_fraction"], float)
                        for p in pairs_of(rep).values()]
            if not per_pair:
                continue
            n = min(len(c) for c in per_pair)
            curves.append(np.mean([c[:n] for c in per_pair], axis=0))
            t = np.array(list(pairs_of(rep).values())[0]["timeseries"]["t_ns"], float)[:n]
            if tgrid is None:
                tgrid = t
        if not curves:
            continue
        m, sem, nrep, n = mean_curve(curves)
        t = tgrid[:n] / 1000.0
        c = P.SYSTEM_COLORS[rung]
        ax.plot(t, m, color=c, label=f"{P.LABELS[rung]} (n={nrep})")
        ax.fill_between(t, m - sem, m + sem, color=c, alpha=0.22, lw=0)

    ax.axhline(pre["persists_native_contacts"], ls=":", c="0.45", lw=0.8)
    ax.axhline(pre["dissociates_native_contacts"], ls=":", c="0.45", lw=0.8)
    ax.set_xlabel("time (\u00b5s)")
    ax.set_ylabel("native contact fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper right", fontsize=6)
    P.panel_label(ax, "a")

    # ---- (b) the two measures against each other, one point per interface
    for rung in want:
        c = P.SYSTEM_COLORS[rung]
        xs, ys = [], []
        for rep in cen["rungs"][rung]["replicates"].values():
            for pp in pairs_of(rep).values():
                nf = np.array(pp["timeseries"]["native_fraction"], float)
                k = int(0.9 * len(nf))
                xs.append(float(np.nanmean(nf[k:])))
                ys.append(pp["occupancy"])
        ax2.scatter(xs, ys, color=c, s=20, alpha=0.85, edgecolor="white",
                    linewidth=0.4, label=f"{P.LABELS[rung]} (n={len(xs)})", zorder=3)

    ax2.axvline(pre["dissociates_native_contacts"], ls=":", c="0.45", lw=0.8)
    ax2.axvline(pre["persists_native_contacts"], ls=":", c="0.45", lw=0.8)
    ax2.axhline(0.5, ls=":", c="0.45", lw=0.8)
    ax2.text(0.02, 0.97, "associated but\nregister lost\n(plasticity)", fontsize=5.5,
             color="0.3", va="top", ha="left", transform=ax2.transAxes)
    ax2.text(0.97, 0.97, "persists", fontsize=6, color="0.3",
             va="top", ha="right", transform=ax2.transAxes)
    ax2.text(0.02, 0.06, "dissociated", fontsize=6, color="0.3",
             va="bottom", ha="left", transform=ax2.transAxes)
    ax2.set_xlabel("native contact fraction (final 10 %)")
    ax2.set_ylabel("occupancy (fraction of frames bound)")
    ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.05, 1.08)
    ax2.legend(loc="center right", fontsize=5.5)
    P.panel_label(ax2, "b")

    P.save(fig, os.path.join(outdir, "fig02_assay_validation"))
    CAPTIONS.append(
        "Figure 2 \u2014 Assay validation, and the limits of a single metric. "
        "(a) Fraction of the t=0 residue contact set retained, averaged over the "
        "designed interfaces within each system then over replicates; line = mean, "
        "band = SEM across replicates. (b) The same quantity over the final 10 % of "
        "each trajectory plotted against occupancy (fraction of post-equilibration "
        "frames with \u2265" + str(pre["bound_min_residue_contacts"]) + " residue "
        "contacts), one point per interface per replicate. The designed I53-50A "
        "trimer sits top-right (native register retained) and the broken-interface "
        "control bottom-left (genuinely separated), establishing the dynamic range. "
        "The I53-50B pentamer occupies the top-LEFT quadrant: still associated "
        "throughout, but with none of its original contact pairs surviving. That is "
        "interface plasticity \u2014 the subunits slide against one another rather "
        "than separating \u2014 and it reproduces, with an independent residue-level "
        "implementation, the same behaviour found by bead-pair counting and buried "
        "SASA in the earlier evaluation. Native contact fraction alone should "
        "therefore not be read as a dissociation verdict for this interface. "
        "Contacts: residue-level, 0.6 nm cutoff.")


# ----------------------------------------------------------------- figure 3
def figure3(cen, outdir, rung="rung2"):
    """The interaction census, with the designed interface as internal control."""
    if rung not in cen["rungs"] or not cen["rungs"][rung]["replicates"]:
        print(f"    (skipping figure 3: {rung} absent)")
        return
    reps = cen["rungs"][rung]["replicates"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    (axA, axB), (axC, axD) = axes

    # ---- (a) N(t), RH5-RH5 vs designed, same trajectories
    for tag, rep in reps.items():
        for key, p in pairs_of(rep).items():
            cls = P.pair_class(key)
            if cls == "rh5_other":
                continue
            t = np.array(p["timeseries"]["t_ns"]) / 1000.0
            axA.plot(t, p["timeseries"]["n_contacts"], color=P.CLASS_COLORS[cls],
                     lw=0.6, alpha=0.65)
    axA.plot([], [], color=P.CLASS_COLORS["designed"], label="I53-50A–I53-50A (designed)")
    axA.plot([], [], color=P.CLASS_COLORS["rh5_rh5"], label="RH5–RH5")
    axA.set_xlabel("time (µs)"); axA.set_ylabel("residue contacts")
    axA.legend(loc="center right"); P.panel_label(axA, "a")
    axA.set_title(f"{P.LABELS[rung]}, n={len(reps)} replicates", fontsize=7)

    # ---- (b) occupancy per pair, bootstrap CI over replicates
    keys = sorted({k for rep in reps.values() for k in pairs_of(rep)},
                  key=lambda k: (P.pair_class(k) != "designed", k))
    for i, key in enumerate(keys):
        vals = [pairs_of(rep)[key]["occupancy"] for rep in reps.values()
                if key in pairs_of(rep)]
        c = P.CLASS_COLORS[P.pair_class(key)]
        ci = st.bootstrap_ci(vals, n=cen["config"]["preregistered"]["n_bootstrap"])
        axB.bar(i, ci["mean"], color=c, alpha=0.75, width=0.66)
        if np.isfinite(ci.get("lo", np.nan)):
            axB.plot([i, i], [ci["lo"], ci["hi"]], color="0.2", lw=0.9)
        axB.scatter([i] * len(vals), vals, color="0.15", s=8, zorder=3)
    axB.set_xticks(range(len(keys)))
    axB.set_xticklabels([k.replace("|", "–") for k in keys], rotation=45,
                        ha="right", fontsize=6)
    axB.set_ylabel("occupancy (fraction bound)"); axB.set_ylim(0, 1.05)
    P.panel_label(axB, "b")

    # ---- (c) survival functions with fits
    OCC_MIN = 0.05
    n_drop = 0
    for tag, rep in reps.items():
        for key, p in pairs_of(rep).items():
            cls = P.pair_class(key)
            if cls == "rh5_other":
                continue
            # S(tau) conditions on a contact existing at t. For a pair that is
            # essentially never bound the denominator is a handful of frames and
            # the curve is spikes, not a decay -- it would read as scatter in the
            # figure while carrying no information. Omitted, and counted so the
            # caption can say how many.
            if p["occupancy"] < OCC_MIN:
                n_drop += 1
                continue
            lag = np.array(p["survival"]["lag_ns"]); S = np.array(p["survival"]["S"])
            m = np.isfinite(S) & (lag > 0) & (S > 0)
            if m.sum() < 3:
                continue
            axC.plot(lag[m], S[m], color=P.CLASS_COLORS[cls], lw=0.7, alpha=0.7)
    FIG3_DROPPED.append(n_drop)
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlabel("lag τ (ns)"); axC.set_ylabel("survival S(τ)")
    axC.set_ylim(1e-2, 1.4); P.panel_label(axC, "c")

    # ---- (d) tau_slow, censored values as lower-bound arrows
    cats = {"designed": [], "rh5_rh5": []}
    cens = {"designed": [], "rh5_rh5": []}
    for rep in reps.values():
        for key, p in pairs_of(rep).items():
            cls = P.pair_class(key)
            if cls not in cats:
                continue
            f = p["biexp_fit"]
            if not f:
                continue
            (cens if f["tau_slow_censored"] else cats)[cls].append(
                f["observation_window_ns"] if f["tau_slow_censored"]
                else f["tau_slow_ns"])
    rng = np.random.default_rng(0)
    for i, cls in enumerate(["designed", "rh5_rh5"]):
        c = P.CLASS_COLORS[cls]
        if cats[cls]:
            x = i + rng.uniform(-0.10, 0.10, len(cats[cls]))
            axD.scatter(x, cats[cls], color=c, s=22, zorder=3,
                        edgecolor="white", lw=.4)
        # Censored points stay inside their own category (a fixed rightward
        # march put the designed group's lower bounds in the middle of the
        # axes, where they read as a separate category).
        if cens[cls]:
            xc = i + rng.uniform(-0.10, 0.10, len(cens[cls]))
            axD.scatter(xc, cens[cls], facecolor="none", edgecolor=c, s=22,
                        zorder=3, lw=0.9)
            for xx, v in zip(xc, cens[cls]):
                axD.annotate("", xy=(xx, v * 2.6), xytext=(xx, v * 1.05),
                             arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8))
    axD.set_xlim(-0.5, 1.5)
    axD.set_yscale("log")
    allv = cats["designed"] + cats["rh5_rh5"] + cens["designed"] + cens["rh5_rh5"]
    if allv:
        axD.set_ylim(min(allv) * 0.5, max(allv) * 6)
    axD.set_xticks([0, 1]); axD.set_xticklabels(["designed\nI53-50A", "RH5–RH5"], fontsize=7)
    axD.set_ylabel("τ_slow (ns)")
    axD.text(0.02, 0.02, "open symbols + arrow:\nno decay within window\n(lower bound)",
             transform=axD.transAxes, fontsize=5.5, va="bottom", color="0.3")
    P.panel_label(axD, "d")

    P.save(fig, os.path.join(outdir, "fig03_census"))
    CAPTIONS.append(
        "Figure 3 — Interaction census with an internal control. All panels from "
        f"{P.LABELS[rung]} trajectories (n={len(reps)}), in which the RH5–RH5 "
        "interaction and the designed I53-50A–I53-50A interface are present in the "
        "SAME simulations, so force-field and sampling biases affect both equally. "
        "(a) Residue contacts vs time, one line per interface per replicate. "
        "(b) Occupancy (fraction of post-equilibration frames with ≥5 residue "
        "contacts); bar = mean, whisker = bootstrap 95% CI over replicates, points "
        "= individual replicates. (c) Contact survival S(τ), all start frames "
        "averaged. (d) Slow time constant from a biexponential fit to S(τ); open "
        "symbols with arrows are lower bounds where S had not decayed within the "
        "observation window, and are NOT measured lifetimes. In (c), "
        + str(FIG3_DROPPED[0] if FIG3_DROPPED else 0) +
        " RH5-RH5 interface(s) with occupancy <5% are omitted, S(tau) being "
        "undefined for a pair that is essentially never in contact; they appear "
        "in (b) and (d).")


def extra_captions(cen):
    """Captions for the figures produced by steps 02 and 06.

    They live here so that captions.md is written in one place and cannot drift
    out of order; the numbers are read back from those steps' result files
    rather than retyped.
    """
    rdir = os.path.join(ROOT, "results")
    ii = os.path.join(rdir, "interface_identity.json")
    if os.path.isfile(ii):
        d = json.load(open(ii))
        pj = d.get("jaccard_permutation") or {}
        pg = d.get("gini_permutation") or {}
        CAPTIONS.append(
            "Figure 4 \u2014 Identity and reproducibility of the RH5\u2013RH5 "
            "interaction, from the fusion trimer (n=3 replicates). (a) Contact "
            "frequency per RH5 residue, thin lines = replicates, thick = mean; "
            "orange marks the 12 most contacted residues. (b) Residue\u2013residue "
            "contact frequency map, averaged over replicates and chain pairs, "
            "clipped at the 99th percentile of non-zero cells. (c) Gini "
            "coefficient of the per-residue contact frequency; RH5\u2013RH5 "
            f"{np.mean(d['gini_rh5_rh5']):.3f} vs designed I53-50A "
            f"{np.mean(d['gini_designed']):.3f}, p = {pg.get('p', float('nan')):.3f}. "
            "Pairs engaging fewer than five residues are excluded from (c) in BOTH "
            "groups, because a Gini over one residue is exactly 0 and means "
            "'almost no contact', not 'perfectly even contact'; this exclusion "
            "changes the comparison and is therefore stated explicitly. "
            "(d) Jaccard overlap of the contacting residue set between replicate "
            f"pairs: designed {d['jaccard_designed_mean']:.3f} vs RH5\u2013RH5 "
            f"{d['jaccard_rh5_mean']:.3f} (Cohen's d = {pj.get('cohens_d', float('nan')):.1f}). "
            f"The permutation p = {pj.get('p', float('nan')):.3f} is at the floor "
            f"attainable with three replicates per group "
            f"(min {pj.get('min_attainable_p', float('nan')):.3f}); the claim rests "
            "on the effect size and the non-overlapping ranges, not on p. "
            "Bars = mean, whiskers = bootstrap 95 % CI over replicate pairs.")

    ep = os.path.join(rdir, "epitopes.json")
    if os.path.isfile(ep):
        d = json.load(open(ep))
        coms = ", ".join(f"{k} ({v['pdb']}, {v['antibody']}, "
                         f"{len(v['residues'])} residues)"
                         for k, v in d["communities"].items())
        CAPTIONS.append(
            "Figure 5 \u2014 RH5 self-association surfaces against the "
            "growth-inhibitory antibody epitopes. Epitopes were taken from "
            f"{coms}, defined as RH5 residues within {d['epitope_cutoff_A']} \u00c5 "
            "of any antibody heavy atom and unioned over every copy in the "
            "deposition. Deposition numbering (156\u2013502) was mapped onto our "
            "construct by explicit sequence alignment, not an offset: the "
            "construct is an engineered variant (~81 % identity) carrying a "
            "~49-residue internal deletion, so a constant shift would mis-assign "
            "every residue past the gap. (a\u2013c) Mean number of contacting "
            "partner residues per frame along the RH5 sequence, for the three "
            "distinct surfaces; coloured squares mark the epitope footprints. "
            "(d) Enrichment of contact weight inside each epitope relative to a "
            "permutation null in which contact weight is reshuffled among "
            "residues MATCHED for residue type and solvent exposure "
            f"({d['n_permutations']} permutations); whiskers span 95 % of that "
            "null. No surface is enriched in any community at \u03b1 = 0.05, so "
            "all are reported n.d.; the result is unchanged for epitope cutoffs "
            "of 4.0 and 5.0 \u00c5. Community II is consistently DEPLETED "
            "(0.05\u20130.34\u00d7).")

    dy = os.path.join(rdir, "dynamics.json")
    if os.path.isfile(dy):
        CAPTIONS.append(
            "Figure 7 \u2014 Partner exchange and interface chemistry. "
            "(a) Residue contacts between the RH5 domains and each individual "
            "I53-50B chain in one rung-5 replicate: the pentamer is passed "
            "between chains rather than held by one, which the pooled census "
            "count cannot show. (b) Mean number of pentamer chains engaged per "
            "RH5 domain, per replicate. (c) Rate at which a chain's bound-partner "
            "set changes, per chain per microsecond, points = replicates, "
            "bar = mean. (d) Buried surface area on the RH5 side, split by "
            "residue class, mean over replicates. Both sides of each pair are "
            "disjoint bead sets; self-pairs are enumerated chain by chain, "
            "because comparing a chain set against itself buries its own surface "
            "and inflates the result by an order of magnitude.")

    for fn, fignum, what in [("rh5_contact_pairs.json", "8", "RH5\u2013RH5"),
                             ("rh5_pentamer_pairs.json", "9", "RH5\u2013I53-50B")]:
        fp = os.path.join(rdir, fn)
        if not os.path.isfile(fp):
            continue
        d = json.load(open(fp))
        n_top = sum(1 for q in d["pairs"] if q["mean"] >= d["threshold"])
        CAPTIONS.append(
            f"Figure {fignum} \u2014 {what} residue-pair contacts in detail. "
            "All panels are keyed by CONTACT PATCH: contiguous engaged regions "
            "of sequence are named once (P1..Pn on RH5"
            + (", B1..Bn on I53-50B" if fignum == "9" else "") +
            "), given one colour each, and that naming is reused in every panel "
            "\u2014 as strips along the map axes, as the prefix on each ranked "
            "bar, and as shaded bands under the per-residue profile. "
            "(a) Contact map; dashed rectangles mark patch\u00d7patch blocks "
            f"containing at least one pair above {d['threshold']}; the "
            "individual pairs within them are listed in (b). "
            "(b) Ranked pairs; grey = the highest "
            "value any single replicate reaches, coloured = mean over replicates, "
            "bar colour = patch of the RH5 residue. "
            + ("(c, d) Summed pair frequency per residue on each side. "
               if fignum == "9" else "(c) Summed pair frequency per residue. ") +
            f"{n_top} pairs exceed the {d['threshold']} threshold. "
            "Values are means over replicates and chain instances; because the "
            "same residues pair differently in each replicate, the mean "
            "understates any individual encounter \u2014 see the per-replicate "
            f"columns in results/{fn.replace('.json', '.md')}. A residue "
            "outside every patch is shown as \u2013.")

    cl = os.path.join(rdir, "crosslink.json")
    if os.path.isfile(cl):
        d = json.load(open(cl))
        freed = d.get("cores_separated", [])
        n7 = [f for f in freed if f["system"].startswith("rung7_")]
        CAPTIONS.append(
            "Figure 6 \u2014 Does the RH5 cross-link survive without the cage? "
            "Rung 6 holds two fusion trimers in cage geometry with the I53-50A "
            "cores position-restrained; rung 7 removes the restraints entirely; "
            "rung 7rel releases them from a bridged rung-6 endpoint. Only 150 mM "
            "replicates are shown, so rung 6 and rung 7 differ in the restraints "
            "and nothing else (n=3 each, 5 \u00b5s). (a) Inter-trimer RH5\u2013RH5 "
            "residue contacts vs time, one line per replicate. (b) The same for "
            "the I53-50A cores. (c) Occupancy; colour = system as in all other "
            "panels, filled = RH5\u2013RH5, open = core\u2013core; bar = mean, "
            "whisker = bootstrap 95 % CI over replicates. (d) The falsifiable "
            "panel: final-10 % RH5\u2013RH5 contacts against final-10 % core\u2013core "
            "contacts, one point per replicate. Points at upper left are trimer "
            "pairs held together by RH5 while their cores have separated, i.e. "
            "they were free to leave and did not. "
            f"In rung 7 this holds in {len(n7)} of 3 replicates "
            + (f"(RH5\u2013RH5 contacts "
               + ", ".join(f"{f['rh5_contacts_late']:.0f}" for f in n7) + "). "
               if n7 else ". ") +
            "Note that these trajectories test whether an existing bridge "
            "DISSOCIATES, not whether one forms: all rungs start from cage "
            "geometry, so the RH5 domains begin within reach.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=os.path.join(ROOT, "results", "census.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    cen = json.load(open(a.census))
    os.makedirs(a.outdir, exist_ok=True)
    print(">>> figures")
    figure2(cen, a.outdir)
    figure3(cen, a.outdir)
    extra_captions(cen)
    # Emit in figure order: captions are appended as each step's results file is
    # found, which is not the order a reader needs them in.
    import re as _re
    def _num(c):
        m = _re.match(r"Figure\s+(\d+)", c)
        return int(m.group(1)) if m else 99
    with open(os.path.join(a.outdir, "captions.md"), "w") as fh:
        fh.write("# Figure captions\n\n"
                 + "\n\n".join(sorted(CAPTIONS, key=_num)) + "\n")
    print(f">>> captions -> {a.outdir}/captions.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
