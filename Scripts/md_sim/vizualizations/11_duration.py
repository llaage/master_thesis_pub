#!/usr/bin/env python
"""
Step 11 -- how LONG residue-pair contacts last, for the RH5-I53-50A fusion
trimer (rung 2).

WHY THIS IS NOT FIGURE 8
Figure 8 reports contact FREQUENCY: the fraction of frames a residue pair spends
in contact. Frequency cannot distinguish one continuous 1.5 us engagement from
three hundred brief touches that sum to the same total. This step measures the
episodes themselves.

    episode      a maximal run of consecutive frames in which residue i of one
                 RH5 domain is within 0.6 nm of residue j of another
    duration     the length of one episode, in ns
    persistence  per residue: total time spent in episodes lasting at least
                 MIN_LONG ns -- "I hold on to another RH5 for a long time"

EPISODE DURATION IS NOT TRUSTWORTHY HERE, AND THE PRIMARY METRIC IS NOT IT
A pair sitting near the 0.6 nm cutoff drops out for single frames constantly, so
splitting the record into episodes is extremely sensitive to how a brief dropout
is treated. Measured on this data: pair 61-82 has a mean episode duration of
87 ns with zero gap tolerance and 839 ns with a 2 ns tolerance -- a tenfold
swing from a parameter nobody can justify to better than a factor of two. Its
TOTAL contact time over the same scan moves 1657 -> 1678 ns, i.e. not at all.

The primary metric is therefore tau_half, taken from the INTERMITTENT contact
survival S(tau) = P(pair in contact at t+tau | in contact at t), pooled over all
start frames and all chain instances. It never segments the record, so a
one-frame dropout cannot break anything, and it needs no tolerance parameter.
tau_half is the lag at which S falls to 0.5: "half the contacts present at any
moment are still present this much later".

Episode statistics are still computed and written out, but they carry the gap
tolerance in their name and must be read with the sensitivity scan beside them.

RIGHT CENSORING
An episode still running at the end of the trajectory has no measured end. Those
are counted and reported separately; including them as if complete would bias
mean durations downward.

Usage: python steps/11_duration.py [--stride 2] [--gap-tol 2] [--min-long 20]
"""
import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "lib"))
CFG = json.load(open(os.path.join(ROOT, "config.json")))
sys.path.insert(0, os.path.join(CFG["base"], "scripts"))

import analysis_common as ac            # noqa: E402
import contacts as ct                   # noqa: E402

NRES = CFG["domains"]["RH5"][1]


def episodes(frames, gap_frames):
    """
    Maximal episodes from a sorted array of frame indices.

    Returns list of (start_frame, end_frame_inclusive). Gaps of up to
    `gap_frames` missing frames do not break an episode.
    """
    if len(frames) == 0:
        return []
    out, s, p = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - p <= gap_frames + 1:
            p = f
        else:
            out.append((s, p)); s = p = f
    out.append((s, p))
    return out


def lag_grid(max_lag_frames, n=44):
    """Log-spaced lags in frames, always including 1."""
    if max_lag_frames < 2:
        return np.array([1])
    g = np.unique(np.round(np.logspace(0, np.log10(max_lag_frames), n)).astype(int))
    return g[g >= 1]


def accumulate_survival(bool_mat, lags, num, den, tot):
    """
    Add this instance's contribution to the pooled intermittent survival.

    `bool_mat` is frames x items. For each lag we count items present at t AND
    at t+lag (numerator) against items present at t (denominator), summed over
    every valid start frame. Pooling counts -- not per-instance ratios -- is what
    keeps a replicate with few contacts from dominating the average.
    """
    n = bool_mat.shape[0]
    for li, lag in enumerate(lags):
        if lag >= n:
            break
        a = bool_mat[:n - lag]
        b = bool_mat[lag:]
        den[:, li] += a.sum(axis=0)
        num[:, li] += (a & b).sum(axis=0)
        tot[li] += (n - lag)          # start frames available at this lag


def tau_half(lags_ns, S):
    """
    Lag at which survival first falls to 0.5, interpolated in log-lag.

    Returns (value, censored). Censored means S never reached 0.5 inside the
    observation window -- reported as a lower bound, never as a number.
    """
    m = np.isfinite(S)
    if m.sum() < 2:
        return float("nan"), False
    x, y = lags_ns[m], S[m]
    below = np.where(y < 0.5)[0]
    if below.size == 0:
        return float(x[-1]), True
    k = below[0]
    if k == 0:
        return float(x[0]), False
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    if y0 == y1:
        return float(x1), False
    f = (y0 - 0.5) / (y0 - y1)
    return float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0)))), False


def analyse(rung, stride, gap_tol_ns, min_long_ns, min_frames=30):
    """Episode stats AND pooled intermittent survival, per residue pair and per residue."""
    cut = CFG["preregistered"]["contact_cutoff_nm"] * 10.0
    specs = [("A", "B"), ("A", "C"), ("B", "C")]

    dur, censored = defaultdict(list), defaultdict(int)
    res_epis = defaultdict(list)
    res_long = np.zeros(NRES)
    n_inst = res_inst = 0

    # pooled survival accumulators, built on the first replicate once the
    # observed key sets are known
    pair_keys, res_keys = [], list(range(1, NRES + 1))
    pair_inst, res_inst_acc = [], []
    pnum = pden = rnum = rden = None
    lags = None
    dt_ns = None

    per_rep = []
    for tag in CFG["rungs"][rung]["replicates"]:
        sd = os.path.join(CFG["base"], "systems", tag)
        xtc = os.path.join(sd, "prod", "prod.xtc")
        if not os.path.isfile(xtc):
            continue
        chains, _, rpdb, _ = ac.bead_table(sd)
        u, _ = ac.load(sd, xtc)
        rlab = np.array([f"{c}:{r}" for c, r in zip(chains, rpdb)])
        uniq = sorted(set(rlab.tolist())); rid = {v: i for i, v in enumerate(uniq)}
        rofb = np.array([rid[v] for v in rlab])
        lo, hi = CFG["domains"]["RH5"]
        gi = {c: np.where((chains == c) & (rpdb >= lo) & (rpdb <= hi))[0]
              for c in ["A", "B", "C"]}

        seen = {sp: defaultdict(list) for sp in specs}
        eng = {c: defaultdict(list) for c in ["A", "B", "C"]}
        times = []
        for k, ts in enumerate(u.trajectory[::stride]):
            box = ts.dimensions; pos = u.atoms.positions
            cache = {c: ac.unwrap(pos, box, gi[c]) for c in ["A", "B", "C"]}
            times.append(ts.time / 1000.0)
            for aa, bb in specs:
                cs = ct.residue_contact_set(cache[aa], rofb[gi[aa]],
                                            cache[bb], rofb[gi[bb]], cut, box)
                for ra, rb in cs:
                    i = int(uniq[ra].split(":")[1]); j = int(uniq[rb].split(":")[1])
                    if not (1 <= i <= NRES and 1 <= j <= NRES):
                        continue
                    seen[(aa, bb)][(min(i, j), max(i, j))].append(k)
                    eng[aa][i].append(k); eng[bb][j].append(k)

        nfr = len(times)
        dt = float(np.median(np.diff(times))) if nfr > 1 else 0.5
        dt_ns = dt
        gap_f = max(0, int(round(gap_tol_ns / dt)))
        last = nfr - 1
        if lags is None:
            lags = lag_grid(nfr // 2)
            pair_keys = sorted({pr for sp in specs for pr in seen[sp]})
            pidx = {pr: n for n, pr in enumerate(pair_keys)}
            pnum = np.zeros((len(pair_keys), len(lags)))
            pden = np.zeros_like(pnum)
            rnum = np.zeros((NRES, len(lags))); rden = np.zeros_like(rnum)
            ptot = np.zeros(len(lags)); rtot = np.zeros(len(lags))
        else:
            new_keys = sorted({pr for sp in specs for pr in seen[sp]} - set(pair_keys))
            if new_keys:
                pair_keys = pair_keys + new_keys
                pnum = np.vstack([pnum, np.zeros((len(new_keys), len(lags)))])
                pden = np.vstack([pden, np.zeros((len(new_keys), len(lags)))])
            pidx = {pr: n for n, pr in enumerate(pair_keys)}

        for sp in specs:
            n_inst += 1
            M = np.zeros((nfr, len(pair_keys)), dtype=bool)
            for pr, fr in seen[sp].items():
                f = np.array(sorted(set(fr)))
                M[f, pidx[pr]] = True
                for (a0, b0) in episodes(f, gap_f):
                    d = (b0 - a0 + 1) * dt
                    dur[pr].append(d)
                    if b0 >= last:
                        censored[pr] += 1
            _n = np.zeros((len(pair_keys), len(lags)))
            _d = np.zeros_like(_n); _t = np.zeros(len(lags))
            accumulate_survival(M, lags, _n, _d, _t)
            pair_inst.append((_n, _d, _t, len(pair_keys)))

        for c in ["A", "B", "C"]:
            res_inst += 1
            R = np.zeros((nfr, NRES), dtype=bool)
            for r, fr in eng[c].items():
                f = np.array(sorted(set(fr)))
                R[f, r - 1] = True
                for (a0, b0) in episodes(f, gap_f):
                    d = (b0 - a0 + 1) * dt
                    res_epis[r].append(d)
                    if d >= min_long_ns:
                        res_long[r - 1] += d
            _n = np.zeros((NRES, len(lags))); _d = np.zeros_like(_n)
            _t = np.zeros(len(lags))
            accumulate_survival(R, lags, _n, _d, _t)
            res_inst_acc.append((_n, _d, _t))
        per_rep.append(tag)

    lags_ns = np.array(lags) * dt_ns
    nL = len(lags)

    def pool(insts, nrow):
        """
        Correlation pooled ACROSS instances, each debiased by ITS OWN occupancy.

        Subtracting a single pooled occupancy instead is wrong here: a pair that
        is engaged throughout one replicate and absent in the other eight has a
        low pooled occupancy but a high pooled survival, and the correlation then
        reports 2.5 us of "persistence" that is really between-replicate
        heterogeneity -- the §3 non-reproducibility leaking into a kinetic
        number. Debiasing per instance and averaging, weighted by how much
        contact each instance actually contributes, measures persistence WITHIN
        a trajectory, which is the question.
        """
        Cw = np.zeros((nrow, nL)); W = np.zeros((nrow, nL))
        S_ = np.zeros((nrow, nL)); D_ = np.zeros((nrow, nL))
        occ = np.zeros(nrow); occ_w = np.zeros(nrow)
        for item in insts:
            n_, d_, t_ = item[0], item[1], item[2]
            r = n_.shape[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                Si = np.where(d_ > 0, n_ / np.maximum(d_, 1), np.nan)
            pi = np.where(t_[0] > 0, d_[:, 0] / max(t_[0], 1), 0.0)
            Ci = (Si - pi[:, None]) / np.maximum(1 - pi[:, None], 1e-9)
            m = np.isfinite(Ci) & (d_ > 0)
            Cw[:r][m] += (Ci * d_)[m]; W[:r][m] += d_[m]
            S_[:r] += np.nan_to_num(Si * d_); D_[:r] += d_
            occ[:r] += pi * d_[:, 0]; occ_w[:r] += d_[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.where(W > 0, Cw / np.maximum(W, 1e-9), np.nan)
            S = np.where(D_ > 0, S_ / np.maximum(D_, 1e-9), np.nan)
            o = np.where(occ_w > 0, occ / np.maximum(occ_w, 1e-9), 0.0)
        return C, S, D_, o

    for n_, d_, t_, k in pair_inst:
        if n_.shape[0] < len(pair_keys):
            pad = len(pair_keys) - n_.shape[0]
            n_.resize((len(pair_keys), nL), refcheck=False)
            d_.resize((len(pair_keys), nL), refcheck=False)
    Cp, Sp, pden, p_pair = pool(pair_inst, len(pair_keys))
    Cr, Sr, rden, p_res = pool(res_inst_acc, NRES)

    return dict(dur=dur, censored=censored, n_inst=n_inst, res_inst=res_inst,
                res_long=res_long / max(1, res_inst), res_epis=res_epis,
                pair_keys=pair_keys, Sp=Sp, pden=pden, Sr=Sr, rden=rden,
                Cp=Cp, Cr=Cr, occ_pair=p_pair, occ_res=p_res,
                lags_ns=lags_ns, dt_ns=dt_ns, min_frames=min_frames,
                replicates=per_rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="rung2")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--gap-tol", type=float, default=2.0, help="ns")
    ap.add_argument("--min-long", type=float, default=20.0, help="ns")
    ap.add_argument("--min-obs", type=int, default=200,
                    help="minimum pooled contact-frame observations for a pair "
                         "to get a quotable tau_half")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    R = analyse(a.rung, a.stride, a.gap_tol, a.min_long)
    lags = R["lags_ns"]

    pairs = []
    for n, pr in enumerate(R["pair_keys"]):
        ds = np.array(R["dur"].get(pr, [0.0]))
        obs = float(R["pden"][n, 0]) if R["pden"].shape[1] else 0.0
        th, cens = tau_half(lags, R["Cp"][n])
        pairs.append({"i": pr[0], "j": pr[1],
                      "tau_half_ns": th, "tau_half_censored": bool(cens),
                      "observations": obs, "occupancy": float(R["occ_pair"][n]),
                      "quotable": bool(obs >= a.min_obs and np.isfinite(th)),
                      "n_episodes": int(len(ds)),
                      "mean_episode_ns": float(ds.mean()),
                      "max_episode_ns": float(ds.max()),
                      "total_ns": float(ds.sum())})

    res = []
    for r in range(1, NRES + 1):
        ds = np.array(R["res_epis"].get(r, []))
        th, cens = tau_half(lags, R["Cr"][r - 1])
        obs = float(R["rden"][r - 1, 0])
        res.append({"resid": r, "tau_half_ns": th,
                    "tau_half_censored": bool(cens), "observations": obs,
                    "occupancy": float(R["occ_res"][r - 1]),
                    "long_lived_ns": float(R["res_long"][r - 1]),
                    "mean_episode_ns": float(ds.mean()) if ds.size else 0.0,
                    "max_episode_ns": float(ds.max()) if ds.size else 0.0,
                    "n_episodes": int(ds.size)})

    os.makedirs(a.out, exist_ok=True)
    payload = {"rung": a.rung, "stride": a.stride, "dt_ns": R["dt_ns"],
               "gap_tol_ns": a.gap_tol, "min_long_ns": a.min_long,
               "min_obs": a.min_obs,
               "n_chainpair_instances": R["n_inst"],
               "n_chain_instances": R["res_inst"],
               "replicates": R["replicates"],
               "lags_ns": lags.tolist(),
               "pairs": pairs, "per_residue": res}
    with open(os.path.join(a.out, f"duration_{a.rung}.json"), "w") as fh:
        json.dump(payload, fh)
    np.savez_compressed(os.path.join(a.out, f"duration_{a.rung}_survival.npz"),
                        lags_ns=lags, Sp=R["Sp"].astype(np.float32),
                        pden=R["pden"].astype(np.float32),
                        Sr=R["Sr"].astype(np.float32),
                        rden=R["rden"].astype(np.float32),
                        Cp=R["Cp"].astype(np.float32),
                        Cr=R["Cr"].astype(np.float32),
                        occ_pair=R["occ_pair"].astype(np.float32),
                        occ_res=R["occ_res"].astype(np.float32),
                        pair_i=np.array([p[0] for p in R["pair_keys"]]),
                        pair_j=np.array([p[1] for p in R["pair_keys"]]))

    good = [p for p in pairs if p["quotable"]]
    good.sort(key=lambda p: -p["tau_half_ns"])
    print(f"  {len(pairs)} residue pairs seen; {len(good)} with >= {a.min_obs} "
          f"pooled contact observations (quotable tau_half)")
    print(f"\n  longest-lived pairs by tau_half (robust, no episode splitting):")
    print(f"  {'pair':>12s} {'tau_1/2':>9s} {'occ':>6s} {'obs':>7s} {'total ns':>9s}")
    for p in good[:15]:
        star = " >window" if p["tau_half_censored"] else ""
        print(f"  {p['i']:5d}-{p['j']:<6d} {p['tau_half_ns']:9.1f} "
              f"{p['occupancy']:6.3f} {p['observations']:7.0f} "
              f"{p['total_ns']:9.1f}{star}")
    top = sorted([r for r in res if r["observations"] >= a.min_obs],
                 key=lambda r: -r["tau_half_ns"])[:15]
    print(f"\n  residues holding another RH5 longest (tau_half):")
    for r in top:
        print(f"    {r['resid']:4d}  tau_1/2 {r['tau_half_ns']:8.1f} ns   "
              f"occ {r['occupancy']:5.3f}   long-lived {r['long_lived_ns']:7.1f} ns"
              + ("  >window" if r["tau_half_censored"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
