#!/usr/bin/env python
"""
Tyrosine control: is Y61/Y64 enrichment specific, or does Martini make every
exposed tyrosine sticky?

THE PROBLEM
-----------
Five of the top seven persistent-contact residues are tyrosines, and the 55-64
patch enriches ~2.8x. Two explanations fit that equally well:

  H1  Y61/Y64 form a specific interface.
  H0  Martini 3 makes every solvent-exposed Tyr sticky, and 55-64 simply
      contains two of the protein's 21 tyrosines.

An enrichment ratio against a UNIFORM null cannot separate these -- and a uniform
null is the wrong null, which is why 2.8x looks more impressive than it is.

TWO CONFOUNDS
-------------
  type      Tyr may be sticky as a class in this force field.
  exposure  a buried residue cannot make intermolecular contacts at all, so
            contact frequency tracks accessible surface trivially. Y61 might
            rank high merely by being more exposed than the other tyrosines.

The null therefore preserves BOTH: residues are only compared against other
residues of the same type in the same exposure band.

Exposure is computed on the isolated monomer, so it is a fixed property of the
protein rather than something entangled with the contacts being tested.

The flexible C-terminal tail is excluded: it tops the raw contact list by
sweeping through space, not by forming an interface, and would inflate the null.

Usage:
    python tyrosine_control.py <run_dir> [--min-lifetime 50] [--stride 5]
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_common as ac
from analyze_contacts import bead_to_residue
from analyze_persistence import contact_matrix_series
from analyze_contacts_persistent import persistent_frames

THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def residue_types(pdb_path):
    """{pdb_resid: one-letter} from the cleaned all-atom PDB."""
    out = {}
    for line in open(pdb_path):
        if line.startswith("ATOM"):
            out[int(line[22:26])] = THREE2ONE.get(line[17:20].strip(), "X")
    return out


def cg_exposure(cg_pdb, b2r, n_res, probe_nm=0.26, n_sphere=200):
    """
    Per-residue solvent-accessible area of the isolated CG monomer.

    Shrake-Rupley with a MARTINI-sized probe. Using the atomistic 1.4 A probe on
    0.47 nm beads reports almost everything as buried, which would silently
    flatten the exposure correction into no correction at all.
    """
    xyz, names = [], []
    for line in open(cg_pdb):
        if line.startswith("ATOM"):
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            names.append(line[12:16].strip())
    xyz = np.array(xyz) / 10.0                       # A -> nm
    bead_r = 0.235                                   # ~half sigma of a regular bead
    R = bead_r + probe_nm

    # golden-spiral points on the unit sphere
    idx = np.arange(n_sphere) + 0.5
    phi = np.arccos(1 - 2 * idx / n_sphere)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    sphere = np.stack([np.cos(theta) * np.sin(phi),
                       np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)

    tree = cKDTree(xyz)
    area = np.zeros(len(xyz))
    for i, c in enumerate(xyz):
        pts = c + R * sphere
        neigh = [j for j in tree.query_ball_point(c, 2 * R) if j != i]
        if neigh:
            d = np.linalg.norm(pts[:, None, :] - xyz[neigh][None, :, :], axis=-1)
            acc = (d > R).all(axis=1)
        else:
            acc = np.ones(len(pts), dtype=bool)
        area[i] = 4 * np.pi * R ** 2 * acc.mean()

    per_res = np.zeros(n_res)
    for i, a in enumerate(area):
        per_res[b2r[i] - 1] += a
    return per_res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--traj", default="prod.xtc")
    ap.add_argument("--cutoff", type=float, default=0.6)
    ap.add_argument("--min-contacts", type=int, default=5)
    ap.add_argument("--min-lifetime", type=float, default=50.0)
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--skip-ns", type=float, default=200.0)
    ap.add_argument("--exclude-tail", type=int, default=310,
                    help="drop PDB residues >= this (flexible C-terminus)")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--targets", default="61,64")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run = os.path.abspath(args.run_dir)
    system_dir = os.path.dirname(run)
    offset = ac.resid_offset(system_dir)

    u, n_copies, n_beads = ac.load(system_dir, os.path.join(run, args.traj))
    m2 = os.path.join(system_dir, "martinize2")
    itp = [f for f in os.listdir(m2) if f.endswith(".itp")][0]
    cgpdb = [f for f in os.listdir(m2) if f.endswith("_cg.pdb")][0]
    b2r = bead_to_residue(os.path.join(m2, itp))
    n_res = b2r.max()

    # --- persistent contact map ---------------------------------------------
    S, times, pair_index = contact_matrix_series(
        u, n_copies, n_beads, args.cutoff * 10.0, args.min_contacts,
        args.stride, args.skip_ns)
    dt = float(np.median(np.diff(times)))
    keep = persistent_frames(S, dt, args.min_lifetime, args.tolerance)

    per_res = np.zeros(n_res)
    copy_id = np.arange(n_copies * n_beads) // n_beads
    res_id = np.tile(b2r, n_copies) - 1
    fi = -1
    for ts in u.trajectory[::args.stride]:
        if ts.time / 1000.0 < args.skip_ns:
            continue
        fi += 1
        box = ts.dimensions[:3]
        pos = np.mod(u.atoms.positions[:n_copies * n_beads], box)
        pairs = cKDTree(pos, boxsize=box).query_pairs(r=args.cutoff * 10.0,
                                                      output_type="ndarray")
        if not len(pairs):
            continue
        ci, cj = copy_id[pairs[:, 0]], copy_id[pairs[:, 1]]
        m = ci != cj
        if not m.any():
            continue
        pp, ci, cj = pairs[m], ci[m], cj[m]
        lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
        for k in range(len(pp)):
            if fi in keep.get(pair_index[(int(lo[k]), int(hi[k]))], ()):
                per_res[res_id[pp[k, 0]]] += 1
                per_res[res_id[pp[k, 1]]] += 1

    # --- exposure + types ----------------------------------------------------
    expo = cg_exposure(os.path.join(m2, cgpdb), b2r, n_res)
    types = residue_types(os.path.join(system_dir, "inputs",
                                       [f for f in os.listdir(os.path.join(system_dir, "inputs"))
                                        if f.endswith("_cleaned.pdb")][0]))

    pdb_ids = np.arange(n_res) + 1 + offset
    live = (pdb_ids < args.exclude_tail) & (expo > 0)
    tchar = np.array([types.get(int(p), "X") for p in pdb_ids])

    # contacts per unit exposed area -- removes the trivial exposure effect
    dens = np.where(expo > 0, per_res / np.maximum(expo, 1e-9), 0.0)

    tyr = live & (tchar == "Y")
    phe = live & (tchar == "F")
    targets = [int(t) for t in args.targets.split(",")]

    print(f"# {os.path.basename(system_dir)}/{os.path.basename(run)}")
    print(f"# {len(times)} frames @ {dt:.0f} ns, persistent >= {args.min_lifetime:.0f} ns, "
          f"tail >= {args.exclude_tail} excluded")
    print(f"# tyrosines analysed: {tyr.sum()}   phenylalanines: {phe.sum()}")

    if per_res[live].sum() == 0:
        sys.exit("no persistent contacts")

    # --- 1. is Tyr elevated AS A CLASS? -------------------------------------
    print("\n[1] residue class vs all other residues (contacts per unit exposure)")
    allm = dens[live].mean()
    for label, mask in (("Tyr", tyr), ("Phe", phe)):
        r = dens[mask].mean() / allm if mask.sum() else float("nan")
        print(f"    {label}: {r:5.2f}x the protein-wide mean"
              f"   ({'force-field stickiness' if r > 1.3 else 'not elevated'})")

    # --- 2. where do the targets rank AMONG TYROSINES? ----------------------
    print("\n[2] rank of each target among tyrosines (exposure-corrected)")
    tyr_idx = np.where(tyr)[0]
    order = tyr_idx[np.argsort(dens[tyr_idx])[::-1]]
    for t in targets:
        i = t - offset - 1
        if i not in tyr_idx:
            print(f"    Y{t}: not a tyrosine in this structure"); continue
        rank = int(np.where(order == i)[0][0]) + 1
        pct = 100 * (1 - (rank - 1) / len(order))
        print(f"    Y{t}: rank {rank} of {len(order)}  ({pct:.0f}th percentile)"
              f"   raw share {per_res[i]/per_res[live].sum()*100:.2f}%")
    print("    full tyrosine ranking: " +
          " ".join(f"Y{int(pdb_ids[i])}" for i in order))

    # --- 3. permutation null preserving type and exposure band --------------
    print(f"\n[3] permutation test, {args.n_perm} shuffles within type x exposure band")
    bands = np.digitize(expo, np.quantile(expo[live], [0.25, 0.5, 0.75]))
    rng = np.random.default_rng(0)
    obs = per_res[[t - offset - 1 for t in targets]].sum()
    null = np.zeros(args.n_perm)
    pool = {}
    for i in np.where(live)[0]:
        pool.setdefault((tchar[i], bands[i]), []).append(i)
    tgt_idx = [t - offset - 1 for t in targets]
    for k in range(args.n_perm):
        s = 0.0
        for i in tgt_idx:
            cand = pool.get((tchar[i], bands[i]), [i])
            s += per_res[cand[rng.integers(len(cand))]]
        null[k] = s
    p = (null >= obs).mean()
    print(f"    observed {obs:.0f}   null median {np.median(null):.0f}   "
          f"ratio {obs/max(np.median(null),1e-9):.2f}x")
    print(f"    p = {p:.4f}  ({'SPECIFIC' if p < 0.05 else 'NOT distinguishable from type+exposure-matched residues'})")


if __name__ == "__main__":
    main()
