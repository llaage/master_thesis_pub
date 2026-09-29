#!/usr/bin/env python
"""
Section 8: does RH5 occlude the surface where I53-50B has to bind?

A mechanism nobody on this project had listed: RH5 does not need to change the
chemistry of the assembly interface to block assembly, it only needs to sit in
front of it often enough to slow association and trap the particle off-pathway.
That predicts something specific and cheap -- the I53-50A residues forming the
A-B interface should be measurably less solvent-accessible in the fusion trimer
(rung 2) than in the bare I53-50A trimer (rung 1).

This costs no extra simulation. It reuses two trajectories that have to be run
anyway, and it is the reason rung 1 was cut out of the SAME AF3 model as rung 2:
against a separately predicted trimer, any SASA difference would be confounded
by the two models simply being different models.

WHICH RESIDUES
The A-B interface set comes from the cage-seeded rung 4 structure, via
find_interface_residues.py -- the same definition the persistence analysis uses,
so the two are talking about the same surface.

READING THE RESULT
The comparison that matters is a RATIO of accessible area between two systems,
not an absolute area. Martini bead radii and a 2.6 A probe make the absolute
number meaningless on its own; both systems carry the same bias, so it cancels
in the ratio. Occlusion by RH5 shows up as rung 2 having systematically less
exposed A-B interface than rung 1, well outside the replicate spread.

Usage:
    python scripts/sasa_occlusion.py \\
        --bare systems/rung1_s150_r1 systems/rung1_s150_r2 \\
        --fusion systems/rung2_s150_r1 systems/rung2_s150_r2 \\
        --interface analysis/iface_AB.tsv
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_common as ac


def load_interface(path):
    """{resid} from a find_interface_residues.py table (PDB numbering)."""
    out = set()
    for i, line in enumerate(open(path)):
        if i == 0 or not line.strip():
            continue
        p = line.split("\t")
        if len(p) >= 2:
            out.add(int(p[1]))
    if not out:
        raise ValueError(f"{path}: no residues parsed")
    return out


def measure(system_dir, iface_resids, traj, stride, i53a_range):
    """Mean exposed SASA of the A-B interface residues, per frame."""
    sd = os.path.abspath(system_dir)
    path = os.path.join(sd, traj)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    chains, _, rpdb, names = ac.bead_table(sd)
    u, _ = ac.load(sd, path)

    lo, hi = i53a_range
    # Only I53-50A beads, and only the interface residues within them. RH5 is
    # excluded from the SELECTION but present in the STRUCTURE, which is the
    # whole point -- it is what may be doing the occluding.
    mask = (rpdb >= lo) & (rpdb <= hi) & np.isin(rpdb, list(iface_resids))
    idx = np.where(mask)[0]
    if not len(idx):
        raise ValueError(f"{sd}: no interface beads found in {lo}-{hi}")

    vals = []
    for ts in u.trajectory[::stride]:
        box = ts.dimensions        # full triclinic box; see analysis_common.unwrap
        pos = u.atoms.positions
        # unwrap the whole protein around chain-by-chain, then measure
        whole = np.vstack([ac.unwrap(pos, box, np.where(chains == c)[0])
                           for c in sorted(set(chains))])
        order = np.concatenate([np.where(chains == c)[0]
                                for c in sorted(set(chains))])
        s = ac.cg_sasa(whole)
        remap = np.empty(len(order), dtype=int)
        remap[order] = np.arange(len(order))
        vals.append(float(s[remap[idx]].sum()))
    return np.array(vals), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bare", nargs="+", required=True,
                    help="rung 1 system dirs (I53-50A trimer alone)")
    ap.add_argument("--fusion", nargs="+", required=True,
                    help="rung 2 system dirs (RH5-I53-50A fusion trimer)")
    ap.add_argument("--interface", required=True,
                    help="find_interface_residues.py table for the A-B interface")
    ap.add_argument("--traj", default="prod/prod.xtc")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--i53a", default="339:554")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.i53a.split(":"))
    iface = load_interface(a.interface)
    print(f">>> A-B interface: {len(iface)} residues from "
          f"{os.path.basename(a.interface)}")

    out = {}
    for label, dirs in (("bare (rung 1)", a.bare), ("fusion (rung 2)", a.fusion)):
        per_rep = []
        for d in dirs:
            try:
                v, nb = measure(d, iface, a.traj, a.stride, (lo, hi))
            except (FileNotFoundError, ValueError) as e:
                print(f"    {os.path.basename(d)}: skipped ({e})")
                continue
            per_rep.append(v.mean())
            print(f"    {os.path.basename(d)}: {nb} interface beads, "
                  f"mean exposed {v.mean():.0f} A^2 (sd over frames {v.std():.0f})")
        if not per_rep:
            sys.exit(f"ERROR: no usable trajectories for {label}")
        out[label] = per_rep
        print(f"  {label}: {np.mean(per_rep):.0f} A^2 "
              f"(n={len(per_rep)} replicates, spread {np.std(per_rep):.0f})")

    b = np.mean(out["bare (rung 1)"])
    f = np.mean(out["fusion (rung 2)"])
    ratio = f / b if b else float("nan")
    spread = max(np.std(out["bare (rung 1)"]), np.std(out["fusion (rung 2)"]))
    print(f"\n  exposed A-B interface, fusion / bare = {ratio:.3f}")
    print(f"  ({(1 - ratio) * 100:+.1f}% change; replicate spread "
          f"{spread / b * 100:.1f}% of the bare mean)")

    if abs(1 - ratio) * b < 2 * spread:
        print("\n  NOT SIGNIFICANT against the replicate spread. This does not")
        print("  support steric occlusion, and with this few replicates it does")
        print("  not exclude it either.")
    elif ratio < 1:
        print("\n  RH5 reduces access to the A-B interface. Steric occlusion is")
        print("  live: the fix would be linker length or attachment geometry")
        print("  rather than surface mutation. Worth pairing with a bench test")
        print("  of slow addition or sub-stoichiometric B.")
    else:
        print("\n  No occlusion; the A-B surface is no less accessible in the")
        print("  fusion. This mechanism is out.")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"bare": out["bare (rung 1)"],
                       "fusion": out["fusion (rung 2)"],
                       "ratio": ratio, "n_interface_residues": len(iface)},
                      fh, indent=2)
        print(f"\n>>> wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
