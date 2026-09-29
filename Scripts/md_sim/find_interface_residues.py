#!/usr/bin/env python
"""
List the residues forming an interface, and emit a martinize2 -mutate string
for the dynamic-range control.

TWO USES

1. Defining an interface once, in one place. The section 8 occlusion analysis
   needs "the I53-50A residues that form the A-B interface" and the persistence
   analysis needs "native interface contacts at t=0". Both must mean the same
   set of residues or the numbers cannot be compared, so both read it from here.

2. Building the broken-interface control. `findings.md` cannot distinguish
   "every rung persists" from "the assay cannot detect dissociation" without a
   system designed to fall apart. This emits the mutation list for one: the
   buriedmost hydrophobic residues of the A-A interface, swapped for charged
   residues so the designed core is replaced by like-charge repulsion.

   Mutating to a SINGLE charge sign is deliberate. Alternating Asp and Lys would
   let the mutated positions pair up and form a new salt-bridge interface, which
   could easily be as stable as the one removed -- a "broken" control that does
   not break is worse than no control, because it reads as the assay having no
   dynamic range.

WHY BURIAL, NOT JUST CONTACT
Residues merely touching across an interface are mostly peripheral and
contribute little binding energy; mutating them gives a control that may not
dissociate. Ranking by buried surface area picks the ones actually holding the
interface together.

Usage:
    # inspect an interface
    python scripts/find_interface_residues.py inputs/rung4_50A_50B.pdb \\
        --group1 ABC --group2 DEFGH

    # emit mutations for the broken control (A-A interface of the 50A trimer)
    python scripts/find_interface_residues.py inputs/rung1_50A_trimer.pdb \\
        --self-oligomer --mutate-to GLU --n-mutate 4
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structlib as sl

HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO", "CYS"}

# Residues that are never mutated for the dynamic-range control, even when they
# rank among the buriedmost.
#
#   PRO  changes the backbone, not the interface. Removing a proline alters
#        local conformation and secondary structure, so a control built on it
#        would confound "the interface was destroyed" with "the fold moved".
#        It also breaks the build outright: martinize2's internal DSSP returns
#        'NA' for a mutated proline and dies with KeyError: np.str_('NA').
#   GLY  same argument -- it is in the backbone-flexibility business.
#   CYS  may be half of a disulfide; -cys auto would then lose a bond that is
#        holding the fold together, which is a different experiment.
NO_MUTATE = {"PRO", "GLY", "CYS"}

# Shrake-Rupley radii, Angstrom. Coarse but adequate: only the RANKING of
# residues by burial is used, and that is insensitive to a few hundredths.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}


def sasa(atoms, probe=1.4, n_sphere=96):
    """Per-atom solvent accessible surface area by Shrake-Rupley."""
    pos = np.array([a.xyz for a in atoms])
    rad = np.array([VDW.get(a.elem.upper()[:1], 1.70) for a in atoms]) + probe
    # golden-spiral points on the unit sphere
    k = np.arange(n_sphere) + 0.5
    phi = np.arccos(1 - 2 * k / n_sphere)
    theta = np.pi * (1 + 5 ** 0.5) * k
    unit = np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])
    tree = cKDTree(pos)
    out = np.zeros(len(atoms))
    for i in range(len(atoms)):
        nb = [j for j in tree.query_ball_point(pos[i], rad[i] + rad.max())
              if j != i]
        if not nb:
            out[i] = 4 * np.pi * rad[i] ** 2
            continue
        pts = pos[i] + unit * rad[i]
        d = np.linalg.norm(pts[:, None, :] - pos[nb][None, :, :], axis=2)
        acc = (d >= rad[nb][None, :]).all(axis=1)
        out[i] = 4 * np.pi * rad[i] ** 2 * acc.mean()
    return out


def residue_sasa(atoms, area):
    out = {}
    for a, s in zip(atoms, area):
        out[(a.chain, a.resid, a.resname)] = out.get((a.chain, a.resid, a.resname), 0.0) + s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    ap.add_argument("--group1", default=None, help="chain IDs, e.g. ABC")
    ap.add_argument("--group2", default=None, help="chain IDs, e.g. DEFGH")
    ap.add_argument("--self-oligomer", action="store_true",
                    help="treat each chain against all the others")
    ap.add_argument("--cutoff", type=float, default=5.0,
                    help="heavy-atom contact cutoff, A")
    ap.add_argument("--n-mutate", type=int, default=0,
                    help="emit this many mutations per chain")
    ap.add_argument("--mutate-to", default="GLU")
    ap.add_argument("--hydrophobic-only", action="store_true", default=True)
    ap.add_argument("--out", default=None, help="write the residue list here")
    a = ap.parse_args()

    atoms = sl.read_pdb(a.pdb)
    bc = sl.by_chain(atoms)

    if a.self_oligomer:
        pairs = [(c, [x for x in bc if x != c]) for c in bc]
    else:
        if not (a.group1 and a.group2):
            sys.exit("ERROR: give --group1 and --group2, or --self-oligomer")
        pairs = [(c, list(a.group2)) for c in a.group1]

    print(f">>> {os.path.basename(a.pdb)}  cutoff {a.cutoff} A")

    # burial = SASA of the isolated chain minus SASA in the complex.
    # Computed per chain rather than for the whole assembly, so the number
    # means "how much of this residue the partner covers".
    lines = []
    mutations = []
    burial_by_res = {}     # (resid, resname) -> [buried area in each chain]
    chains_seen = []
    for ch, partners in pairs:
        mine = bc[ch]
        theirs = [x for p in partners for x in bc[p]]
        if not theirs:
            continue
        tree = cKDTree(np.array([x.xyz for x in theirs]))
        touching = {i for i, n in enumerate(
            tree.query_ball_point(np.array([x.xyz for x in mine]), a.cutoff,
                                  return_length=True)) if n}
        if not touching:
            print(f"  chain {ch}: no contacts within {a.cutoff} A")
            continue

        free = residue_sasa(mine, sasa(mine))
        both = mine + theirs
        comp_all = sasa(both)
        bound = residue_sasa(mine, comp_all[:len(mine)])
        buried = {k: free[k] - bound.get(k, 0.0) for k in free}

        iface = sorted(
            ((k, v) for k, v in buried.items() if v > 1.0),
            key=lambda kv: -kv[1])
        print(f"  chain {ch}: {len(iface)} interface residues, "
              f"total buried {sum(v for _, v in iface):.0f} A^2")
        for (c, r, rn), v in iface[:12]:
            lines.append(f"{c}\t{r}\t{rn}\t{v:.1f}")
            print(f"      {c} {rn}{r:<5d} buried {v:6.1f} A^2"
                  + ("  hydrophobic" if rn in HYDROPHOBIC else ""))

        for (c, r, rn), v in iface:
            if rn in NO_MUTATE:
                continue
            if not a.hydrophobic_only or rn in HYDROPHOBIC:
                burial_by_res.setdefault((r, rn), []).append(v)
        chains_seen.append(ch)

    # Pick the mutation set from burial AVERAGED OVER CHAINS, then apply the
    # same positions to every chain. Ranking each chain independently is the
    # obvious thing and it is wrong: the per-chain rankings differ slightly
    # (the model is not perfectly symmetric), so chain C came out with Pro462
    # where A and B had Phe479. That yields an asymmetric control, where the
    # three interfaces of a C3 trimer are no longer equivalent and a partial
    # dissociation cannot be attributed to the mutation rather than to the
    # asymmetry of the construct.
    if a.n_mutate:
        ranked = sorted(burial_by_res.items(),
                        key=lambda kv: -np.mean(kv[1]))
        chosen = ranked[:a.n_mutate]
        print(f"\n>>> mutation set, ranked by burial averaged over "
              f"{len(chains_seen)} chains:")
        for (r, rn), vals in chosen:
            print(f"      {rn}{r:<5d} mean buried {np.mean(vals):6.1f} A^2 "
                  f"(per chain: {', '.join(f'{v:.0f}' for v in vals)})")
        for ch in chains_seen:
            for (r, rn), _ in chosen:
                mutations.append(f"{ch}-{rn}{r}:{a.mutate_to}")

    if a.out:
        with open(a.out, "w") as fh:
            fh.write("chain\tresid\tresname\tburied_A2\n")
            fh.write("\n".join(lines) + "\n")
        print(f">>> wrote {a.out}")

    if mutations:
        print("\n>>> martinize2 mutation arguments "
              f"({len(mutations)} residues -> {a.mutate_to}):")
        print("    " + " ".join(f"-mutate {m}" for m in mutations))
        print("\n    All mutations are to the SAME charge so the mutated "
              "positions repel each other\n    rather than forming a "
              "replacement salt-bridge interface.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
