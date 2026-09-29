#!/usr/bin/env python
"""
The pre-registered dissociation metrics, for any rung.

These thresholds were written down in the design document BEFORE any trajectory
existed, and they are hardcoded here rather than passed in, so a run cannot be
scored against a threshold chosen after seeing it:

    metric                          persists      dissociates
    native interface contacts        > 60%          < 20%    (of t=0)
    buried interface SASA            > 70%          < 30%    (of t=0)
    COM-COM drift                    < 1 nm         > 3 nm
    complex RMSD                     < 1 nm            -

Anything between the two columns is reported as AMBIGUOUS and stays that way.
The previous campaign's most expensive lesson was reading an early frame as a
result: at 175 ns a lambda=1.04 pentamer looked "9 of 12 intact", and at 2 us it
was 0 of 12. Those complexes were not stable, only slow. So the verdict is taken
from the FINAL WINDOW, not the last frame or an early one, and the time at which
each threshold was first crossed is reported alongside.

WHICH INTERFACE
Give two chain groups. For a homo-oligomer, --self-oligomer scores every chain
against the rest, which is the right question for "does the trimer hold".

    # rung 1: does the I53-50A trimer hold?
    python scripts/interface_persistence.py systems/rung1_s150_r1 --self-oligomer

    # rung 4: does the A-B interface hold?
    python scripts/interface_persistence.py systems/rung4_s150_r1 \\
        --group1 ABC --group2 DEFGH
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

# Pre-registered. Do not edit to fit a result.
PERSIST = {"contacts": 0.60, "sasa": 0.70, "com": 10.0, "rmsd": 10.0}   # A
DISSOC = {"contacts": 0.20, "sasa": 0.30, "com": 30.0}                  # A
CONTACT_CUT = 6.0     # A between CG beads; ~one Martini sigma plus a margin


def verdict(frac_contacts, frac_sasa, com_drift):
    """Score one frame's metrics against the pre-registered table."""
    p = (frac_contacts > PERSIST["contacts"] and com_drift < PERSIST["com"])
    d = (frac_contacts < DISSOC["contacts"] or com_drift > DISSOC["com"])
    if frac_sasa is not None:
        p = p and frac_sasa > PERSIST["sasa"]
        d = d or frac_sasa < DISSOC["sasa"]
    if d:
        return "DISSOCIATES"
    if p:
        return "PERSISTS"
    return "AMBIGUOUS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system_dir")
    ap.add_argument("--traj", default="prod/prod.xtc")
    ap.add_argument("--group1", default=None, help="chain IDs, e.g. ABC")
    ap.add_argument("--group2", default=None, help="chain IDs, e.g. DEFGH")
    ap.add_argument("--self-oligomer", action="store_true",
                    help="score each chain against all the others")
    ap.add_argument("--residues", default=None,
                    help="restrict scoring to this PDB residue range, e.g. "
                         "339:554. REQUIRED to compare a fusion construct with "
                         "the bare domain: with whole chains, a fusion chain's "
                         "centre of mass is dominated by RH5 swinging on its "
                         "linker, so the COM-drift metric reports linker motion "
                         "rather than anything about the interface being tested.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sasa-stride", type=int, default=20,
                    help="SASA is the expensive metric; compute it less often")
    ap.add_argument("--final-window", type=float, default=0.10,
                    help="fraction of the trajectory averaged for the verdict")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    sd = os.path.abspath(a.system_dir)
    traj = os.path.join(sd, a.traj)
    if not os.path.isfile(traj):
        sys.exit(f"ERROR: no trajectory at {traj}")

    chains, _, resid_pdb, names = ac.bead_table(sd)
    u, n_prot = ac.load(sd, traj)

    if a.self_oligomer:
        uniq = sorted(set(chains))
        pairs = [(c, [x for x in uniq if x != c]) for c in uniq]
        label = f"each of {''.join(uniq)} vs the rest"
    else:
        if not (a.group1 and a.group2):
            sys.exit("ERROR: give --group1 and --group2, or --self-oligomer")
        pairs = [(a.group1, list(a.group2))]
        label = f"{a.group1} vs {a.group2}"

    print(f">>> {os.path.basename(sd)}   interface: {label}"
          + (f"   residues {a.residues}" if a.residues else ""))
    print(f"    {n_prot} beads, {len(u.trajectory)} frames")

    results = {}
    for gid, (g1, g2) in enumerate(pairs):
        # per-chain index lists: a group must be unwrapped chain by chain
        sel = np.ones(len(chains), dtype=bool)
        if a.residues:
            lo, hi = (int(x) for x in a.residues.split(":"))
            sel = (resid_pdb >= lo) & (resid_pdb <= hi)
        grp1 = [ix[sel[ix]] for ix in ac.group_indices(chains, list(g1))]
        grp2 = [ix[sel[ix]] for ix in ac.group_indices(chains, list(g2))]
        grp1 = [ix for ix in grp1 if len(ix)]
        grp2 = [ix for ix in grp2 if len(ix)]
        if not grp1 or not grp2:
            sys.exit(f"ERROR: --residues {a.residues} selects no beads in one "
                     f"of the groups")
        idx1 = np.concatenate(grp1)
        idx2 = np.concatenate(grp2)

        nj = ac.NoJump()      # continuous tracking; see analysis_common.NoJump
        times, n_con, com_d, rmsds, sasa_f = [], [], [], [], []
        native = None
        pairs0 = None
        ref_xyz = None
        base_buried = None

        for k, ts in enumerate(u.trajectory[::a.stride]):
            box = ts.dimensions        # full triclinic box; see analysis_common.unwrap
            pos = u.atoms.positions
            # ONE coordinate convention for every geometric metric: no-jump,
            # i.e. each chain followed continuously through time. Mixing
            # conventions is what broke this twice -- minimum image capped the
            # COM drift at the box, and once that was fixed by moving the COMs
            # to no-jump while leaving the SASA groups on per-frame images, the
            # two groups sat in different periodic images and "buried area"
            # came out as 0 or 5e10.
            #
            # With continuous coordinates the configuration is physically real,
            # so distances are plain Euclidean and PBC must NOT be reapplied:
            # wrapping a chain that has genuinely travelled 20 nm back to within
            # 6 nm would invent an interface that is not there.
            p1 = np.vstack([nj.coords(pos, box, ('1', i), ix)
                            for i, ix in enumerate(grp1)])
            p2 = np.vstack([nj.coords(pos, box, ('2', i), ix)
                            for i, ix in enumerate(grp2)])
            c1, c2 = p1.mean(0), p2.mean(0)
            if native is None:
                # Record the ACTUAL t=0 contact pairs, and track those.
                # Counting total contacts each frame and dividing by the t=0
                # count is not "native contacts retained": a complex that comes
                # off the designed interface and re-binds somewhere else scores
                # above 100%. rung4_s150_r1 did exactly that -- 159% by the loose
                # measure while its COM had moved 3.3 nm, which is not a
                # persisting interface, it is a different one.
                pairs0 = ac.contact_pairs(p1, p2, None, CONTACT_CUT)
                native = max(len(pairs0), 1)
                ref_xyz = np.vstack([p1, p2])
                d0 = np.linalg.norm(c2 - c1)
            # how many of the ORIGINAL pairs are still formed
            d_native = np.linalg.norm(p1[pairs0[:, 0]] - p2[pairs0[:, 1]], axis=1)
            c = int((d_native <= CONTACT_CUT).sum())
            times.append(ts.time / 1000.0)
            n_con.append(c)
            com_d.append(abs(np.linalg.norm(c2 - c1) - d0))
            rmsds.append(ac.rmsd_after_fit(np.vstack([p1, p2]), ref_xyz))

            if k % a.sasa_stride == 0:
                s1 = ac.cg_sasa(p1).sum()
                s2 = ac.cg_sasa(p2).sum()
                sc = ac.cg_sasa(np.vstack([p1, p2])).sum()
                buried = s1 + s2 - sc
                if base_buried is None:
                    base_buried = max(buried, 1e-6)
                sasa_f.append((ts.time / 1000.0, buried / base_buried))

        t = np.array(times)
        fc = np.array(n_con) / native
        cd = np.array(com_d)
        rm = np.array(rmsds)

        nwin = max(1, int(len(t) * a.final_window))
        fc_f = float(fc[-nwin:].mean())
        cd_f = float(cd[-nwin:].mean())
        rm_f = float(rm[-nwin:].mean())
        sf = np.array([v for _, v in sasa_f]) if sasa_f else None
        sf_f = float(sf[-max(1, int(len(sf) * a.final_window)):].mean()) if sf is not None else None

        v = verdict(fc_f, sf_f, cd_f)

        def crossed(arr, thresh, below=True):
            hit = np.where(arr < thresh if below else arr > thresh)[0]
            return f"{t[hit[0]]:.0f} ns" if len(hit) else "never"

        name = g1 if len(g1) == 1 else f"{g1}|{''.join(g2)}"
        print(f"\n  [{name}]  {t[-1]:.4g} ns, {native} native contacts at t=0")
        print(f"    native contacts retained : {fc_f * 100:5.1f}%   "
              f"(<20% first at {crossed(fc, DISSOC['contacts'])})")
        if sf is not None:
            print(f"    buried interface SASA    : {sf_f * 100:5.1f}%")
        print(f"    COM-COM drift            : {cd_f / 10:5.2f} nm  "
              f"(>3 nm first at {crossed(cd, DISSOC['com'], below=False)})")
        print(f"    complex RMSD             : {rm_f / 10:5.2f} nm")
        print(f"    VERDICT (final {a.final_window * 100:.0f}% of trajectory): {v}")

        results[name] = {
            "ns": float(t[-1]), "native_contacts_t0": int(native),
            "contacts_retained": fc_f, "buried_sasa_frac": sf_f,
            "com_drift_nm": cd_f / 10, "rmsd_nm": rm_f / 10, "verdict": v,
        }

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"system": os.path.basename(sd), "interfaces": results},
                      fh, indent=2)
        print(f"\n>>> wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
