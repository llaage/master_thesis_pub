"""
Shared plumbing for the assembly-ladder analyses.

Rewritten for this campaign. The previous version assumed the multi-copy layout
-- N identical copies of one molecule, described by `nmol` and `beads_per_copy`
in the MANIFEST. Here every system is a single complex of NON-identical chains,
so bead identity comes from the coarse-grained structure itself rather than from
arithmetic on a copy count.

Four things every analysis here needs, each with a way to go quietly wrong:

1. BEAD IDENTITY. Chains differ in length and residue numbering within one
   system, so "which chain is bead 4173 in" cannot be computed from a stride.
   It is read from the martinize2 CG structure, which is the same file the
   topology was built from.

2. RESIDUE NUMBERING. martinize2 renumbers every chain from 1. Worse, in this
   project the offset differs BETWEEN CHAINS of one system: rung 4 holds a 50A
   trimer numbered 339-554 next to a pentamer numbered 1-159. Any number shown
   to a human must be PDB numbering, and the per-chain offsets come from the
   MANIFEST rather than from an assumption.

3. PERIODIC BOUNDARIES. A dissociating chain will cross a box face, and its
   beads then sit at x~0 and x~L simultaneously; a naive centre of mass lands in
   the middle of the box. Everything here unwraps per chain first. This matters
   more than in the last campaign: dissociation IS the observable here, and a
   PBC artifact looks exactly like it.

4. TRAJECTORY CONTENTS. Production writes protein beads only
   (compressed-x-grps = Protein) while equilibration .xtc files hold the whole
   system. The loader detects which it has rather than assuming.
"""
import os

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib import distances as _D


def read_manifest(system_dir):
    """Parse a build_complex.sh MANIFEST into a dict."""
    path = os.path.join(system_dir, "MANIFEST")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no MANIFEST in {system_dir}")
    out = {}
    for line in open(path):
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def chain_offsets(system_dir):
    """
    {chain_id: offset} with PDB_resid = CG_resid + offset.

    Falls back to a single global offset when every chain shares one.
    """
    man = read_manifest(system_dir)
    raw = man.get("resid_offsets", "").strip()
    if raw:
        out = {}
        for part in raw.split(","):
            if ":" in part:
                c, v = part.split(":")
                out[c.strip()] = int(v)
        if out:
            return out
    return {"*": int(man.get("resid_offset", 0) or 0)}


def cg_structure(system_dir):
    """Path to the martinize2 CG .pdb -- the authority on bead identity."""
    d = os.path.join(system_dir, "martinize2")
    hits = [f for f in os.listdir(d) if f.endswith("_cg.pdb")]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one *_cg.pdb in {d}, found {hits}")
    return os.path.join(d, hits[0])


def bead_table(system_dir):
    """
    Per-bead identity, in trajectory order.

    Returns (chains, resid_cg, resid_pdb, names) as arrays. The CG .pdb lists
    beads in exactly the order the topology and the protein-only .xtc use, so
    index i here is atom i there.
    """
    offs = chain_offsets(system_dir)
    ch, rid, nm = [], [], []
    for line in open(cg_structure(system_dir)):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        ch.append(line[21])
        rid.append(int(line[22:26]))
        nm.append(line[12:16].strip())
    ch = np.array(ch)
    rid = np.array(rid)
    pdb = np.array([r + offs.get(c, offs.get("*", 0)) for c, r in zip(ch, rid)])
    return ch, rid, pdb, np.array(nm)


def protein_reference(system_dir, n_prot):
    """A protein-only .gro to pair with a protein-only .xtc."""
    ref = os.path.join(system_dir, "solvate", "prot_ref.gro")
    if os.path.isfile(ref):
        return ref
    src = os.path.join(system_dir, "solvate", "solv_ions.gro")
    with open(src) as fh:
        lines = fh.readlines()
    with open(ref, "w") as fh:
        fh.write("protein-only reference for xtc analysis\n")
        fh.write(f"{n_prot}\n")
        fh.writelines(lines[2:2 + n_prot])      # protein leads the file
        fh.write(lines[-1])
    return ref


def load(system_dir, xtc):
    """Universe over the protein beads, whatever the .xtc happens to contain."""
    ch, _, _, _ = bead_table(system_dir)
    n_prot = len(ch)
    full_gro = os.path.join(system_dir, "solvate", "solv_ions.gro")

    # Read topology and trajectory separately: binding them in one Universe
    # raises immediately when atom counts differ, which is exactly the case this
    # function exists to handle.
    n_full = len(mda.Universe(full_gro).atoms)
    with mda.lib.formats.libmdaxdr.XTCFile(xtc) as xf:
        n_xtc = xf.n_atoms

    if n_xtc == n_prot:
        u = mda.Universe(protein_reference(system_dir, n_prot), xtc)
    elif n_xtc == n_full:
        u = mda.Universe(full_gro, xtc)
    else:
        raise ValueError(
            f"{xtc}: {n_xtc} atoms matches neither the protein block ({n_prot}) "
            f"nor the full system ({n_full})")
    return u, n_prot


def group_indices(chains, wanted):
    """Bead indices for each chain id in `wanted`."""
    return [np.where(chains == c)[0] for c in wanted]


def unwrap(positions, box, idx):
    """
    Beads of one chain, made whole across periodic boundaries.

    `box` must be the full 6-element MDAnalysis box (a, b, c, alpha, beta,
    gamma), NOT the three lengths.

    THIS IS WHERE THE ORTHORHOMBIC ASSUMPTION BIT. These systems are built in a
    rhombic dodecahedron -- `ts.dimensions` reads
    [128.27, 128.27, 128.27, 60, 60, 90] -- and the usual
    `d - box * round(d / box)` minimum image is only valid for a rectangular
    cell. Applied to a triclinic box it silently tears molecules apart: a folded
    216-residue chain whose real extent is 4.6 nm measured up to 12.8 nm, the
    full box width, and the COM-COM distance derived from it swung by ~2 nm on a
    trimer that never moved. The metric looked broken, which is lucky; had it
    been subtler it would have looked like dissociation.

    `minimize_vectors` applies the minimum image convention correctly for
    triclinic cells.

    SECOND FAILURE, SAME FUNCTION. Imaging every bead against the chain's FIRST
    bead is only valid while the whole chain lies within half a box of it. A
    fusion chain with its RH5 domain swung out on the linker can exceed that:
    in rung 8 replicate 2 (the smallest box of the set, 21.7 nm) 23-30 backbone
    steps came out exactly one box length -- 210 A -- long, scattering core beads
    across the render while the chain's centre of mass stayed correct, so no
    COM-level check caught it.

    Unwrapping SEQUENTIALLY fixes it: take the minimum image of each
    consecutive step and accumulate. Consecutive beads are ~3.5 A apart, never
    close to half a box, so this is valid for any chain extent and any box.
    """
    p = positions[idx]
    if len(p) < 2:
        return p.copy()
    steps = _D.minimize_vectors(np.diff(p, axis=0), box=box)
    out = np.empty_like(p)
    out[0] = p[0]
    out[1:] = p[0] + np.cumsum(steps, axis=0)
    return out


def unwrap_chains(positions, box, idx_list):
    """
    Make a MULTI-chain group whole: each chain unwrapped on its own, then placed
    in the periodic image nearest the first chain.

    Unwrapping several chains against one reference bead -- as
    `unwrap(pos, box, concatenated_indices)` does -- is only right while every
    chain happens to sit in the same image as that bead. When one does not, the
    group reads as spanning the whole box: chains B+C of an intact trimer
    measured 7.7-12.8 nm across, against a real extent near 6 nm.

    Placing each chain by its own COM is also what keeps a genuinely dissociating
    complex honest: the departing chain is put in the image physically nearest
    the rest, which is the separation we want to measure, not an image artefact.

    Returns coordinates concatenated in the order of `idx_list`.
    """
    parts = [unwrap(positions, box, idx) for idx in idx_list]
    ref = parts[0].mean(axis=0)
    out = [parts[0]]
    for p in parts[1:]:
        out.append(p + (min_image(p.mean(axis=0), ref, box) - (p.mean(axis=0) - ref)))
    return np.vstack(out)


class NoJump:
    """
    Track chains continuously through time, so a subunit that leaves keeps
    leaving instead of being folded back by the minimum image.

    Per-frame minimum image is right for contacts and wrong for displacement.
    Once a chain travels further than the minimum-image limit -- about 6 nm in
    these 12.8 nm dodecahedra -- it is wrapped back to the near side, and both
    the COM-drift metric and any movie built the same way report a complex that
    looks intact. Measured on the broken control at 5 us: true chain-chain
    separations of 6.6, 6.6 and 4.1 nm with ZERO interface contacts, but the
    per-frame minimum image reported a 3.2 nm drift and rendered a compact,
    apparently-assembled trimer.

    The fix is standard: accumulate displacement frame to frame, taking the
    minimum image of each STEP rather than of the total. Valid while a chain
    moves less than half a box between frames, which at 500 ps per frame is
    never in doubt.
    """

    def __init__(self):
        self._prev_raw = {}
        self._offset = {}

    def coords(self, positions, box, key, idx, ref_com=None):
        """
        Internally-whole chain `key`, placed on its continuous path.

        `ref_com` seeds the FIRST frame: the chain is put in the periodic image
        nearest that reference point, and tracked continuously from there.

        Without it the first frame accepts whatever image the trajectory happens
        to store. GROMACS wraps molecules into the box, so a protomer of an
        intact trimer can be written on the far side; no-jump then faithfully
        keeps it there for the whole trajectory. That is what made rung 8
        replicates 1 and 2 render as dissociated trimers from frame 1 (core COM
        separations of 28-30 nm on screen against a true 3.5 nm, while the
        contact analysis correctly reported 38-44 contacts at 100 % occupancy).

        Seeding does NOT hide real dissociation: every rung starts assembled, so
        the nearest image at frame 0 is the assembled one, and a chain that
        later departs still accumulates its true displacement.
        """
        p = unwrap(positions, box, idx)
        raw = p.mean(axis=0)
        if key not in self._prev_raw:
            if ref_com is None:
                self._offset[key] = np.zeros(3)
            else:
                self._offset[key] = min_image(raw, ref_com, box) - (raw - ref_com)
        else:
            step = min_image(raw, self._prev_raw[key], box)
            drift = (self._prev_raw[key] + self._offset[key] + step) - raw
            self._offset[key] = drift
        self._prev_raw[key] = raw
        return p + self._offset[key]


def com_of(positions, box, idx):
    """PBC-safe centre of geometry of one chain."""
    return unwrap(positions, box, idx).mean(axis=0)


def min_image(a, b, box):
    """Minimum-image displacement a - b, triclinic-correct."""
    return _D.minimize_vectors(np.asarray(a - b, dtype=np.float32)[None, :],
                               box=box)[0]


def min_image_dist(pa, pb, box):
    """Minimum-image distance matrix between two bead sets, triclinic-correct."""
    return _D.distance_array(np.ascontiguousarray(pa, dtype=np.float32),
                             np.ascontiguousarray(pb, dtype=np.float32),
                             box=box)


def n_contacts(pa, pb, box, cutoff):
    """
    Count bead pairs closer than `cutoff`.

    Pass box=None when the coordinates are already continuous (NoJump). Applying
    the minimum image to no-jump coordinates would wrap a chain that has
    genuinely travelled 20 nm back to within 6 nm and invent contacts across it.
    """
    pairs = _D.capped_distance(
        np.ascontiguousarray(pa, dtype=np.float32),
        np.ascontiguousarray(pb, dtype=np.float32),
        max_cutoff=cutoff, box=box, return_distances=False)
    return int(len(pairs))


def contact_pairs(pa, pb, box, cutoff):
    """Indices (i, j) of bead pairs closer than `cutoff`, minimum image."""
    pairs = _D.capped_distance(
        np.ascontiguousarray(pa, dtype=np.float32),
        np.ascontiguousarray(pb, dtype=np.float32),
        max_cutoff=cutoff, box=box, return_distances=False)
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


# Martini bead radius for CG SASA. Shrake-Rupley with the atomistic 1.4 A probe
# reports almost every Martini bead as buried, which silently flattens an
# exposure correction to no correction at all (pitfalls.md). A Martini water
# bead is ~2.6 A.
CG_PROBE = 2.6
CG_BEAD_RADIUS = 2.35


def cg_sasa(positions, probe=CG_PROBE, radius=CG_BEAD_RADIUS, n_sphere=64):
    """Per-bead SASA for a coarse-grained structure, Shrake-Rupley."""
    from scipy.spatial import cKDTree
    pos = np.asarray(positions, float)
    r = radius + probe
    k = np.arange(n_sphere) + 0.5
    phi = np.arccos(1 - 2 * k / n_sphere)
    theta = np.pi * (1 + 5 ** 0.5) * k
    unit = np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])
    tree = cKDTree(pos)
    out = np.zeros(len(pos))
    area = 4 * np.pi * r ** 2
    for i, nb in enumerate(tree.query_ball_point(pos, 2 * r)):
        nb = [j for j in nb if j != i]
        if not nb:
            out[i] = area
            continue
        pts = pos[i] + unit * r
        d = np.linalg.norm(pts[:, None, :] - pos[nb][None, :, :], axis=2)
        out[i] = area * (d >= r).all(axis=1).mean()
    return out


def rmsd_after_fit(mobile, ref):
    """RMSD of `mobile` onto `ref` after optimal superposition (Kabsch)."""
    cm, cr = mobile.mean(0), ref.mean(0)
    P, Q = mobile - cm, ref - cr
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    R = (V @ np.diag([1.0, 1.0, d]) @ Wt).T
    return float(np.sqrt((((P @ R.T) - Q) ** 2).sum() / len(P)))
