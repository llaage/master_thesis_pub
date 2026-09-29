#!/usr/bin/env python
"""
Prepare a coarse-grained trajectory for looking at by eye.

Three things have to be fixed or the picture is misleading rather than merely
ugly:

1. PERIODIC WRAPPING. Beads are stored wrapped into the box, so any copy sitting
   on a boundary appears torn in half across the screen. Each copy is unwrapped
   into a whole molecule here.

2. NO BONDS. Viewers infer bonds from interatomic distance using atomistic
   expectations (~1.5 A). Martini beads sit ~3.5 A apart, so nothing is drawn
   connected and you get a cloud of dots. Showing beads as spheres is the honest
   representation anyway -- a Martini bead really is a blob of ~4 heavy atoms.

3. NO WAY TO TELL COPIES APART. Aggregation is about which molecule is touching
   which, so each of the 27 copies is written as its own chain. Colour by chain
   and clustering becomes obvious at a glance.

Outputs a reference PDB (topology + first frame) and an XTC of unwrapped frames.

Usage:
    python prepare_view.py <run_dir> -o <outdir> [--stride 20]
"""
import argparse
import gzip
import os
import sys

import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_common as ac

# 27 copies need 27 single-character chain ids; PDB accepts upper and lower case
CHAINS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def cluster_labels(pos, box, n_copies, n_beads, cutoff_A=6.0, min_contacts=5):
    """Connected-component label per copy, used only to anchor the view."""
    wrapped = np.mod(pos, box)
    cid = np.arange(len(wrapped)) // n_beads
    pairs = cKDTree(wrapped, boxsize=box).query_pairs(r=cutoff_A, output_type="ndarray")
    if len(pairs):
        ci, cj = cid[pairs[:, 0]], cid[pairs[:, 1]]
        k = ci != cj
        ci, cj = ci[k], cj[k]
        if len(ci):
            lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
            uniq, cnt = np.unique(np.stack([lo, hi], 1), axis=0, return_counts=True)
            sel = uniq[cnt >= min_contacts]
            if len(sel):
                adj = coo_matrix((np.ones(len(sel)), (sel[:, 0], sel[:, 1])),
                                 shape=(n_copies, n_copies))
                _, lab = connected_components(adj, directed=False)
                return lab
    return np.arange(n_copies)


def anchored_positions(pos, box, n_copies, n_beads, ref=None):
    """
    Whole molecules, placed in the image nearest a CONTINUOUS reference point.

    Returns (positions, new_ref).

    Two failure modes this has to avoid:

    * Wrapping each copy independently splits any cluster lying on a box face
      across opposite sides of the screen, showing dispersal that is not there.

    * Re-choosing the anchor every frame -- e.g. "lowest-index member of the
      largest cluster" -- makes the anchor switch molecule whenever cluster
      membership changes. The whole system then jumps to a new origin between
      frames while all internal distances stay fixed: the box appears to
      teleport. That is a rendering artefact, not motion.

    So the reference is carried forward from the previous frame's centroid and
    only drifts smoothly. The frame is finally centred on its own centroid, which
    removes bulk drift without introducing discontinuities.
    """
    pos = ac.unwrap_copies(pos.copy(), box, n_copies, n_beads)
    coms = pos.reshape(n_copies, n_beads, 3).mean(axis=1)

    if ref is None:                      # first frame: seed on the copy 0 centre
        ref = coms[0].copy()

    d = coms - ref
    shift = box * np.round(d / box)
    pos -= np.repeat(shift, n_beads, axis=0)

    placed = coms - shift
    new_ref = placed.mean(axis=0)        # smooth reference for the next frame
    pos -= new_ref                       # centre the view, no per-frame jumps
    return pos, new_ref


def write_pdb_states(path, prot, frames, chain_ids):
    """Multi-state PDB (gzipped): one file that plays as a movie."""
    names = [n[:4] for n in prot.names]
    resids = prot.resids
    resnames = [r[:3] for r in prot.resnames]
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as fh:
        fh.write("REMARK  Martini CG, one chain per copy; show as spheres\n")
        for si, (coords, t_ns) in enumerate(frames, 1):
            fh.write(f"MODEL     {si:4d}\nREMARK  t = {t_ns:.0f} ns\n")
            for i, c in enumerate(coords):
                # PDB serial/resid columns overflow past 99999/9999, so wrap them;
                # they are cosmetic here and PyMOL keeps chain-based colouring.
                fh.write(f"ATOM  {(i+1)%100000:5d} {names[i]:<4s} {resnames[i]:>3s} "
                         f"{chain_ids[i]}{resids[i]%10000:4d}    "
                         f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00\n")
            fh.write("ENDMDL\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--traj", default="prod.xtc")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--center", action="store_true",
                    help="recentre on the largest cluster each frame")
    args = ap.parse_args()

    run = os.path.abspath(args.run_dir)
    system_dir = os.path.dirname(run)
    traj = os.path.join(run, args.traj)
    os.makedirs(args.outdir, exist_ok=True)

    u, n_copies, n_beads = ac.load(system_dir, traj)
    n_prot = n_copies * n_beads
    prot = u.atoms[:n_prot]

    # one chain per copy, so "colour by chain" separates the molecules
    chain_ids = np.repeat([CHAINS[i % len(CHAINS)] for i in range(n_copies)], n_beads)
    if not hasattr(prot, "chainIDs"):
        u.add_TopologyAttr("chainIDs")
    prot.chainIDs = chain_ids
    # segids give viewers a second handle and survive more file formats
    if not hasattr(prot, "segids"):
        u.add_TopologyAttr("segid")

    ref_pdb = os.path.join(args.outdir, "view_ref.pdb")
    out_xtc = os.path.join(args.outdir, "view_traj.xtc")
    movie = os.path.join(args.outdir, "view_movie.pdb.gz")

    frames, ref = [], None
    with mda.Writer(out_xtc, n_atoms=n_prot) as W:
        for ts in u.trajectory[::args.stride]:
            box = ts.dimensions[:3]
            newpos, ref = anchored_positions(prot.positions, box,
                                             n_copies, n_beads, ref)
            prot.positions = newpos
            if not frames:
                prot.write(ref_pdb)
            W.write(prot)
            frames.append((prot.positions.copy(), ts.time / 1000.0))

    # single self-contained file that plays as a movie; gzip because an all-bead
    # multi-state PDB is ~90 MB uncompressed and PyMOL reads .pdb.gz directly
    write_pdb_states(movie, prot, frames, chain_ids)

    pml = os.path.join(args.outdir, "view.pml")
    # NOT an f-string: the PyMOL python block below contains {ch} and {i}, which
    # belong to code PyMOL runs later. An f-string tries to interpolate them at
    # write time, raises NameError, and leaves the .pml truncated to zero bytes.
    # The movie path is substituted explicitly instead, and is absolute so the
    # script works regardless of PyMOL's working directory.
    template = """load __MOVIE__, sys
hide everything
show spheres, sys
# Martini beads are ~0.47 nm across; no bonds are drawn because CG spacing does
# not look covalent to a viewer, and spheres are the honest representation.
set sphere_scale, 0.5

# Explicit palette rather than util.cbc: that helper imports an optional
# colorblindfriendly module which is absent in this PyMOL build, and when it
# fails every chain silently stays dark grey.
python
from pymol import cmd
palette = ["red","green","blue","yellow","magenta","cyan","orange","purple",
           "salmon","lime","slate","wheat","teal","hotpink","olive","marine",
           "chocolate","forest","deepblue","firebrick","limon","violet",
           "skyblue","sand","raspberry","palegreen","grey70"]
for i, ch in enumerate(sorted(cmd.get_chains("sys"))):
    cmd.color(palette[i % len(palette)], f"sys and chain {ch}")
python end

bg_color white
set ray_opaque_background, 1
set orthoscopic, 1

# A dense cloud of ~20k small spheres self-shadows badly: with default lighting
# the interior of the box renders almost black and the chain colours are lost.
# Shadows off + lifted ambient keeps every copy readable.
set ray_shadows, off
set ambient, 0.35
set direct, 0.55
set specular, 0.2

set movie_fps, 8
zoom sys, 5
mplay
"""
    with open(pml, "w") as fh:
        fh.write(template.replace("__MOVIE__", os.path.abspath(movie)))

    mb = os.path.getsize(movie) / 1e6
    print(f"wrote {ref_pdb}")
    print(f"wrote {out_xtc}   ({len(frames)} frames, stride {args.stride})")
    print(f"wrote {movie}   ({len(frames)} states, {mb:.0f} MB gzipped)")
    print(f"wrote {pml}")
    print(f"      {n_copies} copies x {n_beads} beads, one chain each")
    print()
    print(f"  play it:  pymol {pml}")
    print(f"  or:       pymol {movie}   then: show spheres; util.cbc; mplay")


if __name__ == "__main__":
    main()
