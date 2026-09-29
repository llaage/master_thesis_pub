#!/usr/bin/env python
"""
Render a dissociation trajectory as a multi-state PDB, a PyMOL movie and an mp4.

Watching one of these is the fastest way to catch an analysis that is measuring
the wrong thing, which this project has already needed twice.

FOUR THINGS THAT MAKE A CG MOVIE LIE, all of them hit in the previous campaign
and all handled here:

1. PERIODIC WRAPPING. GROMACS writes coordinates wrapped into the cell, so a
   chain straddling a face appears in two pieces at opposite sides of the
   screen. Each chain is unwrapped separately, with the TRICLINIC minimum image
   -- these boxes are rhombic dodecahedra, and rectangular-box arithmetic tears
   a 4.6 nm domain into something spanning the full 12.8 nm cell.

2. CHAINS IN DIFFERENT IMAGES. Once whole, each chain is placed in the image
   nearest the complex, so a subunit that is really 1 nm away is not drawn a box
   width away. This is the difference between "the trimer is intact" and "the
   trimer exploded" and it is purely a rendering choice.

3. A REFERENCE THAT JUMPS. Anchoring on a fixed chain makes the whole system
   leap whenever that chain crosses a boundary -- measured last campaign at a
   mean 66 A and a maximum of 207 A against a 208 A box, while every internal
   distance stayed put. The reference here is the previous frame's centroid,
   carried forward, so motion on screen is real motion.

4. LIGHTING AND COLOUR. Thousands of small spheres ray-trace almost black under
   default lighting, so shadows go off and ambient goes up. Colours are set
   explicitly per chain: `util.cbc` fails silently in this PyMOL build, leaving
   everything grey, and Martini bead names collide with element symbols (BB
   reads as boron, SC* as sulfur) so an element column is written as carbon.

Usage:
    python scripts/make_movie.py systems/ctrl_s150_r1 --stride 25 --mp4
    python scripts/make_movie.py systems/rung1_s150_r1 --stride 25 --mp4 \\
        --split-domains          # colour RH5 and I53-50A separately
"""
import argparse
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_common as ac
import structlib as sl

# PyMOL 2.5.8, not 3.1.3. Both are installed site-wide, but only the 2.x build
# picks up the licence in ~/.pymol/license.lic; PyMOL 3 renders every frame with
# a "No License File - For Evaluation Only" banner burned into the image.
PYMOL = "/net/software/pymol-2/bin/pymol"

# Explicit, colour-blind-safe palette. util.cbc is not usable in this build.
PALETTE = ["skyblue", "salmon", "palegreen", "wheat", "lightpink",
           "paleyellow", "lightblue", "deepteal"]

# (core, RH5) pairs for --split-domains: the folded I53-50A core in a saturated
# colour, the RH5 module it carries in the pale version of the same hue. Keeping
# the hue tells you which chain a module belongs to; the lightness tells you
# whether you are looking at the trimer or at the antigen swinging off it.
DOMAIN_PAIRS = [("marine", "skyblue"), ("firebrick", "salmon"),
                ("forest", "palegreen"), ("orange", "wheat"),
                ("purple", "lightpink"), ("olive", "paleyellow")]


def write_multistate(sd, traj, out_pdb, stride, start_ns, end_ns, ref_chain=None,
                     ref_residues=None):
    """One MODEL per frame, chains whole and mutually nearest, PBC-safe."""
    chains, _, rpdb, names = ac.bead_table(sd)
    u, n_prot = ac.load(sd, traj)
    uniq = sorted(set(chains))
    groups = ac.group_indices(chains, uniq)
    ref_gi = uniq.index(ref_chain) if ref_chain in uniq else 0
    # Which beads OF the reference chain define the orientation. Restricting
    # this to a domain is what makes a fusion rung comparable with the bare one.
    ref_mask = None
    if ref_residues:
        lo, hi = (int(x) for x in ref_residues.split(":"))
        gi = groups[ref_gi]
        ref_mask = (rpdb[gi] >= lo) & (rpdb[gi] <= hi)
        if not ref_mask.any():
            raise ValueError(f"--ref-residues {ref_residues} selects nothing "
                             f"in chain {uniq[ref_gi]}")

    nj = ac.NoJump()
    ref0 = None            # reference chain in state 1, centred
    n_states = 0
    times = []
    with open(out_pdb, "w") as fh:
        fh.write(f"REMARK   {os.path.basename(sd)}  stride {stride}\n")
        fh.write("REMARK   chains unwrapped individually (triclinic minimum "
                 "image), placed in the\n")
        fh.write("REMARK   image nearest a time-continuous reference\n")
        for ts in u.trajectory[::stride]:
            t_ns = ts.time / 1000.0
            if t_ns < start_ns or (end_ns and t_ns > end_ns):
                continue
            box = ts.dimensions
            pos = u.atoms.positions

            # No-jump coordinates: a chain that leaves keeps leaving. Placing
            # each chain by the minimum image of its CURRENT position instead
            # folds a departed subunit back onto the complex, and the movie then
            # shows an intact trimer for a run whose chains are really 25 nm
            # apart with zero interface contacts. That is not a cosmetic
            # difference -- it is the whole observable.
            # The reference chain is placed first; every other chain is then
            # seeded into the periodic image nearest it. Without that seeding
            # the movie inherits the trajectory's wrapping and can show an
            # intact complex as three separated bodies (see NoJump.coords).
            parts = [None] * len(groups)
            parts[ref_gi] = nj.coords(pos, box, ref_gi, groups[ref_gi])
            ref_com_now = parts[ref_gi].mean(axis=0)
            for i, idx in enumerate(groups):
                if i == ref_gi:
                    continue
                parts[i] = nj.coords(pos, box, i, idx, ref_com=ref_com_now)

            # Lock the camera to one chain (default the first): superpose every
            # frame on that chain alone, so it sits still and everything else
            # moves relative to it. Fitting on the whole complex instead lets a
            # departing subunit drag the view after it.
            cur_ref = parts[ref_gi] if ref_mask is None else parts[ref_gi][ref_mask]
            if ref0 is None:
                ref0 = cur_ref.copy()
                shift0 = cur_ref.mean(axis=0)
                placed = [p - shift0 for p in parts]
                ref0 = ref0 - shift0
            else:
                R, t, _ = sl.kabsch(cur_ref, ref0)
                placed = [p @ R.T + t for p in parts]

            n_states += 1
            times.append(t_ns)
            fh.write(f"MODEL     {n_states:4d}\n")
            k = 0
            for gi, idx in enumerate(groups):
                ch = uniq[gi]
                for j in range(len(idx)):
                    x, y, z = placed[gi][j]
                    k += 1
                    # element column forced to C: Martini bead names collide
                    # with the periodic table and PyMOL would colour BB as boron
                    fh.write(f"ATOM  {k % 100000:5d} {names[idx[j]]:<4s}"
                             f"{'BEA':>4s} {ch}{(rpdb[idx[j]]) % 10000:4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                             f"{'':10s} C\n")
            fh.write("ENDMDL\n")
    return uniq, n_states, times


def write_pml(out_pml, pdb, uniq, tag, domains=None, mp4_dir=None, n_states=1,
              groups=None, view=None, lock_zoom=False, turn=None,
              colsel=None):
    """
    PyMOL script. Built by concatenation, never by f-string.

    A .pml written as an f-string tried to interpolate a PyMOL-side variable,
    raised NameError, and left the file truncated to zero bytes -- a silent
    failure that only showed up downstream (pitfalls.md).
    """
    L = []
    A = L.append
    A("load " + pdb + ", traj")
    A("hide everything")
    A("show spheres, traj")
    # Martini regular bead sigma is 0.47 nm; draw at the real radius so touching
    # spheres mean touching protein
    A("alter traj, vdw=2.35")
    A("rebuild")
    A("bg_color white")
    A("set ray_shadows, 0")
    A("set ambient, 0.35")
    A("set direct, 0.6")
    A("set specular, 0.2")
    A("set sphere_scale, 1.0")
    A("set all_states, off")

    if colsel:
        for spec in colsel.split(";"):
            sel, colour = spec.rsplit("=", 1)
            A("color " + colour.strip() + ", traj and (" + sel.strip() + ")")
    elif groups:
        # explicit chain groups, e.g. "ABC:marine,DEFGH:salmon" -- the way to
        # make a two-body dissociation legible when there are eight chains
        for spec, colour in groups:
            for ch in spec:
                A("color " + colour + ", traj and chain " + ch)
    elif domains:
        for i, ch in enumerate(uniq):
            core_c, rh5_c = DOMAIN_PAIRS[i % len(DOMAIN_PAIRS)]
            A("color grey70, traj and chain " + ch)
            A("color " + rh5_c + ", traj and chain " + ch +
              " and resi " + domains["rh5"])
            A("color red, traj and chain " + ch + " and resi " + domains["linker"])
            A("color " + core_c + ", traj and chain " + ch +
              " and resi " + domains["core"])
    else:
        for i, ch in enumerate(uniq):
            A("color " + PALETTE[i % len(PALETTE)] + ", traj and chain " + ch)

    # Frame on the FIRST state only. Zooming to fit every state would shrink
    # an intact complex to a dot just because one chain later flies 25 nm away;
    # locking the scale to the assembled starting structure lets departing
    # chains simply leave the frame, which reads correctly.
    if view:
        # Identical camera to another system, so the two can be compared side by
        # side rather than merely described as similar.
        A("set_view (" + view + ")")
        if not lock_zoom:
            # A borrowed view matrix carries the SOURCE system's zoom and
            # clipping as well as its rotation. Reusing rung 1's zoom -- fitted
            # to a 7 nm bare trimer -- on the ~15 nm fusion cuts the RH5 modules
            # off at the frame edge. `zoom` re-fits distance and clipping while
            # leaving the rotation untouched, so the orientation is shared and
            # nothing is clipped. --view-lock-zoom keeps the source scale
            # instead, for a strict like-for-like size comparison.
            A("create _f1, traj, 1, 1")
            A("zoom _f1, 6")
            A("delete _f1")
    else:
        A("create _f1, traj, 1, 1")
        A("orient _f1")
        A("zoom _f1, 6")
        A("delete _f1")
    if turn:
        # After the view and the zoom, so it rotates the finished camera rather
        # than something `zoom` would then override.
        for t in turn.split(";"):
            ax, deg = t.split(",")
            A("turn " + ax.strip() + ", " + deg.strip())
    A("set movie_fps, 15")
    # "mset 1 -N" plays states 1..N. "mset 1 xN" -- which is what this said --
    # means "state 1, repeated N times", so every frame of every movie was the
    # starting structure. The rendering looked perfectly healthy: an intact
    # trimer for 5 us, including for a run whose chains genuinely end up 38 nm
    # apart with zero interface contacts.
    A("mset 1 -" + str(n_states))
    if mp4_dir:
        A("set ray_trace_frames, 1")
        A("set cache_frames, 0")
        A("mpng " + os.path.join(mp4_dir, "frm") + ", width=900, height=700")
        A("quit")
    with open(out_pml, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system_dir")
    ap.add_argument("--traj", default="prod/prod.xtc")
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--start-ns", type=float, default=0.0)
    ap.add_argument("--end-ns", type=float, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--color-sel", default=None,
                    help="arbitrary PyMOL selections, ';'-separated as "
                         "'selection=colour'. Needed when chains alone cannot "
                         "express the grouping -- in rung 5 the pentamer is "
                         "numbered 1-159, which falls inside the fusion's RH5 "
                         "range 1-322, so --split-domains would colour the "
                         "pentamer as if it were RH5.")
    ap.add_argument("--color-groups", default=None,
                    help="colour chain sets rather than individual chains, "
                         "e.g. 'ABC:marine,DEFGH:salmon'")
    ap.add_argument("--ref-chain", default=None,
                    help="chain the camera is locked to (default: the first)")
    ap.add_argument("--ref-residues", default=None,
                    help="restrict the frame-to-frame superposition to this PDB "
                         "residue range, e.g. 339:554. Needed to put a fusion "
                         "rung in the same frame as the bare domain: aligning "
                         "on the WHOLE chain lets RH5, swinging on its linker, "
                         "drive the orientation, so the shared core ends up "
                         "pointing somewhere different in every frame.")
    ap.add_argument("--view", default=None,
                    help="explicit 18-float PyMOL view matrix instead of "
                         "orient/zoom, so two systems can be rendered from "
                         "exactly the same camera")
    ap.add_argument("--turn", default=None,
                    help="camera rotation applied AFTER the view is set, as "
                         "'axis,degrees' or several separated by ';', e.g. "
                         "'y,180' to look at the far side. Applied to the "
                         "camera, so the same --view plus the same --turn puts "
                         "two systems in the same frame.")
    ap.add_argument("--view-lock-zoom", action="store_true",
                    help="with --view, keep the source system's zoom too "
                         "(identical scale; may clip a larger system)")
    ap.add_argument("--suffix", default="",
                    help="append to the output name, to avoid overwriting an "
                         "existing movie for the same system")
    ap.add_argument("--no-cartoon", action="store_true",
                    help="skip the companion CA-renamed PDB. Off by default: "
                         "BOTH nomenclatures are kept, because neither is a "
                         "superset of the other -- the topology and every "
                         "analysis script speak BB/SC, and PyMOL will only draw "
                         "a backbone from CA with real residue names.")
    ap.add_argument("--no-clock", action="store_true",
                    help="omit the burned-in simulation time")
    ap.add_argument("--split-domains", action="store_true",
                    help="colour RH5 / linker / I53-50A separately (fusion rungs)")
    a = ap.parse_args()

    sd = os.path.abspath(a.system_dir.rstrip("/"))
    tag = os.path.basename(sd) + a.suffix
    base = os.path.dirname(os.path.dirname(sd))
    out = a.outdir or os.path.join(base, "analysis", "video")
    os.makedirs(out, exist_ok=True)
    traj = os.path.join(sd, a.traj)
    if not os.path.isfile(traj):
        sys.exit(f"ERROR: no trajectory at {traj}")

    pdb = os.path.join(out, tag + ".pdb")
    print(f">>> {tag}: writing states (stride {a.stride})")
    uniq, n_states, times = write_multistate(sd, traj, pdb, a.stride,
                                             a.start_ns, a.end_ns, a.ref_chain,
                                             a.ref_residues)
    print(f"    {n_states} states, chains {''.join(uniq)}, "
          f"{times[0]:.0f}-{times[-1]:.0f} ns  -> {pdb}")

    domains = None
    if a.split_domains:
        domains = {"rh5": "1-322", "linker": "323-338", "core": "339-554"}
    groups = None
    if a.color_groups:
        groups = [(g.split(":")[0], g.split(":")[1])
                  for g in a.color_groups.split(",")]

    pml = os.path.join(out, tag + ".pml")
    write_pml(pml, pdb, uniq, tag, domains, None, n_states, groups, a.view,
              a.view_lock_zoom, a.turn, a.color_sel)
    print(f"    {pml}   (open with: pymol {pml})")

    # Companion CA-named copy, written alongside the BB-named original and
    # never in place of it. A Martini-named PDB loads into PyMOL with zero bonds
    # -- for an unrecognised residue name PyMOL bonds atoms only within a
    # residue -- so it can be shown as spheres but never as a backbone. The two
    # files serve different tools and both are kept.
    if not a.no_cartoon:
        conv = [sys.executable, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "make_cartoon_pdb.py"), pdb,
            "--system-dir", sd]
        if a.split_domains:
            conv.append("--split-domains")
        if a.color_groups:
            conv += ["--color-groups", a.color_groups]
        r = subprocess.run(conv, capture_output=True, text=True)
        if r.returncode:
            print("    WARNING: cartoon PDB not written:")
            print((r.stderr or r.stdout)[-400:])
        else:
            print("    " + r.stdout.strip().split("\n")[0].replace(">>> ", "+ also wrote "))

    if a.mp4:
        frm = os.path.join(out, tag + "_frames")
        os.makedirs(frm, exist_ok=True)
        pml_r = os.path.join(out, tag + "_render.pml")
        write_pml(pml_r, pdb, uniq, tag, domains, frm, n_states, groups, a.view,
                  a.view_lock_zoom, a.turn, a.color_sel)
        print("    rendering frames ...")
        r = subprocess.run([PYMOL, "-cq", pml_r], capture_output=True, text=True)
        pngs = sorted(f for f in os.listdir(frm) if f.endswith(".png"))
        if not pngs:
            print("    ERROR: no frames rendered")
            print((r.stderr or r.stdout)[-800:])
            return 1
        mp4 = os.path.join(out, tag + ".mp4")
        # burn in the simulation clock: a dissociation movie is uninterpretable
        # without knowing whether a separation happened at 50 ns or 4 us
        vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        if not a.no_clock:
            # ffmpeg's eif() only formats integers -- 'f' raises
            # "Invalid format 'f' specified, allowed values: 'x','X','d','u'" --
            # so the clock counts in whole nanoseconds rather than fractional
            # microseconds. Readable either way at 5000 ns full scale.
            per = int(round(times[1] - times[0])) if len(times) > 1 else 1
            t0 = int(round(times[0]))
            font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            vf += (":drawtext=fontfile=" + font +
                   ":text='t = %{eif\\:" + str(t0) + "+n*" + str(per) +
                   "\\:d} ns':x=24:y=20:fontsize=30:fontcolor=black")
            vf = vf.replace('2:drawtext', '2,drawtext')
        cmd = ["ffmpeg", "-y", "-framerate", str(a.fps),
               "-i", os.path.join(frm, "frm%04d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", vf, mp4]
        rr = subprocess.run(cmd, capture_output=True, text=True)
        if rr.returncode != 0:
            print("    ffmpeg failed:"); print(rr.stderr[-800:]); return 1
        mb = os.path.getsize(mp4) / 1e6
        print(f"    {len(pngs)} frames -> {mp4} ({mb:.1f} MB, "
              f"{len(pngs)/a.fps:.0f} s at {a.fps} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
