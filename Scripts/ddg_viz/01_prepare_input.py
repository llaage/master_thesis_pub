#!/usr/bin/env python3
"""
Step 1 - build the cleaned WT input PDB for complex III (8pwx) from the deposited cif.

8pwx is PfRH5 bound to mAb R5.008, one copy in the asymmetric unit. Two things differ
from complexes I and II:

  * the deposited chains are swapped - chain A is the scFv, chain B is RH5. They are
    relabelled here so the pipeline convention holds everywhere:
        A = RH5  = target (-t A),  B = scFv = binder (-b B)
  * RH5 is numbered 157-505 with residues 245-298 disordered, i.e. the full-length
    numbering including the internal loop. 8pww used the loop-deleted (ΔNL) construct
    numbering, so the two are NOT interchangeable past residue ~244. Everything
    downstream goes through the alignment in 02_numbering_map.py.

Chain A is written first so the pose order matches complexes I and II.
"""
from pathlib import Path

from pymol import cmd

CIF = "/home/llaage/projects/rh5_folding/260513_ddg_interface_scoring/com_III/8pwx.cif"
OUT = Path("/home/llaage/projects/rh5_folding/260513_ddg_interface_scoring/com_III/inputs")

AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
          'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
          'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

cmd.set("assembly", "")           # asymmetric unit, not the biological assembly
cmd.load(CIF, "raw")

# RH5 (deposited chain B) becomes A, scFv (deposited chain A) becomes B.
cmd.create("rh5", "raw and polymer.protein and chain B")
cmd.create("scfv", "raw and polymer.protein and chain A")
cmd.alter("rh5", "chain='A'")
cmd.alter("scfv", "chain='B'")
cmd.delete("raw")

cmd.create("wt", "rh5 or scfv")
cmd.delete("rh5")
cmd.delete("scfv")

cmd.remove("wt and not alt ''+A")
cmd.alter("wt", "alt=''")
cmd.remove("wt and hydro")
cmd.alter("wt", "q=1.0")
cmd.alter("wt", "segi=''")     # drop the stale segid left behind by the chain swap
cmd.sort("wt")

OUT.mkdir(parents=True, exist_ok=True)

# PyMOL keeps the deposited atom order, which puts the scFv first. Write the two
# chains separately and concatenate so chain A (RH5) leads, as in complexes I and II.
parts = []
for ch in ("A", "B"):
    tmp = OUT / f".tmp_chain{ch}.pdb"
    cmd.save(str(tmp), f"wt and chain {ch}")
    parts.append([ln for ln in tmp.read_text().splitlines()
                  if ln.startswith(("ATOM", "HETATM", "TER"))])
    tmp.unlink()

with open(OUT / "8pwx.pdb", "w") as fh:
    serial = 0
    for block in parts:
        for ln in block:
            if ln.startswith("TER"):
                fh.write("TER\n")
                continue
            serial += 1
            fh.write(f"{ln[:6]}{serial:5d}{ln[11:]}\n")
    fh.write("END\n")

for ch, what in (("A", "RH5"), ("B", "scFv R5.008")):
    seen, seq, nums = set(), [], []
    for a in cmd.get_model(f"wt and chain {ch} and name CA").atom:
        if a.resi in seen:
            continue
        seen.add(a.resi)
        seq.append(AA3TO1.get(a.resn, "X"))
        nums.append(int(a.resi))
    seq = "".join(seq)
    gaps = [(nums[i], nums[i + 1]) for i in range(len(nums) - 1) if nums[i + 1] != nums[i] + 1]
    with open(OUT / f"8pwx_chain{ch}.fasta", "w") as fh:
        fh.write(f">8pwx|chain:{ch}|{what}\n")
        for i in range(0, len(seq), 80):
            fh.write(seq[i:i + 80] + "\n")
    print(f"chain {ch} ({what}): {len(seq)} residues, {nums[0]}-{nums[-1]}, gaps {gaps}")

print("wrote", OUT / "8pwx.pdb")
