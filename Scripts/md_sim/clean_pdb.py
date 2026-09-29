#!/usr/bin/env python
"""
Clean an X-ray PDB for martinize2 input (workflow v2, §4.1 prep step).

- Drops all HETATM records (crystallographic waters, buffer ions/sulfates,
  etc.) -- solvent/ions are added back later at the CG level via
  gmx solvate/genion, so none of this is needed.
- Resolves alternate side-chain conformers: keeps altLoc ' ' and 'A',
  drops 'B' (and any other non-primary altLoc), and clears the altLoc
  column on the kept atoms so downstream tools see a single conformer.
- Keeps ATOM/TER/END/SSBOND records; drops REMARK/HEADER/CRYST1 clutter
  (CRYST1 crystal-symmetry info is meaningless once we discard packing and
  build our own synthetic multi-copy box in a later step -- and a
  Z-value-incomplete CRYST1 line trips a martinize2 warning for no benefit).

Usage: python clean_pdb.py <input.pdb> <output.pdb>
"""
import sys

KEEP_ALTLOC = {" ", "A"}


def clean(in_path, out_path):
    n_atom_in = n_atom_out = n_hetatm_dropped = n_altloc_dropped = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = line[:6]
            if rec in ("SSBOND", "TER   ", "END   ", "TER\n", "END\n"):
                fout.write(line)
                continue
            if rec == "HETATM":
                n_hetatm_dropped += 1
                continue
            if rec == "ATOM  ":
                n_atom_in += 1
                altloc = line[16]
                if altloc not in KEEP_ALTLOC:
                    n_altloc_dropped += 1
                    continue
                if altloc != " ":
                    line = line[:16] + " " + line[17:]
                fout.write(line)
                n_atom_out += 1
                continue
            # skip everything else (REMARK, HEADER, SEQRES, etc.)
    print(f"ATOM records: {n_atom_in} -> {n_atom_out} "
          f"(dropped {n_altloc_dropped} non-primary altLoc atoms)")
    print(f"HETATM records dropped: {n_hetatm_dropped}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <input.pdb> <output.pdb>")
    clean(sys.argv[1], sys.argv[2])
