#!/usr/bin/env python3

import re
import sys
import argparse
from pathlib import Path
from typing import Optional

from pymol import cmd

AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA3_TO_1 = {v: k for k, v in AA1_TO_3.items()}
MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def parse_mutation(mut_str: str):
    mut_str = mut_str.strip().upper()
    m = MUT_RE.fullmatch(mut_str)
    if not m:
        raise ValueError(f"Invalid mutation '{mut_str}'. Use format like C58K")
    wt_aa, resi, mut_aa = m.groups()
    if wt_aa == mut_aa:
        raise ValueError(f"Mutation '{mut_str}' does not change the residue")
    return wt_aa, resi, mut_aa


def residue_selection(obj_name: str, resi: str, chain: Optional[str]):
    if chain:
        return "/%s//%s/%s" % (obj_name, chain, resi)
    return "/%s///%s" % (obj_name, resi)


def get_wt_residue(obj_name: str, resi: str, chain: Optional[str]):
    sel = residue_selection(obj_name, resi, chain) + "/CA"
    model = cmd.get_model(sel)

    if len(model.atom) == 0:
        if chain:
            raise ValueError("Residue %s not found in chain %s" % (resi, chain))
        raise ValueError("Residue %s not found" % resi)

    if len(model.atom) > 1 and chain is None:
        chains = sorted(set(a.chain for a in model.atom))
        raise ValueError(
            "Residue %s exists in multiple chains %s. Use --chain." % (resi, chains)
        )

    atom = model.atom[0]
    resn = atom.resn
    if resn not in AA3_TO_1:
        raise ValueError("Residue %s is not a standard amino acid: %s" % (resi, resn))

    return AA3_TO_1[resn], atom.chain


def wrap_sequence(seq: str, width: int = 80):
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def get_object_chain_sequences(obj_name: str, only_chain: Optional[str] = None):
    """
    Extract protein sequence from a PyMOL object using resolved CA atoms.
    This gives the ATOM-coordinate sequence, not SEQRES.
    """
    sel = "(%s) and polymer.protein and name CA" % obj_name
    if only_chain:
        sel += " and chain %s" % only_chain

    model = cmd.get_model(sel)

    chain_to_seq = {}
    seen_residues = set()

    for atom in model.atom:
        chain = atom.chain or "no_chain"
        resi_key = (chain, atom.resi)

        # Avoid duplicate CA atoms from alternate conformations
        if resi_key in seen_residues:
            continue
        seen_residues.add(resi_key)

        aa = AA3_TO_1.get(atom.resn.upper(), "X")
        chain_to_seq.setdefault(chain, []).append(aa)

    return {
        chain: "".join(seq)
        for chain, seq in chain_to_seq.items()
    }


def save_object_as_fasta(
    obj_name: str,
    fasta_path: Path,
    record_prefix: str,
    only_chain: Optional[str] = None,
):
    chain_sequences = get_object_chain_sequences(obj_name, only_chain=only_chain)

    if not chain_sequences:
        raise ValueError("No protein sequence found in object %s" % obj_name)

    with open(fasta_path, "w") as handle:
        for chain, seq in sorted(chain_sequences.items()):
            header = "%s|chain_%s" % (record_prefix, chain)
            handle.write(">%s\n" % header)
            handle.write("%s\n" % wrap_sequence(seq))


def mutate_one(input_pdb: Path, output_dir: Path, mut_str: str, chain: Optional[str]):
    wt_aa, resi, mut_aa = parse_mutation(mut_str)

    obj_name = "wt_tmp"
    cmd.delete("all")
    cmd.load(str(input_pdb), obj_name)

    actual_wt_aa, resolved_chain = get_wt_residue(obj_name, resi, chain)
    if actual_wt_aa != wt_aa:
        raise ValueError(
            "%s: wildtype mismatch at residue %s. PDB has %s, not %s."
            % (mut_str, resi, actual_wt_aa, wt_aa)
        )

    mut_obj = "mut_%s" % mut_str
    cmd.create(mut_obj, obj_name)

    sel = residue_selection(mut_obj, resi, resolved_chain)

    cmd.wizard("mutagenesis")
    cmd.do("refresh_wizard")
    cmd.get_wizard().set_mode(AA1_TO_3[mut_aa])
    cmd.get_wizard().do_select(sel)

    # First rotamer = most probable
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.set_wizard()

    out_stem = "%s_%s_%s" % (input_pdb.stem, resolved_chain, mut_str)

    out_pdb = output_dir / ("%s.pdb" % out_stem)
    out_fasta = output_dir / ("%s.fasta" % out_stem)

    cmd.save(str(out_pdb), mut_obj)

    # Save FASTA for the mutated chain only
    save_object_as_fasta(
        obj_name=mut_obj,
        fasta_path=out_fasta,
        record_prefix=out_stem,
        only_chain=resolved_chain,
    )

    cmd.delete(mut_obj)
    cmd.delete(obj_name)

    return out_pdb, out_fasta


def main():
    parser = argparse.ArgumentParser(
        description="Generate one single-mutant PDB per mutation using PyMOL"
    )
    parser.add_argument("input_pdb", type=Path, help="Path to WT PDB")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("mutations", nargs="+", help="Mutations like C58K A60E")
    parser.add_argument("--chain", default=None, help="Chain ID, e.g. A")

    args = parser.parse_args()

    if not args.input_pdb.is_file():
        sys.exit("Input PDB not found: %s" % args.input_pdb)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for mut in args.mutations:
        try:
            out_pdb, out_fasta = mutate_one(args.input_pdb, args.output_dir, mut, args.chain)
            print("OK   %s -> %s ; %s" % (mut, out_pdb, out_fasta))
        except Exception as e:
            failed = True
            print("FAIL %s -> %s" % (mut, e), file=sys.stderr)

    cmd.quit(1 if failed else 0)


if __name__ == "__main__":
    main()