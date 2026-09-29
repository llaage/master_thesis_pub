"""Build the tidy per-residue SAP table + sequence alignment for the three
backbones kept for the thesis figures: RH5.1, 97.04 and v3hd-119.

Per-residue SAP is read from the Rosetta score JSONs rather than the comparison
CSVs, because the CSVs predate the v3hd-119 run. The JSONs reproduce the CSVs to
5e-10, so nothing changes for the two structures that appear in both.
"""
import os, json
import numpy as np, pandas as pd
from align import nw

SAPDIR = "/home/llaage/scratch/projects/rh5_folding/260420_sap_comparison/sap_scores"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
os.makedirs(OUT, exist_ok=True)

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

# label -> (score json, source pdb, chain).  Order = plotting order.
STRUCT = [
    ("RH5.1",    "4wat.json",
     "/home/llaage/projects/rh5_folding/260420_sap_comparison/4wat.pdb", "A",
     "4wat.pdb", "crystal structure"),
    ("97.04",    "MDL97_refine_007.json",
     "/home/llaage/projects/rh5_folding/260420_sap_comparison/MDL97_refine_007.pdb", "A",
     "MDL97_refine_007.pdb", "design, refined"),
    ("v3hd-119", "RH5hd3.2-119.json",
     "/home/llaage/projects/orders/rh5_comI_comIII_screen/input_structures/RH5hd3.2-119.pdb", "A",
     "RH5hd3.2-119.pdb", "design"),
]
REF = "RH5.1"

def pdb_residues(path, chain):
    out, seen = [], set()
    for line in open(path):
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        num = int(line[22:26])
        if num in seen:
            continue
        seen.add(num)
        out.append((num, AA3.get(line[17:20].strip(), "X")))
    return out

def per_residue(js):
    d = json.load(open(os.path.join(SAPDIR, js)))
    pre = "sapbf_res_sap_score_"
    rows = [(k[len(pre):-1], k[-1], float(v)) for k, v in d.items() if k.startswith(pre)]
    return d["sap_score"], pd.DataFrame(rows, columns=["resnum", "chain", "sap"]).astype({"resnum": int})

rows, seqmap, totals = [], {}, {}
print(f"{'label':<10} {'chain':<6} {'n_sap':>6} {'n_pdb':>6} {'dropped':>8}  reported_total")
for label, js, pdb, chain, fname, kind in STRUCT:
    total, pr = per_residue(js)
    pr = pr[pr.chain == chain]
    res = dict(pdb_residues(pdb, chain))
    pr["aa"] = pr.resnum.map(res)
    drop = pr.aa.isna().sum()          # ligand / glycan rows carried into the SAP output
    if drop:
        print(f"   dropped non-protein residues: {sorted(pr.resnum[pr.aa.isna()])}")
    pr = pr.dropna(subset=["aa"]).sort_values("resnum")
    pr["label"] = label
    print(f"{label:<10} {chain:<6} {len(pr) + drop:>6} {len(res):>6} {drop:>8}  {total:.4f}")
    seqmap[label] = dict(zip(pr.resnum, pr.aa))
    totals[label] = total
    rows.append(pr)
tidy = pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------- centre-star MSA
seqs = {lab: ("".join(seqmap[lab][n] for n in sorted(seqmap[lab])), sorted(seqmap[lab]))
        for lab in seqmap}
ref_seq, ref_nums = seqs[REF]
others = [lab for lab, *_ in STRUCT if lab != REF]

ins = [0] * (len(ref_seq) + 1)
pairs = {}
for lab in others:
    a_ref, a_q = nw(ref_seq, seqs[lab][0])
    pairs[lab] = (a_ref, a_q)
    ri = run = 0
    for cr, cq in zip(a_ref, a_q):
        if cr == "-":
            run += 1
        else:
            ins[ri] = max(ins[ri], run); run = 0; ri += 1
    ins[ri] = max(ins[ri], run)

col_of_ref, col = [], 0
for i in range(len(ref_seq)):
    col += ins[i]; col_of_ref.append(col); col += 1
NCOL = col + ins[len(ref_seq)]

aln_rows = []
for lab, *_ in STRUCT:
    seq, nums = seqs[lab]
    if lab == REF:
        cols = col_of_ref
    else:
        a_ref, a_q = pairs[lab]
        cols, ri, run = [], 0, 0
        for cr, cq in zip(a_ref, a_q):
            if cr != "-":
                if cq != "-": cols.append(col_of_ref[ri])
                ri += 1; run = 0
            else:
                base = col_of_ref[ri] if ri < len(col_of_ref) else NCOL
                if cq != "-": cols.append(base - ins[ri] + run)
                run += 1
    assert len(cols) == len(nums), (lab, len(cols), len(nums))
    aln_rows += [(lab, n, c) for n, c in zip(nums, cols)]

aln = pd.DataFrame(aln_rows, columns=["label", "resnum", "aln_col"])
ref_num_of_col = dict(zip(col_of_ref, ref_nums))
merged = tidy.merge(aln, on=["label", "resnum"], how="left")
merged["ref_resnum"] = merged.aln_col.map(ref_num_of_col)
merged.to_csv(f"{OUT}/tidy_sap.csv", index=False)

ident = {}
print(f"\n%% identity to {REF} (aligned columns):")
for lab in others:
    a_ref, a_q = pairs[lab]
    m = sum(1 for x, y in zip(a_ref, a_q) if x == y and x != "-")
    ov = sum(1 for x, y in zip(a_ref, a_q) if x != "-" and y != "-")
    ident[lab] = 100 * m / ov
    print(f"   {lab:<12} {ident[lab]:5.1f} %  ({ov} aligned columns)")

json.dump({"ncol": NCOL, "ref": REF,
           "order": [s[0] for s in STRUCT],
           "files": {s[0]: s[4] for s in STRUCT},
           "kind":  {s[0]: s[5] for s in STRUCT},
           "totals": totals, "identity": ident,
           "ref_num_of_col": {str(k): int(v) for k, v in ref_num_of_col.items()}},
          open(f"{OUT}/meta.json", "w"), indent=1)
print("\nalignment columns:", NCOL)
print("wrote", OUT)
