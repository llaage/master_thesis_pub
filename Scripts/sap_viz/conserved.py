"""Which aligned positions are hotspots in *every* backbone.

Those get named once, in the reference (RH5.1) numbering, so the same physical
site carries the same name in every panel instead of three different numbers.
"""
import pandas as pd

def conserved_hotspots(df, meta, thresh=2.0):
    ORDER, ref = meta["order"], meta["ref"]
    piv = df.pivot_table(index="aln_col", columns="label", values="sap")[ORDER]
    aa  = df.pivot_table(index="aln_col", columns="label", values="aa", aggfunc="first")[ORDER]
    ok  = piv.notna().all(1) & (piv >= thresh).all(1)
    refnum = {int(k): v for k, v in meta["ref_num_of_col"].items()}
    cols = [int(c) for c in piv.index[ok]]
    # every conserved column happens to carry the same residue in all three;
    # fall back to listing the variants if that ever stops being true
    tag = {}
    for c in cols:
        letters = {aa.loc[c, l] for l in ORDER}
        tag[c] = f"{aa.loc[c, ref]}{int(refnum[c])}" if len(letters) == 1 \
                 else f"{'/'.join(sorted(letters))}{int(refnum[c])}"
    return cols, tag

def stagger(cols, min_sep, levels=3):
    """Assign each label the lowest free row so close labels do not overlap."""
    last, lvl = {}, {}
    for c in sorted(cols):
        for L in range(levels):
            if c - last.get(L, -1e9) >= min_sep:
                lvl[c] = L; last[L] = c; break
        else:
            lvl[c] = levels - 1; last[levels - 1] = c
    return lvl
