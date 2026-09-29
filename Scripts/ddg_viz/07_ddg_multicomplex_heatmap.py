#!/usr/bin/env python3
"""
Step 7 - per-complex ΔΔG heatmap.

The generator is shared by every complex and lives in com_II/scripts, because it has
to read all of their mutation_summary.csv files at once. It writes the same figures
into each complex's output/comparison_plots/, so running it from here or from com_II
is equivalent. This file exists so com_III/scripts has the full numbered sequence.
"""
import runpy

runpy.run_path(
    "/home/llaage/projects/rh5_folding/260513_ddg_interface_scoring/com_II/scripts/"
    "07_ddg_multicomplex_heatmap.py",
    run_name="__main__",
)
