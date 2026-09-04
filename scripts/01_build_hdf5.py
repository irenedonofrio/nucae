#!/usr/bin/env python
"""Build one sample's HDF5: raw counts, mask, and sequence, one chromosome at a time.

Run by hand, checking the output before the next step. Memory is one chromosome
per pass -- chr1 is 249 Mbp and dominates.
"""
import argparse
from pathlib import Path

from nucae import data

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--sample", required=True)
ap.add_argument("--level", required=True, help="coverage level, e.g. full")
ap.add_argument("--counts-dir", type=Path, required=True,
                help="directory holding {chrom}_counts.tsv.gz")
ap.add_argument("--fasta", type=Path, required=True, help=".fai must sit beside it")
ap.add_argument("--mappable", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True, help="one file per sample, e.g. BH01.h5")
ap.add_argument("--chrom", action="append", help="repeatable; default all autosomes")
ap.add_argument("--nominal-depth", type=float, default=float("nan"))
ap.add_argument("--measured-depth", type=float, default=float("nan"))
ap.add_argument("--n-fragments", type=int, default=-1)
ap.add_argument("--bam-path", default="")
ap.add_argument("--gc-table-md5", default="")
ap.add_argument("--subtype", default="")
ap.add_argument("--tumour-fraction", type=float, default=float("nan"))
ap.add_argument("--sex", default="")
ap.add_argument("--cohort", default="")
a = ap.parse_args()

level_attrs = {"nominal_depth": a.nominal_depth, "measured_depth": a.measured_depth,
               "gc_table_md5": a.gc_table_md5, "bam_path": a.bam_path,
               "n_fragments": a.n_fragments}
sample_attrs = {"subtype": a.subtype, "tumour_fraction": a.tumour_fraction,
                "sex": a.sex, "cohort": a.cohort}

a.out.parent.mkdir(parents=True, exist_ok=True)
h5 = data.open_h5(a.out, reference_genome=a.fasta.name, sample_attrs=sample_attrs)
print(f"{a.sample}/{a.level} -> {a.out}")
for chrom in (a.chrom or data.AUTOSOMES):
    data.build_chromosome(h5, a.sample, a.level, chrom,
                          a.counts_dir / f"{chrom}_counts.tsv.gz",
                          a.fasta, a.mappable, level_attrs)
h5.close()
print(f"done: {a.out} ({a.out.stat().st_size / 1e6:.0f} MB)")
