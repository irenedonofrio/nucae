#!/usr/bin/env python
"""TFBS composite dip through the dataloader output. One TF, one chromosome.

Checks that the data path carries real biological signal, independent of any
model: a bound transcription factor should show a central depletion in fragment
centre coverage.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from nucae import evaluate
from nucae.data import WindowDataset

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--h5", type=Path, required=True)
ap.add_argument("--sites", type=Path, required=True, help="TFBS BED, e.g. CTCF.bed")
ap.add_argument("--chrom", required=True)
ap.add_argument("--level", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--scalar", type=float, default=1.0,
                help="global normalising S; 1.0 leaves the dip in smoothed FCC units")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dataset = WindowDataset(a.h5, input_level=a.level, target_level=a.level)
track = evaluate.chromosome_track(dataset, a.chrom)
sites = evaluate.load_sites(a.sites, a.chrom)
dips, profiles, kept = evaluate.per_site_dips(track, sites, a.scalar)
if len(dips) == 0:
    raise SystemExit(f"no site on {a.chrom} had enough non-NaN data to measure")

point, lo, hi = evaluate.dip_ci(dips, seed=a.seed)
composite = np.nanmean(profiles, axis=0)
flank_level, centre_level = evaluate.band_levels(composite)
result = {
    "metric_version": evaluate.METRIC_VERSION, "tf": a.sites.stem,
    "chrom": a.chrom, "level": a.level, "scalar": a.scalar, "seed": a.seed,
    "n_sites_total": len(sites), "n_sites_kept": len(dips),
    "centre_bp": evaluate.CENTRE, "flank_bp": [evaluate.FLANK_LO, evaluate.FLANK_HI],
    "dip": point, "ci95": [lo, hi],
    "flank_level": flank_level, "centre_level": centre_level,
    "fwhm_bp": evaluate.footprint_fwhm(composite),
}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=1))
np.save(a.out.with_suffix(".composite.npy"), composite)
print(json.dumps(result, indent=1))
print(f"-> {a.out}")
