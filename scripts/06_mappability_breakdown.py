#!/usr/bin/env python
"""Does the reconstruction score depend on whether the region is mappable?

THE QUESTION THIS ANSWERS. scripts/05 measured that ~16% of the inherited
windows on chr21 are majority-unmappable yet carry full-amplitude signal: reads
placed where short reads cannot be placed reliably. The pre-windowed format has
no mask, so that signal was trained on and is scored on as if it were real.

What that costs is not knowable from the summary metrics, because they average
over both kinds of window. This splits them.

  gain = r(recon) - r(input), per window. The quantity the result rests on.

  If the gain is FLAT across mappability, the untrustworthy windows are diluting
  the average and nothing more -- quote the number, note the limitation.

  If the gain is LARGER where the region is unmappable, the model is scoring on
  structure that exists in the pileup and not in the genome. That is the case
  that has to be said out loud, and the clean-window number becomes the one to
  report.

The mask is built from the mappability BED the inherited pipeline never applied,
so this reports which windows WOULD be flagged, not what that pipeline believed.
No counts and no sequence are read: the mask depends only on the BED and the
chromosome length, which is why this needs no HDF5 build.

COORDINATES. The window starts come from the pre-windowed file and are 1-based
(scripts/05: 1,199 of 1,425 chr21 windows align at lag -1). They are converted
here. Over 25,000 positions one base cannot change a fraction materially, but an
unexplained convention is how a real off-by-one gets normalised into the code.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nucae.data import build_mask, chrom_length

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--metrics", type=Path, required=True,
                help="test_signal_metrics.csv from scripts/04")
ap.add_argument("--mappable", type=Path, required=True, help="k100 keep-list BED")
ap.add_argument("--fai", type=Path, required=True, help="reference .fai")
ap.add_argument("--window", type=int, default=25_000)
ap.add_argument("--one-based", action="store_true", default=True,
                help="window starts are 1-based (the inherited convention)")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

a.out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(a.metrics)
# h5py byte labels survive a CSV round trip as the literal text "b'chr8'".
df["chrom"] = (df["chrom"].astype(str)
               .str.removeprefix("b'").str.removesuffix("'").str.strip())

required = {"chrom", "start", "pearson_recon", "pearson_under"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"{a.metrics}: missing columns {sorted(missing)}")

# One mask per chromosome, not per window: chr8 is 145 Mbp and rebuilding it
# 5,000 times is the whole runtime.
fractions = np.empty(len(df))
for chrom, rows in df.groupby("chrom"):
    n = chrom_length(a.fai, chrom)
    mask = build_mask(a.mappable, chrom, n)          # True = unmappable
    starts = rows["start"].to_numpy() - (1 if a.one_based else 0)
    for pos, idx in zip(starts, rows.index):
        fractions[df.index.get_loc(idx)] = mask[pos:pos + a.window].mean()
    print(f"  {chrom}: {n:,} bp, {mask.mean():.1%} unmappable, {len(rows):,} windows")

df["masked_frac"] = fractions
df["gain"] = df["pearson_recon"] - df["pearson_under"]

BINS = [-0.001, 0.01, 0.10, 0.50, 1.001]
LABELS = ["clean (<1%)", "some (1-10%)", "mixed (10-50%)", "mostly masked (>50%)"]
df["band"] = pd.cut(df["masked_frac"], bins=BINS, labels=LABELS)
df.to_csv(a.out / "mappability_breakdown.csv", index=False)

table = df.groupby("band", observed=False).agg(
    windows=("gain", "size"),
    r_recon=("pearson_recon", "median"),
    r_input=("pearson_under", "median"),
    gain=("gain", "median"),
)

lines = [
    "=" * 66,
    f"Reconstruction quality by mappability  [{a.metrics.parent.name}]",
    "=" * 66,
    f"  windows            : {len(df):,}",
    f"  median masked frac : {df['masked_frac'].median():.2%}",
    f"  windows >50% masked: {int((df['masked_frac'] > 0.5).sum()):,}"
    f"  ({(df['masked_frac'] > 0.5).mean():.1%})",
    "",
    f"  {'band':<22}{'windows':>9}{'r recon':>10}{'r input':>10}{'gain':>9}",
    "  " + "-" * 60,
]
for band, row in table.iterrows():
    n = int(row["windows"])
    if n == 0:
        lines.append(f"  {band:<22}{n:>9}{'':>10}{'':>10}{'':>9}")
        continue
    lines.append(f"  {band:<22}{n:>9}{row['r_recon']:>10.4f}"
                 f"{row['r_input']:>10.4f}{row['gain']:>9.4f}")

clean = df[df["masked_frac"] < 0.01]
dirty = df[df["masked_frac"] > 0.50]
lines += ["", "-- the comparison that matters -----------------------------------"]
if len(clean) and len(dirty):
    dg, cg = dirty["gain"].median(), clean["gain"].median()
    lines += [
        f"  median gain, clean windows     : {cg:.4f}  (n={len(clean):,})",
        f"  median gain, >50% masked       : {dg:.4f}  (n={len(dirty):,})",
        f"  difference                     : {dg - cg:+.4f}",
        "",
        ("  The gain is LARGER in unmappable regions. Some of the headline"
         if dg > cg else
         "  The gain is no larger in unmappable regions. Those windows"),
        ("  improvement comes from structure in the pileup rather than in"
         if dg > cg else
         "  dilute the average without inflating it; the clean-window"),
        ("  the genome. Report the clean-window number." if dg > cg else
         "  number is the honest headline either way."),
        "",
        f"  clean-window r (recon / input) : {clean['pearson_recon'].median():.4f}"
        f" / {clean['pearson_under'].median():.4f}",
    ]
else:
    lines.append("  one side is empty; nothing to compare.")
lines += ["",
          "  Medians throughout: a handful of centromeric windows would",
          "  otherwise decide the answer.",
          "=" * 66]

summary = "\n".join(lines)
print("\n" + summary)
(a.out / "mappability_summary.txt").write_text(summary + "\n")
print(f"\nwritten: {a.out}")
