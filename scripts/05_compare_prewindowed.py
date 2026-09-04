#!/usr/bin/env python
"""Measure how far the counts-derived pipeline is from the inherited windows.

NOT A TEST. There is no pass criterion here and there should not be: the two
files were built by different code from different intermediates, and the point
is to put numbers on the difference rather than to assert it is small.

    old   old_nucae/convert_to_hdf5.py -> /windows/{split}/{full,...}
          25 kb windows, already smoothed and already scaled by an upstream
          pipeline whose normalisation is not recorded anywhere.

    new   scripts/01_build_hdf5.py -> /levels/{level}/{chrom}/counts_gc
          raw counts on the reference axis; smoothing happens here, through
          nucae.preprocess.smooth, exactly as the dataloader does it.

For every old window on `--chrom`, the same coordinates are cut from the new
file with a HALO on each side, smoothed, and cropped back -- the dataloader's
own path, so what is compared is what a model would actually be fed.

WHAT IS MEASURED, and why each one is here:

  lag         Offset in bp maximising correlation, searched over +-MAX_LAG.
              MEASURED, NEVER ASSUMED: the old windows carry a `start` whose
              base convention is not documented, and a whole-pipeline off-by-one
              is invisible in every other number in this table.
  on_grid     Whether the old start lies on the new tiling (multiples of 25,000
              from 0). If it does not, the two pipelines never cut the same
              windows and every other column compares neighbours, not pairs.
  r           Pearson of old against new at the best lag. Shape agreement,
              blind to amplitude by construction.
  scale       Least-squares a in old ~ a * new, i.e. cov/var. This is the one
              number that estimates the unrecorded upstream normalisation; a
              constant across windows means it was a single global factor.
  masked_frac Fraction of the window the new mask calls unmappable. The old
              format has no mask, so this is signal it stored as real.

Reported per window to a CSV and summarised as medians. Medians, not means:
a handful of centromeric windows would otherwise decide the answer.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from nucae.preprocess import HALO, smooth
from nucae.data import WINDOW

MAX_LAG = 20          # bp; a real convention mismatch is 1, a real misassembly is not 20
SPLITS = ("train", "val", "test")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--old", type=Path, required=True, help="pre-windowed cfdna_*.h5")
ap.add_argument("--new", type=Path, required=True, help="built by scripts/01_build_hdf5.py")
ap.add_argument("--chrom", default="chr21",
                help="chromosome name in the NEW file, i.e. the reference .fai name")
ap.add_argument("--old-chrom", default=None,
                help="label in the OLD file, when it differs (e.g. `21` vs `chr21`). "
                     "Defaults to --chrom. Never guessed: the two files were built "
                     "from different intermediates and a silent fallback would "
                     "compare whatever happened to match.")
ap.add_argument("--level", default="full", help="level in the new file to compare against")
ap.add_argument("--old-arm", default="full", choices=("full", "under"),
                help="which arm of the old window pair; `full` is the target")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--limit", type=int, default=None, help="first N windows")
a = ap.parse_args()
old_chrom = a.old_chrom if a.old_chrom is not None else a.chrom

a.out.mkdir(parents=True, exist_ok=True)


def old_windows(h5: h5py.File, chrom: str):
    """Every window labelled `chrom`, as (split, row index, start).

    Read from `chrom` and `start` rather than assuming an ordering: the file's
    own builder is known to have shifted chromosome LABELS on a parse failure
    (docs/OPEN_ISSUES.md entry 8), so a contiguous run cannot be taken on trust.
    """
    found = []
    target = chrom.encode()
    for split in SPLITS:
        group = h5.get(f"windows/{split}")
        if group is None:
            continue
        labels = group["chrom"][:]
        starts = group["start"][:]
        for i in np.flatnonzero(labels == target):
            found.append((split, int(i), int(starts[i])))
    found.sort(key=lambda row: row[2])
    return found


with h5py.File(a.old, "r") as old_h5, h5py.File(a.new, "r") as new_h5:
    key = f"levels/{a.level}/{a.chrom}/counts_gc"
    if key not in new_h5:
        available = sorted(new_h5.get(f"levels/{a.level}", {}))
        raise SystemExit(
            f"{a.new.name}: no {key}\n"
            f"  chromosomes present under levels/{a.level}: {available or '(none)'}\n"
            f"  levels present: {sorted(new_h5.get('levels', {}))}\n"
            f"  --chrom names the NEW file; use --old-chrom for the old file's label."
        )
    counts = new_h5[key][:].astype(np.float64)
    mask = new_h5[f"shared/{a.chrom}/mask"][:].astype(bool)
    n_bp = counts.size
    windows = old_windows(old_h5, old_chrom)
    if not windows:
        labels = set()
        for split in SPLITS:
            group = old_h5.get(f"windows/{split}")
            if group is not None and "chrom" in group:
                labels |= {v.decode() for v in np.unique(group["chrom"][:])}
        raise SystemExit(
            f"{a.old.name}: no windows labelled {old_chrom!r}\n"
            f"  labels present: {sorted(labels) or '(none)'}\n"
            f"  pass --old-chrom with one of these."
        )
    if a.limit:
        windows = windows[:a.limit]
    print(f"{a.chrom}: {n_bp:,} bp in {a.new.name}, "
          f"{len(windows):,} windows labelled {old_chrom!r} in {a.old.name}")

    rows = []
    for split, i, start in windows:
        stored = old_h5[f"windows/{split}/{a.old_arm}"][i].astype(np.float64)
        length = stored.size
        lo, hi = start - HALO, start + length + HALO
        if lo < 0 or hi > n_bp:
            # No halo available: smoothing the edge would compare a haloed
            # window against an unhaloed one, which is a different quantity.
            rows.append({"split": split, "row": i, "start": start,
                         "on_grid": start % WINDOW == 0, "skipped": "edge"})
            continue

        # The dataloader's path: smooth with halo, then crop it away.
        smoothed = smooth(counts[lo:hi].copy(), mask[lo:hi]).astype(np.float64)

        # Slide, and let the data say where it lines up.
        best = max(
            ((np.corrcoef(stored, smoothed[HALO + lag: HALO + lag + length])[0, 1], lag)
             for lag in range(-MAX_LAG, MAX_LAG + 1)),
            key=lambda pair: (-np.inf if np.isnan(pair[0]) else pair[0]))
        r, lag = best
        aligned = smoothed[HALO + lag: HALO + lag + length]

        variance = float(np.var(aligned))
        scale = float(np.cov(stored, aligned)[0, 1] / variance) if variance > 0 else np.nan
        rows.append({
            "split": split, "row": i, "start": start,
            "on_grid": start % WINDOW == 0,
            "skipped": "",
            "lag": lag, "r": float(r), "scale": scale,
            "std_old": float(np.std(stored)), "std_new": float(np.std(aligned)),
            "max_abs_diff_scaled": float(np.abs(stored - scale * aligned).max()),
            "masked_frac": float(mask[start:start + length].mean()),
        })

df = pd.DataFrame(rows)
df.to_csv(a.out / f"compare_{a.chrom}.csv", index=False)

scored = df[df["skipped"] == ""]
lines = [
    "=" * 60,
    f"{a.old.name}  vs  {a.new.name}",
    f"  [new {a.chrom} / old {old_chrom}, level={a.level}, arm={a.old_arm}]",
    "=" * 60,
    f"  windows compared          : {len(scored)} of {len(df)}"
    f" ({int((df['skipped'] == 'edge').sum())} skipped, no halo)",
    f"  old starts on the new grid: {int(scored['on_grid'].sum())} / {len(scored)}",
]
if len(scored):
    lag_counts = scored["lag"].value_counts().sort_index()
    lines += [
        "",
        "-- alignment -----------------------------------------------",
        f"  best lag (median)         : {scored['lag'].median():.0f} bp",
        "  lag distribution          : "
        + ", ".join(f"{int(k):+d}:{int(v)}" for k, v in lag_counts.items()),
        "   A median lag of 0 means the coordinate conventions agree.",
        "   +-1 means a base-convention mismatch somewhere upstream.",
        "",
        "-- agreement -----------------------------------------------",
        f"  Pearson r (median)        : {scored['r'].median():.4f}",
        f"  scale old/new (median)    : {scored['scale'].median():.4f}",
        f"  scale IQR                 : {scored['scale'].quantile(.25):.4f}"
        f" .. {scored['scale'].quantile(.75):.4f}",
        "   A tight IQR means one global factor explains the old",
        "   normalisation; a wide one means it was per-window.",
        f"  std old / std new (median): "
        f"{(scored['std_old'] / scored['std_new']).median():.4f}",
        f"  max|old - scale*new| (med): {scored['max_abs_diff_scaled'].median():.6f}",
        "",
        "-- what the mask would remove ------------------------------",
        f"  masked fraction (median)  : {scored['masked_frac'].median():.4%}",
        f"  windows >10% masked       : {int((scored['masked_frac'] > 0.1).sum())}",
        "   The old format has no mask; these positions were stored as",
        "   real signal and trained on.",
    ]
lines.append("=" * 60)
summary = "\n".join(lines)
print("\n" + summary)
(a.out / f"compare_{a.chrom}_summary.txt").write_text(summary + "\n")
print(f"\nwritten: {a.out}")
