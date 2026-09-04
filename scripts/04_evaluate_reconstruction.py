#!/usr/bin/env python
"""Evaluate a checkpoint on the test split. THE REPRODUCTION GATE.

Reproduces the outputs of the cluster's evaluate.py so a checkpoint trained
there can be re-scored here and the numbers compared against the
test_summary.txt already on disk. Run this before training anything new: if it
does not reproduce, the port is wrong and no new result would be interpretable.

Reported per window: MSE, Pearson r of the reconstruction, and Pearson r of the
RAW UNDERSAMPLED INPUT against the same target. The input baseline is the
comparison that matters -- a reconstruction r of 0.9 means nothing until you
know what the input already scored.

Peaks come from the frozen nucae/MaxFinder.py at (375, 35, 10), the parameters
the earlier work used -- not get_peaks' own defaults, which differ.

Windows are scored one at a time, batch size 1, as the original did. Batching
would be faster but can shift the last decimals through different kernels, and
this script exists to compare decimals.

    python scripts/04_evaluate_reconstruction.py \
        --data cfdna_chr89.h5 --ckpt epoch=49-val_loss=0.0017.ckpt --out results/eval_v1
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from nucae.MaxFinder import get_peaks
from nucae.metrics import nearest_peak_distances, pearson
from nucae.module import NucAEModule
from nucae.prewindowed import CfDNAWindows

PEAK_PARAMS = (375, 35, 10)      # window, max_gap, min_len

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", type=Path, required=True, help="pre-windowed cfdna_*.h5")
ap.add_argument("--ckpt", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--dual-stream", action="store_true",
                help="required for a V3 checkpoint: its module saved no such hparam")
ap.add_argument("--limit", type=int, default=None, help="first N windows, for a smoke test")
a = ap.parse_args()

a.out.mkdir(parents=True, exist_ok=True)
predictions_path = a.out / "predictions.tsv.gz"
truth_path = a.out / "ground_truth.tsv.gz"
for path in (predictions_path, truth_path):
    path.unlink(missing_ok=True)          # appended to below; never extend an old run

overrides = {"dual_stream": True} if a.dual_stream else {}
model = NucAEModule.load_from_checkpoint(a.ckpt, map_location="cpu", **overrides).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# use_sequence comes from the checkpoint, never from a flag: a no-sequence
# checkpoint fed six channels would fail, and the reverse would train-test skew.
dataset = CfDNAWindows(a.data, "test", use_sequence=model.hparams.use_sequence)
n_windows = len(dataset) if a.limit is None else min(a.limit, len(dataset))
print(f"{a.ckpt.name}: {n_windows} test windows, use_sequence={model.hparams.use_sequence}")


def append_bedgraph(path: Path, chrom: str, start: int, signal: np.ndarray) -> None:
    """Append one window in the four-column bedGraph the earlier pipeline used."""
    with gzip.open(path, "at") as handle:
        handle.writelines(f"{chrom}\t{start + i}\t\t{value}\n"
                          for i, value in enumerate(signal))


signal_rows, peak_rows = [], []
for i in range(n_windows):
    x, y = dataset[i]
    with torch.no_grad():
        prediction = model(x.unsqueeze(0).to(device))

    original = y.squeeze().cpu().numpy()
    undersampled = x[0].cpu().numpy()
    reconstructed = prediction.squeeze().cpu().numpy()

    meta = dataset.metadata(i)
    append_bedgraph(predictions_path, meta["chrom"], meta["start"], reconstructed)
    append_bedgraph(truth_path, meta["chrom"], meta["start"], original)

    signal_rows.append({
        "window_idx": i, "chrom": meta["chrom"], "start": meta["start"],
        "mse_recon": float(np.mean((reconstructed - original) ** 2)),
        "pearson_recon": pearson(reconstructed, original),
        "pearson_under": pearson(undersampled, original),
    })

    peaks = {name: get_peaks(signal, *PEAK_PARAMS)[0] for name, signal in
             (("orig", original), ("recon", reconstructed), ("under", undersampled))}
    if any(len(p) == 0 for p in peaks.values()):
        continue                          # not measurable here; counted as skipped below

    distances = {name: np.median(nearest_peak_distances(peaks[name], peaks["orig"]))
                 for name in ("recon", "under")}
    peak_rows.append({
        "window_idx": i, "chrom": meta["chrom"], "start": meta["start"],
        # Peak COUNTS are not decoration: nearest_peak_distances is one-directional,
        # so a distance is uninterpretable without knowing how many peaks were called.
        "n_peaks_orig": len(peaks["orig"]), "n_peaks_recon": len(peaks["recon"]),
        "n_peaks_under": len(peaks["under"]),
        "median_dist_recon": float(distances["recon"]),
        "median_dist_under": float(distances["under"]),
        "improvement_bp": float(distances["under"] - distances["recon"]),
        "beats_input": bool(distances["recon"] < distances["under"]),
    })

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{n_windows}")

df_signal = pd.DataFrame(signal_rows)
df_peaks = pd.DataFrame(peak_rows)
df_signal.to_csv(a.out / "test_signal_metrics.csv", index=False)
df_peaks.to_csv(a.out / "test_peak_distances.csv", index=False)

# Paired across windows: each window supplies both arms, so the test asks whether
# reconstruction beats its own input, not whether two populations differ.
statistic, p_two = wilcoxon(df_peaks["median_dist_under"], df_peaks["median_dist_recon"],
                            alternative="two-sided")
_, p_one = wilcoxon(df_peaks["median_dist_under"], df_peaks["median_dist_recon"],
                    alternative="greater")

summary = "\n".join([
    "=" * 55,
    f"Test set evaluation - {a.ckpt.name}",
    "=" * 55,
    f"  windows scored            : {len(df_signal)}",
    f"  windows without peaks     : {len(df_signal) - len(df_peaks)} (skipped)",
    "",
    "-- signal --------------------------------------------",
    f"  MSE  (recon vs original)  : {df_signal['mse_recon'].mean():.5f} +- "
    f"{df_signal['mse_recon'].std():.5f}",
    f"  Pearson r (recon)         : {df_signal['pearson_recon'].mean():.4f} +- "
    f"{df_signal['pearson_recon'].std():.4f}",
    f"  Pearson r (input)         : {df_signal['pearson_under'].mean():.4f} +- "
    f"{df_signal['pearson_under'].std():.4f}",
    "",
    "-- peak position -------------------------------------",
    f"  median dist recon (bp)    : {df_peaks['median_dist_recon'].median():.2f}",
    f"  median dist input (bp)    : {df_peaks['median_dist_under'].median():.2f}",
    f"  mean improvement (bp)     : {df_peaks['improvement_bp'].mean():.2f}",
    f"  windows recon beats input : {df_peaks['beats_input'].mean() * 100:.1f}%",
    f"  peaks called orig/recon   : {df_peaks['n_peaks_orig'].median():.0f} / "
    f"{df_peaks['n_peaks_recon'].median():.0f}  (median; the distance above is",
    "                              one-directional, so extra peaks are free)",
    "",
    "-- Wilcoxon signed-rank ------------------------------",
    "  H0: no difference in peak distance",
    f"  statistic                 : {statistic:.4f}",
    f"  p two-sided               : {p_two:.4g}",
    f"  p one-sided               : {p_one:.4g}",
    "=" * 55,
])
print("\n" + summary)
(a.out / "test_summary.txt").write_text(summary + "\n")
print(f"\nwritten: {a.out}")
