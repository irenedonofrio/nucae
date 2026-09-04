"""Evaluation metrics. Ported from pipeline_audit/metric.py, LOCKED 2026-08-24.

    d = mean(profile, flank band) - mean(profile, centre band)

in units of raw GC-corrected FCC divided by S. Bands are frozen: centre is
+/- 30 bp, flank is 1000-2000 bp on both sides.

NOTHING HERE IS SCALE-FREE, AND THAT IS THE POINT. Pearson, Spearman and
z-scored anything measure SHAPE agreement only and must never be used to make a
claim about amplitude. `match` below is a shape metric and carries the same
warning: it is one-directional, so nothing in it penalises extra peaks and a
signal with peaks everywhere scores well.

Ported from `metric.py`, not `metrics.py` -- the latter defines the same
quantities with different bands and carries a second `footprint_fwhm` with a
different signature. `match` is from `peak_calling.py`, not the near-duplicate
in `peak_calling_comparison.py`.

Subtraction, not a ratio: after the running median is subtracted the flank sits
at ~0, so any divide-by-baseline definition explodes or flips sign. Subtraction
is the only definition comparable across a median-subtracted arm and one
without.

Why the dip is computed per site and then averaged: d is a LINEAR functional of
the profile, so d(mean over sites) == mean over sites of d(profile_i) exactly.
Reducing each site to one scalar up front makes the bootstrap ~1000x cheaper
and the paired bootstrap trivial, with no change to the answer.

COORDINATES. Everything here is 0-BASED, matching the arrays the HDF5 stores.
`metric.py` documented its signal as 1-based; the conversion happens once, in
`load_sites`, and nowhere else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

METRIC_VERSION = "1.0"

# Band definitions, in bp from motif centre. Frozen.
HALF = 2000            # half-window pulled around each site
CENTRE = 30            # centre band is +/- CENTRE
FLANK_LO = 1000        # flank band is FLANK_LO..FLANK_HI on both sides
FLANK_HI = 2000

MAX_NAN_FRAC = 0.05    # sites with more NaN than this are dropped

_CENTRE_BAND = slice(HALF - CENTRE, HALF + CENTRE + 1)
_FLANK_BAND = np.r_[np.arange(HALF - FLANK_HI, HALF - FLANK_LO),
                    np.arange(HALF + FLANK_LO, HALF + FLANK_HI)]


def load_sites(bed: Path, chrom: str) -> list[tuple[int, str]]:
    """TFBS BED -> [(0-based position, strand)] on `chrom`.

    BED is 0-based half-open and these files are one bp wide per site, so the
    site is at 0-based index `start` -- the same axis the HDF5 arrays use, with
    no offset. GTRD meta-cluster beds are unstranded and carry '.', which
    `per_site_dips` leaves unflipped.
    """
    sites = []
    for line in Path(bed).read_text().splitlines():
        if not line or line.startswith(("#", "track", "browser")):
            continue
        fields = line.split("\t")
        if fields[0] != chrom:
            continue
        strand = fields[5] if len(fields) > 5 else "."
        sites.append((int(fields[1]), strand))
    return sites


def chromosome_track(dataset, chrom: str) -> np.ndarray:
    """Tile a WindowDataset's windows back onto the chromosome axis.

    Returns float32 of the chromosome's length, NaN everywhere the dataset does
    not cover or marks invalid. NaN rather than zero because the metric's job is
    to DROP sites it cannot measure, and a zero would be measured as signal.

    This is what "through the dataloader output" means: the composite is built
    from exactly the tensors the model would be trained on, not from a separate
    read of the same file.
    """
    import h5py

    with h5py.File(dataset.path, "r") as h5:
        length = h5[f"shared/{chrom}/mask"].shape[0]
        index = h5["index"]
        rows = [(i, int(index["start"][int(r)]), int(index["end"][int(r)]))
                for i, r in enumerate(dataset.rows)
                if index["chrom"][int(r)].decode() == chrom]

    track = np.full(length, np.nan, dtype=np.float32)
    for i, start, end in rows:
        model_input, _, valid = dataset[i]
        values = model_input[0].numpy().copy()
        values[valid.numpy() == 0] = np.nan
        track[start:end] = values
    return track


def per_site_dips(signal: np.ndarray, sites, scalar: float):
    """(dips, profiles, kept_idx) over sites with enough non-NaN data.

    signal : 0-based position-indexed 1-D array; unusable positions are NaN.
    sites  : [(0-based pos, strand)], strand in {'+', '-', '.'}
    scalar : one global normalising number for the whole sample.

    `profiles` is kept for plotting only -- the metric never touches it.
    """
    dips, profiles, kept = [], [], []
    n = len(signal)
    for i, (pos, strand) in enumerate(sites):
        lo, hi = pos - HALF, pos + HALF + 1
        if lo < 0 or hi > n:
            continue
        window = signal[lo:hi]
        if np.isnan(window).mean() > MAX_NAN_FRAC:
            continue
        if strand == "-":
            window = window[::-1]
        window = window / scalar
        d = np.nanmean(window[_FLANK_BAND]) - np.nanmean(window[_CENTRE_BAND])
        if not np.isfinite(d):
            continue
        dips.append(d)
        profiles.append(window.astype(np.float32))
        kept.append(i)
    if not dips:
        return np.array([]), np.zeros((0, 2 * HALF + 1), np.float32), []
    return np.array(dips), np.vstack(profiles), kept


def dip_ci(dips: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Point estimate and 95% percentile bootstrap CI for the composite dip."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dips), size=(n_boot, len(dips)))
    boot = dips[idx].mean(axis=1)
    return (float(dips.mean()),
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def retention_ci(dips_variant: np.ndarray, dips_reference: np.ndarray,
                 n_boot: int = 1000, seed: int = 0):
    """PAIRED bootstrap for retention = d_variant / d_reference.

    Variants are computed on the SAME sites, so the same resampled indices must
    be used for both. Bootstrapping them independently would inflate the CI and
    could hide a real difference.
    """
    if len(dips_variant) != len(dips_reference):
        raise ValueError("variants must share sites")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dips_reference), size=(n_boot, len(dips_reference)))
    ratio = dips_variant[idx].mean(axis=1) / dips_reference[idx].mean(axis=1)
    return (float(dips_variant.mean() / dips_reference.mean()),
            float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5)))


def band_levels(profile: np.ndarray) -> tuple[float, float]:
    """(flank level, centre level) of a composite profile, in its own units."""
    return (float(np.nanmean(profile[_FLANK_BAND])),
            float(np.nanmean(profile[_CENTRE_BAND])))


def footprint_fwhm(profile: np.ndarray) -> float:
    """Full width at half maximum of the central dip, in bp.

    Not part of the metric: it is the covariate for asking whether damage from a
    fixed-kernel running median scales with footprint width. Measure it on the
    reference composite only.
    """
    p = np.asarray(profile, float)
    base = np.nanmean(p[_FLANK_BAND])
    depth = base - np.nanmin(p[_CENTRE_BAND])
    if depth <= 0:
        return np.nan
    half = base - depth / 2.0
    lo = hi = HALF
    while lo > 0 and p[lo] < half:
        lo -= 1
    while hi < len(p) - 1 and p[hi] < half:
        hi += 1
    return float(hi - lo)


def match(a: np.ndarray, b: np.ndarray, tol: int):
    """Fraction of reference peaks `a` with a peak in `b` within `tol` bp.

    Both must be SORTED ascending -- it is a searchsorted lookup.

    ONE-DIRECTIONAL, and that is a known blind spot: nothing here penalises
    extra peaks in `b`, so a prediction that puts peaks everywhere scores well.
    Never quote this without also quoting how many peaks `b` contains.
    """
    if len(a) == 0:
        return np.nan, np.array([], dtype=bool)
    if len(b) == 0:
        return 0.0, np.zeros(len(a), bool)
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    nearest = np.minimum(np.abs(b[idx] - a), np.abs(b[idx - 1] - a))
    hit = nearest <= tol
    return float(hit.mean()), hit
