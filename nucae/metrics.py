"""Reconstruction-quality metrics: how well a prediction matches its target.

SEPARATE FROM evaluate.py ON PURPOSE. That module measures the TFBS composite
dip and its docstring states that nothing in it is scale-free, because a dip
depth is an amplitude claim. Everything here is the opposite: Pearson r and
nearest-peak distance are SHAPE metrics, blind to amplitude by construction. A
model can score perfectly on both while predicting the right waveform at the
wrong scale. Keeping them in different modules keeps that distinction visible.

One implementation each. `per_window_pearson` is used by the training module's
validation logging and by scripts/04_evaluate.py, so the number reported at the
end of training and the number in the evaluation summary cannot disagree.
"""

from __future__ import annotations

import numpy as np
import torch

# Floor on the L2 norm of a centred window, applied INSIDE each sqrt rather than
# as a clamp on the product of norms. At zero variance the derivative is then
# 0/sqrt(eps) -- finite -- instead of the 0/0 a clamp leaves behind. Measured on
# a flat prediction: gradient norm 1.0e4 here against 8.8e8 for the clamp form
# used by old_nucae/model_v1._pearson_r. A real 25 kb window has a centred norm
# of order 79, some six orders of magnitude above sqrt(eps) = 1e-4.
EPS = 1e-8

# Below this product of centred norms, `pearson` reports NaN rather than a
# number. Matches the guard in the cluster's evaluate.py, so summary means stay
# comparable to the results already on disk.
MIN_NORM_PRODUCT = 1e-8


def per_window_pearson(pred: torch.Tensor, target: torch.Tensor,
                       eps: float = EPS) -> torch.Tensor:
    """Pearson r of each window independently. (B, 1, L) -> (B,).

    Reduced over the length axis only, never pooled across the batch: a pooled
    coefficient is dominated by whichever windows carry the most amplitude, so a
    model can score well by fitting loud windows and ignoring quiet ones.

    Returns ~0 for a window where either side is flat, and the caller sees it as
    a poor score rather than a NaN.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    if pred.ndim != 3 or pred.shape[1] != 1:
        # A (B, L) tensor would broadcast against (B, 1, L) into (B, B, L) and
        # return a plausible number computed over the wrong axes.
        raise ValueError(f"expected (B, 1, L), got {tuple(pred.shape)}")

    p = pred.squeeze(1)
    t = target.squeeze(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)

    numerator = (p * t).sum(dim=1)
    denominator = (torch.sqrt((p * p).sum(dim=1) + eps)
                   * torch.sqrt((t * t).sum(dim=1) + eps))
    return numerator / denominator


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1D arrays, for reporting. NaN if either is flat.

    Delegates to per_window_pearson so there is one definition of the arithmetic,
    then applies a DIFFERENT flat-window convention on top, deliberately:

      - training (per_window_pearson) needs a finite number with a finite
        gradient, so a flat window scores ~0 and takes the full penalty.
      - a reported summary needs the window EXCLUDED, because r is undefined
        where there is no variance. Averaging a 0 in would silently drag the
        mean down in proportion to how many blacklisted or all-zero windows the
        split happens to contain.

    NaN is what the cluster's evaluate.py returned, so mean r over a split stays
    comparable to the numbers already recorded.

    Computed in float64; the original ran in float32. The difference is far
    below the precision any summary is quoted at.
    """
    centred = [np.asarray(x, dtype=np.float64) - np.mean(x) for x in (a, b)]
    if np.linalg.norm(centred[0]) * np.linalg.norm(centred[1]) < MIN_NORM_PRODUCT:
        return float("nan")
    pair = [torch.from_numpy(x)[None, None, :] for x in centred]
    return float(per_window_pearson(*pair)[0])


def nearest_peak_distances(query_peaks, ref_peaks) -> np.ndarray | None:
    """For each peak in `ref_peaks`, the distance to the nearest peak in `query_peaks`.

    ONE-DIRECTIONAL, AND THAT IS A BLIND SPOT. Nothing here penalises peaks the
    query has and the reference does not, so a prediction with a peak everywhere
    scores a distance of zero. Never quote this without the peak counts beside
    it; scripts/04_evaluate.py reports n_peaks for all three signals.

    Returns None when either side has no peaks, which the caller must treat as
    "not measurable here" rather than as a distance of zero.
    """
    if len(query_peaks) == 0 or len(ref_peaks) == 0:
        return None
    query = np.asarray(query_peaks)
    return np.array([np.min(np.abs(query - p)) for p in np.asarray(ref_peaks)])
