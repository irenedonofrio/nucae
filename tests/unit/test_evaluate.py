"""The metric's own properties, on synthetic signal with a planted answer.

Every claim in the audit this metric came from is a RATIO, so what must hold is
that the ratio between two variants is unbiased. The absolute recovered depth is
NOT expected to equal the planted depth: the metric is a band difference, not a
pointwise minimum, so a narrow footprint averages to less than its peak over a
+/- 30 bp centre band.

Synthetic only, deliberately: this validates the measuring harness, not any
scientific claim about real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from nucae import evaluate


def planted_chromosome(depth=0.30, fwhm=40, n_sites=400, length=4_000_000,
                       baseline=1.0, period=190, osc=0.15, noise=0.05, seed=1):
    """Position-indexed synthetic chromosome with a composite dip of known depth."""
    rng = np.random.default_rng(seed)
    x = np.arange(length, dtype=np.float64)
    signal = baseline + osc * np.cos(2 * np.pi * x / period)
    signal += 0.20 * np.sin(2 * np.pi * x / 300_000)        # CNV-scale wobble
    sites = []
    step = length // (n_sites + 2)
    sd = fwhm / 2.355
    for k in range(1, n_sites + 1):
        pos = step * k
        w = np.arange(-3 * fwhm, 3 * fwhm + 1)
        signal[pos - 3 * fwhm: pos + 3 * fwhm + 1] -= depth * np.exp(-w ** 2 / (2 * sd ** 2))
        sites.append((pos, "+" if k % 2 else "-"))
    signal += rng.normal(0, noise, length)
    return signal.astype(np.float32), sites


@pytest.mark.parametrize("fwhm", [40, 150, 350])
def test_halving_the_planted_depth_halves_the_measured_dip(fwhm):
    """The ratio must be unbiased even where the absolute recovery is not."""
    deep, sites = planted_chromosome(depth=0.30, fwhm=fwhm)
    shallow, _ = planted_chromosome(depth=0.15, fwhm=fwhm)
    d_deep = evaluate.per_site_dips(deep, sites, 1.0)[0].mean()
    d_shallow = evaluate.per_site_dips(shallow, sites, 1.0)[0].mean()
    assert abs(d_shallow / d_deep - 0.5) < 0.02


def test_a_planted_dip_is_detected_and_a_flat_signal_is_not():
    signal, sites = planted_chromosome(depth=0.30, fwhm=150)
    flat, _ = planted_chromosome(depth=0.0, fwhm=150)
    _, lo, _ = evaluate.dip_ci(evaluate.per_site_dips(signal, sites, 1.0)[0])
    flat_point, flat_lo, flat_hi = evaluate.dip_ci(
        evaluate.per_site_dips(flat, sites, 1.0)[0])
    assert lo > 0, "a planted dip must be detected"
    assert flat_lo <= 0 <= flat_hi, f"flat signal produced a dip: {flat_point:.5f}"


def test_the_scalar_divides_the_dip():
    """d is linear in 1/S. Halving S must double the dip."""
    signal, sites = planted_chromosome(depth=0.30, fwhm=150)
    one = evaluate.per_site_dips(signal, sites, 1.0)[0].mean()
    half = evaluate.per_site_dips(signal, sites, 0.5)[0].mean()
    assert np.isclose(half, 2 * one, rtol=1e-6)


def test_sites_with_too_much_nan_are_dropped():
    signal, sites = planted_chromosome(depth=0.30, fwhm=150, n_sites=20)
    holed = signal.copy()
    pos = sites[0][0]
    holed[pos - 500:pos + 500] = np.nan          # 1000 bp of 4001 = 25% > MAX_NAN_FRAC
    _, _, kept = evaluate.per_site_dips(holed, sites, 1.0)
    assert 0 not in kept


def test_bands_are_frozen():
    """Changing a band changes every number this metric ever produced."""
    assert (evaluate.CENTRE, evaluate.FLANK_LO, evaluate.FLANK_HI,
            evaluate.HALF) == (30, 1000, 2000, 2000)
    assert evaluate.METRIC_VERSION == "1.0"


# ── match: the peak-distance metric ───────────────────────────────────────────


def test_match_is_one_for_identical_peaks():
    peaks = np.array([100, 500, 900])
    assert evaluate.match(peaks, peaks, tol=0)[0] == 1.0


def test_match_respects_the_tolerance():
    reference = np.array([100, 500])
    shifted = np.array([105, 505])
    assert evaluate.match(reference, shifted, tol=5)[0] == 1.0
    assert evaluate.match(reference, shifted, tol=4)[0] == 0.0


def test_match_with_no_predicted_peaks_is_zero_not_nan():
    assert evaluate.match(np.array([100]), np.array([]), tol=10)[0] == 0.0


def test_match_with_no_reference_peaks_is_nan():
    assert np.isnan(evaluate.match(np.array([]), np.array([100]), tol=10)[0])


def test_match_is_one_directional_and_extra_peaks_are_free():
    """The documented blind spot, pinned so nobody is surprised by it."""
    reference = np.array([500])
    everywhere = np.arange(0, 1000, 10)
    assert evaluate.match(reference, everywhere, tol=5)[0] == 1.0
