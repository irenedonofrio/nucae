"""Peak-distance metrics, and the blind spot the reverse direction closes.

The forward measurement rewards calling more peaks: every extra peak can only
shorten some reference peak's nearest-neighbour distance and is never itself
charged for. That is not hypothetical -- on the fullgenome test split the raw
input calls a median of 100 peaks against 114 true while the reconstruction
calls 120, so the historical 22 bp vs 39 bp comparison is biased toward the
reconstruction by an amount the forward number cannot express.
"""

from __future__ import annotations

import numpy as np
import pytest

from nucae.metrics import nearest_peak_distances, peak_distances_both_ways

TRUE_PEAKS = [100, 300, 500]


def test_forward_direction_is_unchanged():
    """The historical measurement must keep returning exactly what it returned."""
    assert list(nearest_peak_distances([100, 300, 500], TRUE_PEAKS)) == [0, 0, 0]
    assert list(nearest_peak_distances([110, 310, 510], TRUE_PEAKS)) == [10, 10, 10]


def test_a_peak_everywhere_scores_a_perfect_forward_distance():
    """The blind spot, stated as a test: dense noise beats an honest prediction."""
    dense = list(range(0, 600, 5))          # a peak every 5 bp, mostly spurious
    honest = [102, 298, 505]                # three real peaks, each off by a few bp

    assert np.median(nearest_peak_distances(dense, TRUE_PEAKS)) == 0
    assert np.median(nearest_peak_distances(honest, TRUE_PEAKS)) > 0


def test_reverse_direction_charges_for_invented_peaks():
    """Same two signals, measured the other way round: the ranking inverts."""
    dense = list(range(0, 600, 5))
    honest = [102, 298, 505]

    _, dense_precision = peak_distances_both_ways(dense, TRUE_PEAKS)
    _, honest_precision = peak_distances_both_ways(honest, TRUE_PEAKS)

    assert np.median(dense_precision) > np.median(honest_precision)


def test_recall_arm_is_the_forward_function():
    """One definition of the arithmetic: the pair's first element IS the old call."""
    query = [110, 290, 495]
    recall, _ = peak_distances_both_ways(query, TRUE_PEAKS)
    assert list(recall) == list(nearest_peak_distances(query, TRUE_PEAKS))


def test_lengths_follow_the_direction_measured():
    """One distance per reference peak forward, one per query peak reverse."""
    query = [100, 200, 300, 400]
    recall, precision = peak_distances_both_ways(query, TRUE_PEAKS)
    assert len(recall) == len(TRUE_PEAKS)
    assert len(precision) == len(query)


@pytest.mark.parametrize("query, ref", [([], TRUE_PEAKS), (TRUE_PEAKS, []), ([], [])])
def test_no_peaks_is_not_measurable_rather_than_zero(query, ref):
    """None, never 0.0 -- a window with nothing to compare is not a perfect score."""
    assert nearest_peak_distances(query, ref) is None
    assert peak_distances_both_ways(query, ref) is None
