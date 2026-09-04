"""The halo assertion: the claim the storage design rests on.

A 25 kb window read with a 400 bp halo, smoothed, then cropped is compared
against the same slice of a whole-chromosome smooth. Three parts, because the
chain does not have one uniform answer:

1. WHITTAKER OFF -- bit-identical, and this is the real test of the halo.
   The Gaussian reaches 4*sigma = 120 bp and medfilt(375) reaches 187 bp, both
   inside 400, so a haloed window must reproduce the whole chromosome exactly.

2. FULL CHAIN -- max|delta| below one float32 ULP at the signal's scale.
   Whittaker-Eilers is a GLOBAL banded solve with no compact support, so its
   influence never reaches zero and bit-identity is unattainable at ANY halo.
   Measured: still 4 differing positions at a halo of 20,000. The criterion is
   not that the residual is small, it is that it is invisible at the precision
   the counts are stored in.

3. PEAK RECOVERY -- measured and reported, NOT gated. It is the number
   CLAUDE.md quotes for the halo, so it is recorded here; making it a pass/fail
   threshold would mean choosing the threshold after seeing the value.

Slow by construction: it smooths a whole chromosome twice so there is a real
answer to compare against.

Point NUCAE_TEST_H5 at an HDF5 built by scripts/01_build_hdf5.py. The test skips
if unset -- building one needs the reference genome and a counts file, neither
of which lives in this repository.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from nucae import preprocess
from nucae.MaxFinder import get_peaks
from nucae.data import HALO, WINDOW, WindowDataset

H5_PATH = os.environ.get("NUCAE_TEST_H5", "")
N_WINDOWS = 50
N_PEAK_WINDOWS = 200

# One float32 ULP at the smoothed signal's scale: max|value| ~ 0.32 lies in
# [0.25, 0.5), where the spacing between representable floats is 2^-25 = 3.0e-8.
# DERIVED, not fitted to the observed residual -- the criterion is that the
# windowing residual cannot be seen at the precision the data is stored in.
FLOAT32_ULP_AT_SIGNAL_SCALE = 2e-8

# MaxFinder's __main__ hardcodes min_len=10 while the signature defaults to 30,
# so it is always passed explicitly.
PEAK_PARAMS = dict(window=375, max_gap=35, min_len=10)

pytestmark = pytest.mark.skipif(
    not H5_PATH or not Path(H5_PATH).is_file(),
    reason="set NUCAE_TEST_H5 to an HDF5 built by scripts/01_build_hdf5.py",
)


@contextlib.contextmanager
def whittaker_disabled():
    """Replace only the Whittaker solve with the identity.

    Everything else in `preprocess.smooth` -- the blacklist zeroing, the
    Gaussian, the median subtraction, the float32 cast -- runs for real. This
    isolates the one step with global support without adding a mode switch to
    production code or restating the chain inside a test.
    """
    class _IdentitySmoother:
        def __init__(self, **kwargs):
            pass

        def smooth(self, x):
            return x

    original = preprocess.WhittakerSmoother
    preprocess.WhittakerSmoother = _IdentitySmoother
    try:
        yield
    finally:
        preprocess.WhittakerSmoother = original


@pytest.fixture(scope="module")
def built() -> dict:
    dataset = WindowDataset(H5_PATH, input_level="full", target_level="full")
    with h5py.File(H5_PATH, "r") as h5:
        index = h5["index"]
        chrom = index["chrom"][int(dataset.rows[0])].decode()
        counts = h5[f"levels/full/{chrom}/counts_gc"][:]
        mask = h5[f"shared/{chrom}/mask"][:]
        bounds = [(i, int(index["start"][int(r)]), int(index["end"][int(r)]))
                  for i, r in enumerate(dataset.rows)]
    chrom_len = len(mask)
    return dict(
        dataset=dataset, chrom=chrom, chrom_len=chrom_len, counts=counts, mask=mask,
        whole=preprocess.smooth(counts, mask),
        interior=[(i, s, e) for i, s, e in bounds
                  if s - HALO >= 0 and e + HALO <= chrom_len],
    )


def _compare(dataset, rows, whole):
    """(windows differing, positions differing, max abs delta) over `rows`."""
    n_windows = n_positions = 0
    worst = 0.0
    for i, start, end in rows:
        windowed = dataset[i][0][0].numpy()
        expected = whole[start:end]
        n_diff = int(np.count_nonzero(windowed.view(np.int32) != expected.view(np.int32)))
        if n_diff:
            n_windows += 1
            n_positions += n_diff
            worst = max(worst, float(np.abs(windowed.astype(np.float64)
                                            - expected.astype(np.float64)).max()))
    return n_windows, n_positions, worst


# ── 1. the halo itself ────────────────────────────────────────────────────────


def test_enough_interior_windows_to_test(built):
    assert len(built["interior"]) >= N_WINDOWS


def test_haloed_window_is_bit_identical_without_whittaker(built):
    """THE GATE. Compact-support filters must be reproduced exactly."""
    rows = built["interior"][:N_WINDOWS]
    assert len(rows) == N_WINDOWS
    with whittaker_disabled():
        whole = preprocess.smooth(built["counts"], built["mask"])
        n_windows, n_positions, worst = _compare(built["dataset"], rows, whole)
    assert n_windows == 0, (
        f"{n_windows} of {N_WINDOWS} windows differ over {n_positions:,} positions "
        f"(max|delta| {worst:.3e}). With Whittaker off the chain has compact support "
        f"120 and 187 bp, both inside the {HALO} bp halo, so this must be exact."
    )


# ── 2. the full chain ─────────────────────────────────────────────────────────


def test_full_chain_residual_is_below_float32_storage_precision(built):
    """Whittaker is global, so bits differ. They must not differ visibly."""
    rows = built["interior"][:N_WINDOWS]
    n_windows, n_positions, worst = _compare(built["dataset"], rows, built["whole"])
    print(f"\n  full chain: {n_windows}/{N_WINDOWS} windows, {n_positions:,} positions, "
          f"max|delta| {worst:.3e}  (tolerance {FLOAT32_ULP_AT_SIGNAL_SCALE:.1e})")
    assert worst < FLOAT32_ULP_AT_SIGNAL_SCALE, (
        f"max|delta| {worst:.3e} reaches float32 storage precision "
        f"({FLOAT32_ULP_AT_SIGNAL_SCALE:.1e}); the windowing residual is no longer "
        f"invisible and the halo must be reconsidered."
    )


# ── 3. peak recovery: measured, not gated ─────────────────────────────────────


def test_peak_recovery_is_measured_and_reported(built):
    """Reported for the record. No threshold -- see the module docstring."""
    dataset, counts, mask, whole = (built["dataset"], built["counts"],
                                    built["mask"], built["whole"])

    def peaks(arr):
        found, _, _ = get_peaks(arr, **PEAK_PARAMS)
        return set(np.asarray(found, dtype=int).tolist())

    recoveries, identical = [], 0
    for i, start, end in built["interior"][:N_PEAK_WINDOWS]:
        reference = peaks(whole[start:end])
        if not reference:
            continue
        got = peaks(dataset[i][0][0].numpy())
        recoveries.append(len(reference & got) / len(reference))
        identical += reference == got

    r = np.array(recoveries)
    print(f"\n  peak recovery over {len(r)} windows with peaks: "
          f"median {np.median(r):.4f}  min {r.min():.4f}  mean {r.mean():.4f}")
    print(f"  identical peak sets: {identical}/{len(r)} ({identical / len(r):.2%})")
    assert len(r) > 0, "no interior window produced peaks; nothing was measured"


# ── the returned tensors ──────────────────────────────────────────────────────


def test_target_branch_takes_the_same_path(built):
    i, start, end = built["interior"][0]
    target = built["dataset"][i][1][0].numpy()
    delta = np.abs(target.astype(np.float64)
                   - built["whole"][start:end].astype(np.float64)).max()
    assert delta < FLOAT32_ULP_AT_SIGNAL_SCALE


def test_shapes_and_dtypes(built):
    model_input, target, valid = built["dataset"][0]
    assert model_input.shape == (6, WINDOW) and model_input.dtype is torch.float32
    assert target.shape == (1, WINDOW) and target.dtype is torch.float32
    assert valid.shape == (WINDOW,) and valid.dtype is torch.float32


def test_sequence_is_one_hot_exactly_once_per_position(built):
    model_input = built["dataset"][0][0]
    assert torch.equal(model_input[1:].sum(dim=0), torch.ones(WINDOW))


def test_no_mask_channel(built):
    """The mask is a loss weight. A 7th channel is a deferred experiment."""
    assert built["dataset"][0][0].shape[0] == 6


def test_valid_is_a_subset_of_the_unmasked_positions(built):
    """Eroding can only remove positions, never add them."""
    i, start, end = built["interior"][0]
    valid = built["dataset"][i][2].numpy() > 0
    with h5py.File(H5_PATH, "r") as h5:
        unmasked = ~h5[f"shared/{built['chrom']}/mask"][start:end]
    assert np.all(unmasked[valid]), "a position marked valid is masked on disk"


def test_erosion_removes_positions_next_to_a_gap(built):
    """A window straddling a masked region must lose its border to erosion."""
    with h5py.File(H5_PATH, "r") as h5:
        stored_mask = h5[f"shared/{built['chrom']}/mask"]
        for i, start, end in built["interior"]:
            unmasked = ~stored_mask[start:end]
            if unmasked.all() or not unmasked.any():
                continue
            valid = built["dataset"][i][2].numpy() > 0
            assert valid.sum() < unmasked.sum(), "erosion removed nothing at a gap"
            return
    pytest.skip("no partially-masked window among the interior rows")
