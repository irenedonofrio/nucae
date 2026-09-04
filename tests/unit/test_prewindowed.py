"""The pre-windowed reader returns what the model expects, from a synthetic file.

Fixtures only; no real data ever enters a test.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from nucae.prewindowed import CfDNAWindows, CfDNAWindowsDataModule

LENGTH = 256
N = 4
SEED = 0


@pytest.fixture
def h5_path(tmp_path):
    """A file with the layout convert_to_hdf5.py writes, filled with known values."""
    rng = np.random.default_rng(SEED)
    path = tmp_path / "cfdna_fixture.h5"
    with h5py.File(path, "w") as h5:
        for split in ("train", "val", "test"):
            group = h5.create_group(f"windows/{split}")
            group["full"] = rng.normal(size=(N, LENGTH)).astype(np.float32)
            group["under"] = rng.normal(size=(N, LENGTH)).astype(np.float32)
            # 0-3 only, as the real files carry: the N code is never used.
            group["sequence"] = rng.integers(0, 4, size=(N, LENGTH), dtype=np.uint8)
            group["chrom"] = np.array([b"chr8"] * N, dtype="|S10")
            group["start"] = np.arange(N, dtype=np.int32) * LENGTH
            group["end"] = np.arange(1, N + 1, dtype=np.int32) * LENGTH
        h5.create_group("metadata").attrs["signal_length"] = LENGTH
    return path


def test_returns_a_pair_with_the_documented_shapes_and_dtypes(h5_path):
    x, y = CfDNAWindows(h5_path, "train")[0]
    assert x.shape == (6, LENGTH) and x.dtype == torch.float32
    assert y.shape == (1, LENGTH) and y.dtype == torch.float32


def test_without_sequence_the_input_is_coverage_alone(h5_path):
    x, y = CfDNAWindows(h5_path, "train", use_sequence=False)[0]
    assert x.shape == (1, LENGTH)
    assert y.shape == (1, LENGTH)


def test_channel_zero_is_the_undersampled_signal_and_the_target_is_the_full_one(h5_path):
    dataset = CfDNAWindows(h5_path, "train")
    x, y = dataset[2]
    with h5py.File(h5_path, "r") as h5:
        assert np.array_equal(x[0].numpy(), h5["windows/train/under"][2])
        assert np.array_equal(y[0].numpy(), h5["windows/train/full"][2])


def test_one_hot_has_exactly_one_base_set_per_position(h5_path):
    x, _ = CfDNAWindows(h5_path, "train")[1]
    one_hot = x[1:]
    assert torch.equal(one_hot.sum(dim=0), torch.ones(LENGTH))
    with h5py.File(h5_path, "r") as h5:
        expected = h5["windows/train/sequence"][1]
    assert np.array_equal(one_hot.argmax(dim=0).numpy(), expected)


def test_metadata_matches_what_was_written(h5_path):
    meta = CfDNAWindows(h5_path, "test").metadata(3)
    assert meta == {"chrom": "chr8", "start": 3 * LENGTH, "end": 4 * LENGTH,
                    "split": "test"}


def test_handle_is_reopened_in_a_new_process(h5_path):
    """The fork fix. A worker inheriting the parent's HDF5 handle can read
    silent garbage, so the handle is keyed on the pid that opened it."""
    dataset = CfDNAWindows(h5_path, "train")
    first = dataset.h5
    assert dataset.h5 is first                  # same process, same handle

    dataset._pid = -1                           # stand in for "opened before a fork"
    assert dataset.h5 is not first


def test_unknown_split_is_refused(h5_path):
    with pytest.raises(ValueError):
        CfDNAWindows(h5_path, "trian")


def test_datamodule_builds_only_the_splits_a_stage_needs(h5_path):
    """setup() must not touch a window: fetching one opens the handle in the
    parent, which is exactly what the workers must not inherit."""
    datamodule = CfDNAWindowsDataModule(h5_path, batch_size=2, num_workers=0)
    datamodule.setup("fit")
    assert set(datamodule.datasets) == {"train", "val"}
    assert all(d._h5 is None for d in datamodule.datasets.values())

    x, y = next(iter(datamodule.train_dataloader()))
    assert x.shape == (2, 6, LENGTH) and y.shape == (2, 1, LENGTH)
