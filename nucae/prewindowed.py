"""Reader for the pre-windowed HDF5 that produced the existing checkpoints.

    /windows/{train,val,test}/{full, under, sequence, chrom, start, end}
    /metadata/  signal_length, n_train, n_val, n_test, normalised, ...

INHERITED INPUT, NOT PRODUCED HERE. These files were built by
old_nucae/convert_to_hdf5.py from text exports of an earlier pipeline. The
signals arrive already smoothed and already scaled: `metadata/normalised` is
written as True unconditionally by that script, so it records a belief, not a
measurement. What the upstream normalisation actually was is not recoverable
from any code in this repository -- run old_nucae/inspect_h5.py, whose
std(full)/std(under) line settles it. Do not describe these windows as
normalised in any particular way until that has been run.

TWO KNOWN DEFECTS IN THE FILES THEMSELVES, both from convert_to_hdf5.py and
both silent. A line that failed to parse left an all-zero window in place rather
than dropping it, and any such failure also shifted every subsequent chromosome
LABEL by one relative to its own start/end. inspect_h5.py detects both. Neither
is repaired here: this module reads what is on disk.

NO MASK. Every one of the 25,000 positions is returned as real signal, which is
what the cluster trained on. nucae.data.WindowDataset, over the counts-derived
files, additionally returns a validity mask; this format carries no equivalent.
See docs/OPEN_ISSUES.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

N_NT = 5                 # one-hot A, C, G, T, N; this format uses 0-3 only
SPLITS = ("train", "val", "test")


class CfDNAWindows(Dataset):
    """One window -> (x, y).

        x : (6, L) float32   ch 0 undersampled coverage, ch 1-5 one-hot sequence
                             (1, L) when use_sequence is False
        y : (1, L) float32   fullysampled coverage, the reconstruction target
    """

    def __init__(self, path: Path, split: str = "train",
                 use_sequence: bool = True) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.path = Path(path)
        self.split = split
        self.use_sequence = use_sequence
        self._h5: h5py.File | None = None
        self._pid: int | None = None

        with h5py.File(self.path, "r") as h5:
            full = h5[f"windows/{split}/full"]
            self.n, self.signal_length = int(full.shape[0]), int(full.shape[1])

    @property
    def h5(self) -> h5py.File:
        # An HDF5 handle is not safe across a fork: a worker that inherits one
        # opened in the parent can read silent garbage. Reopen per process.
        # Same reason and same fix as nucae.data.WindowDataset.
        if self._h5 is None or self._pid != os.getpid():
            self._h5 = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._h5

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        group = self.h5[f"windows/{self.split}"]
        under = torch.from_numpy(group["under"][i].astype("float32"))
        full = torch.from_numpy(group["full"][i].astype("float32"))

        if not self.use_sequence:
            return under[None, :], full[None, :]

        sequence = torch.from_numpy(group["sequence"][i].astype("int64"))
        one_hot = F.one_hot(sequence, num_classes=N_NT).T.float()   # (L,) -> (5, L)
        return torch.cat([under[None, :], one_hot], dim=0), full[None, :]

    def metadata(self, i: int) -> dict[str, object]:
        """Coordinates for window i, for writing predictions back to bedGraph.

        `chrom` may be misaligned with start/end -- see the module docstring.
        """
        group = self.h5[f"windows/{self.split}"]
        return {"chrom": group["chrom"][i].decode("ascii"),
                "start": int(group["start"][i]),
                "end": int(group["end"][i]),
                "split": self.split}


class CfDNAWindowsDataModule(pl.LightningDataModule):
    """Train/val/test loaders over one pre-windowed HDF5."""

    def __init__(self, path: Path, batch_size: int = 16, num_workers: int = 4,
                 pin_memory: bool = True, use_sequence: bool = True) -> None:
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_sequence = use_sequence
        self.datasets: dict[str, CfDNAWindows] = {}

    def setup(self, stage: str | None = None) -> None:
        # Nothing is read from the file here beyond its shapes, and no window is
        # fetched: touching a window in the parent process opens the HDF5 handle
        # that the workers would then inherit across the fork.
        wanted = SPLITS if stage is None else {"fit": ("train", "val"),
                                               "validate": ("val",),
                                               "test": ("test",)}.get(stage, SPLITS)
        for split in wanted:
            self.datasets[split] = CfDNAWindows(self.path, split, self.use_sequence)

    def _loader(self, split: str, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(self.datasets[split], batch_size=self.batch_size,
                          shuffle=shuffle, drop_last=drop_last,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False, drop_last=False)
