"""HDF5 builder: raw counts, mask, and sequence on the reference axis.

One file per sample, coverage levels inside it. The sample IS the file, so
sample_id is not a group in the path and the per-sample attrs sit on the root.
A level name is a key -- it identifies a group -- never also stored as a value.

`shared/` holds mask and sequence because they depend on the sample and the
reference genome, not on sequencing depth. Storing them once rather than once
per level.

What goes on disk is raw. Smoothing and normalisation happen in the dataloader,
because smoothing is destructive, some of its parameters are still open, and one
file has to serve consumers that want different things from it.

Everything is written on the reference chromosome's own axis, padded to its .fai
length, so every coverage level of every sample aligns by construction and a
genomic position means the same thing in all of them.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pysam
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from . import preprocess

HALO = preprocess.HALO
N_INPUT_CHANNELS = 6          # ch 0 coverage, ch 1-5 one-hot A,C,G,T,N

SCHEMA_VERSION = "1"
WINDOW = 25_000
AUTOSOMES = tuple(f"chr{i}" for i in range(1, 23))

# Chromosome -> split. Anything unlisted is train. `split` is a column in
# /index, so moving a chromosome is a column write, not a rebuild.
SPLITS = {"chr7": "val", "chr8": "test", "chr9": "test"}

# The Gaussian's finite support, derived from the one place SIGMA is defined.
# A position whose kernel touched a masked base carries invented signal, so
# window validity is measured AFTER dilating the mask by this much -- the same
# radius the dataloader's validity channel uses, so the two cannot drift.
DILATION_RADIUS = 4 * preprocess.SIGMA        # 120 bp at sigma=30

KEEP_CRITERION = "valid_frac > 0"

BASES = "ACGT"
N_CODE = 4

# Fixed-width bytes rather than h5py variable-length strings: /index is read one
# row at a time by the dataloader, and vlen strings make that a heap lookup.
INDEX_DTYPES = {
    "sample_id": "S32", "level": "S16", "chrom": "S8", "split": "S8",
    "start": "i8", "end": "i8", "keep": "?",
    "valid_frac": "f4", "target_std": "f4",
}

# A level's attrs must carry enough to interpret its counts. Required so that an
# unknown value is written as an explicit blank by whoever builds the file,
# rather than silently missing.
LEVEL_ATTRS = ("nominal_depth", "measured_depth", "gc_table_md5",
               "bam_path", "n_fragments")
SAMPLE_ATTRS = ("subtype", "tumour_fraction", "sex", "cohort")

# Case folding lives in the lookup table rather than in seq.upper(), which would
# copy a 249 MB string on chr1. hg38 is soft-masked, so lowercase acgt are real
# bases; letting them fall through to N would zero out every repeat-masked
# region. Everything that is not ACGT -- N and every IUPAC ambiguity code -- is 4.
_BASE_CODES = np.full(256, N_CODE, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _BASE_CODES[ord(_b)] = _i
    _BASE_CODES[ord(_b.lower())] = _i


def chrom_length(fai: Path, chrom: str) -> int:
    """Sequence length of `chrom`, from a samtools .fai index."""
    for line in Path(fai).read_text().splitlines():
        name, length = line.split("\t")[:2]
        if name == chrom:
            return int(length)
    raise KeyError(f"{Path(fai).name}: no entry for {chrom!r}")


def load_sequence(fasta: Path, chrom: str, n: int) -> np.ndarray:
    """Reference bases as uint8: A=0 C=1 G=2 T=3, everything else 4.

    The FASTA is the authority on length, so disagreeing with `n` is an error
    rather than something to pad around.
    """
    with pysam.FastaFile(str(fasta)) as fh:
        if chrom not in fh.references:
            raise KeyError(f"{Path(fasta).name}: no sequence named {chrom!r}")
        seq = fh.fetch(chrom)
    if len(seq) != n:
        raise ValueError(f"{Path(fasta).name}: {chrom} is {len(seq):,} bp but n is "
                         f"{n:,}; the FASTA and the .fai disagree")
    return _BASE_CODES[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def build_mask(mappable_bed: Path, chrom: str, n: int) -> np.ndarray:
    """True = invalid. The complement of the mappable regions, and nothing else.

    Griffin's k100_minus_exclusion_lists BED already has the ENCODE exclusion
    list subtracted, so its complement subsumes the blacklist; a second BED
    would mask nothing new. Measured on chr21: of 2,451,213 ENCODE blacklist
    positions, 0 fall inside the mappable regions.

    The mask never comes from the data. A position that is zero at 1x and
    non-zero at 45x is empty, not missing, and a counts file that stops short of
    the chromosome end is not evidence of anything -- on chr21 those trailing
    10,077 bp are already unmappable.

    One BED reader for both meanings: `preprocess.load_blacklist_mask` returns
    True INSIDE an interval, which is what a blacklist means and the opposite of
    what a keep-list means, so the complement is taken here and nowhere else.
    """
    return ~preprocess.load_blacklist_mask(mappable_bed, chrom, n)


def dilated_valid(mask: np.ndarray, radius: int = DILATION_RADIUS) -> np.ndarray:
    """~mask, grown by `radius`. True = a position a loss may use."""
    if radius == 0:
        return ~mask
    return ~binary_dilation(mask, np.ones(2 * radius + 1, dtype=bool))


def window_bounds(chrom_len: int, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Starts and ends of every COMPLETE window, tiled from 0.

    A trailing partial window is not emitted: the model takes a fixed shape, so
    a short window is not a window.
    """
    n = chrom_len // window
    starts = np.arange(n, dtype=np.int64) * window
    return starts, starts + window


def index_rows(sample_id: str, level: str, chrom: str, chrom_len: int,
               mask: np.ndarray, window: int = WINDOW) -> dict[str, np.ndarray]:
    """One row per complete window. `keep` is a flag; nothing is ever dropped.

    `valid_frac` is measured on the DILATED mask, so it counts exactly the
    positions the dataloader will mark trainable. `target_std` is a diagnostic
    of the smoothed target and would cost a smoothing pass over the chromosome,
    which this stage does not do; it is written as NaN for whoever needs it.
    """
    starts, ends = window_bounds(chrom_len, window)
    n = starts.size
    valid = dilated_valid(mask)[:n * window].reshape(n, window).mean(axis=1)
    return {
        "sample_id": np.full(n, sample_id.encode(), dtype=INDEX_DTYPES["sample_id"]),
        "level": np.full(n, level.encode(), dtype=INDEX_DTYPES["level"]),
        "chrom": np.full(n, chrom.encode(), dtype=INDEX_DTYPES["chrom"]),
        "split": np.full(n, SPLITS.get(chrom, "train").encode(),
                         dtype=INDEX_DTYPES["split"]),
        "start": starts, "end": ends,
        "keep": valid > 0,
        "valid_frac": valid.astype("f4"),
        "target_std": np.full(n, np.nan, dtype="f4"),
    }


def open_h5(path: Path, reference_genome: str, sample_attrs: dict,
            git_sha: str = "", window: int = WINDOW) -> h5py.File:
    """Open one sample's file, creating the root attrs and /index the first time.

    The sample attrs describe the whole file, so they are written once here
    rather than re-stamped by every chromosome.
    """
    missing = [k for k in SAMPLE_ATTRS if k not in sample_attrs]
    if missing:
        raise ValueError(f"sample attrs missing {missing}")
    h5 = h5py.File(Path(path), "a")
    if "index" not in h5:
        h5.attrs.update(sample_attrs)
        h5.attrs.update({
            "schema_version": SCHEMA_VERSION,
            "build_date": str(np.datetime64("now")),
            "git_sha": git_sha,
            "reference_genome": reference_genome,
            "window": int(window),
            "keep_criterion": KEEP_CRITERION,
            "dilation_radius": int(DILATION_RADIUS),
        })
        index = h5.create_group("index")
        for name, dtype in INDEX_DTYPES.items():
            index.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                 chunks=(4096,))
    return h5


def append_index(h5: h5py.File, rows: dict[str, np.ndarray]) -> int:
    """Append index rows, returning the new total."""
    if set(rows) != set(INDEX_DTYPES):
        raise ValueError(f"index columns {sorted(rows)} != {sorted(INDEX_DTYPES)}")
    index = h5["index"]
    n_new = len(rows["start"])
    n_old = index["start"].shape[0]
    for name in INDEX_DTYPES:
        dataset = index[name]
        dataset.resize((n_old + n_new,))
        dataset[n_old:] = rows[name]
    return n_old + n_new


def build_chromosome(h5: h5py.File, sample_id: str, level: str, chrom: str,
                     counts: Path, fasta: Path, mappable_bed: Path,
                     level_attrs: dict,
                     window: int = WINDOW, log=print) -> int:
    """One (sample, level, chromosome) into an open HDF5. Returns index total.

    Counts are scattered by genomic position onto the .fai axis, never by row
    index: legacy blacklist filtering deleted rows, and any downstream
    coordinate taken from a row number after that is wrong by however many rows
    were removed above it.
    """
    missing = [k for k in LEVEL_ATTRS if k not in level_attrs]
    if missing:
        raise ValueError(f"{sample_id}/{level}: level attrs missing {missing}")

    fai = Path(str(fasta) + ".fai")
    n = chrom_length(fai, chrom)

    counts_arrays = preprocess.load_counts(counts, n)
    mask = build_mask(mappable_bed, chrom, n)
    sequence = load_sequence(fasta, chrom, n)

    shared = h5.require_group(f"shared/{chrom}")
    shared.create_dataset("mask", data=mask, dtype="?", chunks=(window,))
    shared.create_dataset("sequence", data=sequence, dtype="u1", chunks=(window,))

    level_group = h5.require_group(f"levels/{level}")
    level_group.attrs.update(level_attrs)
    chrom_group = level_group.require_group(chrom)
    # GC-corrected only. Nothing downstream reads the uncorrected column now
    # that M is gone, and it was 4 of the 10 bytes stored per position.
    chrom_group.create_dataset("counts_gc", data=counts_arrays.gc_corrected,
                               dtype="f4", chunks=(window,))

    rows = index_rows(sample_id, level, chrom, n, mask, window)
    total = append_index(h5, rows)
    log(f"  {chrom}: {n:,} bp, {mask.mean():.1%} masked, "
        f"{len(rows['start']):,} windows, {int(rows['keep'].sum()):,} kept "
        f"(index {total:,})")
    return total


# ── Window assembly ───────────────────────────────────────────────────────────


class WindowDataset(Dataset):
    """One /index row -> (input, target, valid).

        input   (6, WINDOW)  ch 0 smoothed coverage, ch 1-5 one-hot A,C,G,T,N
        target  (1, WINDOW)  the same window smoothed at the target level
        valid   (WINDOW,)    1.0 where the position carries real data

    Both branches take the identical chain and differ only in which level they
    read. The mask is a LOSS WEIGHT, not an input channel: whether the model
    benefits from seeing it is a separate measurable question, recorded in
    docs/OPEN_ISSUES.md rather than assumed here.

    The halo is READ FROM THE STORED FULL-CHROMOSOME ARRAY as [start-400,
    end+400], never manufactured by padding a bare window -- padding
    reintroduces exactly the boundary artifact the halo exists to remove.
    """

    def __init__(self, path: Path, input_level: str, target_level: str,
                 split: str | None = None, keep_only: bool = True,
                 halo: int = HALO):
        self.path = Path(path)
        self.input_level = input_level
        self.target_level = target_level
        self.halo = halo
        self._h5: h5py.File | None = None
        self._pid: int | None = None

        with h5py.File(self.path, "r") as h5:
            for level in (input_level, target_level):
                if f"levels/{level}" not in h5:
                    raise KeyError(f"{self.path.name}: no level {level!r}")
            index = h5["index"]
            select = np.ones(index["start"].shape[0], dtype=bool)
            if split is not None:
                select &= index["split"][:] == split.encode()
            if keep_only:
                select &= index["keep"][:]
            self.rows = np.flatnonzero(select)
        if self.rows.size == 0:
            raise ValueError(f"{self.path.name}: no rows for split={split!r} "
                             f"keep_only={keep_only}")

    def __len__(self) -> int:
        return int(self.rows.size)

    @property
    def h5(self) -> h5py.File:
        # h5py handles do not survive a fork, so each DataLoader worker opens its
        # own rather than inheriting the parent's.
        if self._h5 is None or self._pid != os.getpid():
            self._h5 = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._h5

    def _smoothed(self, level: str, chrom: str, lo: int, hi: int,
                  start: int, end: int, mask: np.ndarray) -> np.ndarray:
        """Read with halo, smooth, crop back to the window."""
        counts = self.h5[f"levels/{level}/{chrom}/counts_gc"][lo:hi]
        return preprocess.smooth(counts, mask)[start - lo:end - lo]

    def _valid(self, mask: np.ndarray, lo: int, start: int, end: int,
               chrom_len: int) -> np.ndarray:
        """Which positions of [start, end) carry real data.

        Invert the mask, erode by the Gaussian's reach, THEN crop: a position
        within 120 bp of a gap has a value computed from a partial kernel, and
        eroding before the crop is what lets the window's own edges be judged
        using their real neighbours instead of the crop boundary.

        Positions within HALO of a chromosome terminus have no real halo to
        read, so they are excluded here rather than repaired.

        Not included: "beyond the last written record". The HDF5 does not store
        where the counts file stopped, and for a sparse writer that boundary is
        the last non-zero position -- deriving it would make validity a function
        of coverage, which is the one thing it must never be. On the dense-era
        chr21 the region it would mark is already 100% masked by the mappable
        complement. See docs/OPEN_ISSUES.md.
        """
        valid = dilated_valid(mask, DILATION_RADIUS)[start - lo:end - lo]
        positions = np.arange(start, end)
        return valid & (positions >= self.halo) & (positions < chrom_len - self.halo)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.rows[i])
        index = self.h5["index"]
        chrom = index["chrom"][row].decode()
        start, end = int(index["start"][row]), int(index["end"][row])
        width = end - start

        chrom_len = self.h5[f"shared/{chrom}/mask"].shape[0]
        lo, hi = max(0, start - self.halo), min(chrom_len, end + self.halo)
        mask = self.h5[f"shared/{chrom}/mask"][lo:hi]

        coverage = self._smoothed(self.input_level, chrom, lo, hi, start, end, mask)
        target = self._smoothed(self.target_level, chrom, lo, hi, start, end, mask)
        valid = self._valid(mask, lo, start, end, chrom_len)

        # Sequence is stored as codes and one-hot encoded here, at load time.
        sequence = self.h5[f"shared/{chrom}/sequence"][start:end]
        model_input = np.zeros((N_INPUT_CHANNELS, width), dtype=np.float32)
        model_input[0] = coverage
        model_input[1 + sequence.astype(np.int64), np.arange(width)] = 1.0

        return (torch.from_numpy(model_input),
                torch.from_numpy(target[None, :].astype(np.float32)),
                torch.from_numpy(valid.astype(np.float32)))
