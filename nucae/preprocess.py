"""The inherited FCC preprocessing chain, and the two readers that feed it.

The chain is:

    GC-corrected midpoints
      -> blacklist positions zeroed
      -> Whittaker-Eilers smoothing (lambda=1000, order=2)
      -> Gaussian smoothing (sigma=30)
      -> subtract running median (kernel=375)

Parts of it are known to be redundant or harmful (see ../pipeline_audit/AUDIT.md).
It is deliberately unchanged: a master's student is working against these results
and they must stay comparable until the replacement is measured and agreed.
`tests/reference/` is what proves `smooth` still reproduces production.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.signal import medfilt
from whittaker_eilers import WhittakerSmoother

SIGMA = 30
MEDFILT_KERNEL = 375
WHITTAKER_LAMBDA = 1000
WHITTAKER_ORDER = 2
HALO = 400   # reach = 4*SIGMA + (MEDFILT_KERNEL-1)//2 = 307, rounded up

COUNTS_COLUMNS = ("chrom", "position", "uncorrected", "gc_corrected")


class Counts(NamedTuple):
    """One chromosome's two count columns, on the 0-based reference axis.

    Named rather than a bare tuple because the whole point of the column-count
    assert below is that binding the wrong count column is silent, and a bare
    pair is one transposition away from exactly that.
    """

    uncorrected: np.ndarray
    gc_corrected: np.ndarray


def load_counts(path: Path, chrom_len: int) -> Counts:
    """Sparse counts TSV -> zero-filled float32 arrays of length `chrom_len`.

    02_fcc_count.sh writes only covered positions. An absent position is
    genuine zero coverage, so it is left as zero here; whether a zero position
    is USABLE is the mask's question, never this function's.

    Positions are 1-based in the file and must be strictly increasing within
    1..chrom_len. Contiguity is deliberately not required. Two chromosomes
    concatenated would restart positions, which the strictly-increasing check
    rejects, so the chrom column does not need reading -- as object dtype it
    costs one Python string per covered position.

    Reads the whole file at once, by PLAN Task 2. chr1 at full coverage is the
    case to watch on a 6 GB laptop.
    """
    path = Path(path)
    n_cols = pd.read_csv(path, sep="\t", header=None, nrows=1).shape[1]
    if n_cols != len(COUNTS_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(COUNTS_COLUMNS)} columns {COUNTS_COLUMNS}, "
            f"found {n_cols}. A file rewritten in place by a legacy script carries "
            f"extra columns and is NOT raw counts."
        )

    df = pd.read_csv(path, sep="\t", header=None, usecols=[1, 2, 3],
                     names=["position", "uncorrected", "gc_corrected"],
                     dtype={"position": "int64", "uncorrected": "float32",
                            "gc_corrected": "float32"})
    pos = df["position"].to_numpy()
    if pos.size == 0:
        raise ValueError(f"{path.name}: no rows")
    if np.any(np.diff(pos) <= 0):
        raise ValueError(
            f"{path.name}: positions are not strictly increasing. Either the file "
            f"is unsorted, or two chromosomes have been concatenated."
        )
    if pos[0] < 1 or pos[-1] > chrom_len:
        raise ValueError(
            f"{path.name}: positions {pos[0]}..{pos[-1]} fall outside 1..{chrom_len}; "
            f"the counts file and the reference disagree about the coordinate system."
        )

    out = Counts(np.zeros(chrom_len, dtype=np.float32),
                 np.zeros(chrom_len, dtype=np.float32))
    idx = pos - 1                      # 1-based file position -> 0-based index
    out.uncorrected[idx] = df["uncorrected"].to_numpy()
    out.gc_corrected[idx] = df["gc_corrected"].to_numpy()
    return out


def load_blacklist_mask(bed: Path, chrom: str, n: int) -> np.ndarray:
    """Boolean mask over 1-based positions 1..n, True = blacklisted.

    BED is 0-based half-open [start, end). A 1-based position p is inside iff
    start < p <= end, i.e. 0-based array index p-1 in [start, end). So the BED
    coordinates index the array directly and no offset is needed.
    """
    intervals = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 1, 2],
                            names=["chrom", "start", "end"],
                            dtype={"chrom": "string", "start": "int64",
                                   "end": "int64"})
    intervals = intervals[intervals["chrom"] == chrom]

    mask = np.zeros(n, dtype=bool)
    for start, end in zip(intervals["start"], intervals["end"]):
        # An interval clipped to nothing gives an empty slice, which assigns
        # nothing -- off-chromosome intervals need no special case.
        mask[max(0, start):min(n, end)] = True
    return mask


def smooth(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The inherited chain. float32 out, same length as `signal`.

    With `mask` all False this is bit-identical to
    `fcc_pipeline/04_smooth_and_normalize.py:smooth_and_normalise` given the
    same float64 input, which is what `tests/reference/` asserts.

    Works in float64 because the reference does: it parses each value straight
    from text into a float64 array. Handing this function float32 counts (what
    `load_counts` returns, and what the HDF5 stores) is a different input, so
    the result is the chain applied faithfully to float32 data -- not the
    reference's bits.

    Do not reorder the steps, drop the Whittaker, or move the median.
    """
    signal = np.asarray(signal)
    if mask.shape != signal.shape:
        raise ValueError(f"mask {mask.shape} != signal {signal.shape}")

    x = signal.astype(np.float64, copy=True)
    x[mask] = 0.0

    smoother = WhittakerSmoother(lmbda=WHITTAKER_LAMBDA, order=WHITTAKER_ORDER,
                                 data_length=len(x))
    x = np.array(smoother.smooth(x))
    x = scipy.ndimage.gaussian_filter1d(x, sigma=SIGMA)
    x = x - medfilt(x, kernel_size=MEDFILT_KERNEL)
    return x.astype(np.float32)
