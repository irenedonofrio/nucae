"""The reference gate: `nucae.preprocess.smooth` must reproduce production.

`reference_verbatim.py` is a byte-identical copy of the production script
`fcc_pipeline/04_smooth_and_normalize.py`. If this test goes red, `preprocess.py`
is wrong -- fix it, do not relax the test. It is what makes results produced
before and after the restructuring comparable to each other.

The fixture is chr21 positions 20,000,001..25,000,000 (1-based, inclusive) cut
from the real counts file -- 10.8% of positions covered. A slice with no coverage
proves nothing here: every chain maps all-zeros to all-zeros, so the gate would
pass with the Whittaker step deleted.

The fixture is parsed to float64 ONCE, with the reference's own parse, and the
same values go to both sides. That matters: the reference builds its input with
`float(parts[3])` straight into float64, so handing one side text-parsed float64
and the other side float32 counts would compare two different inputs and fail
for a reason that has nothing to do with the chain. `load_counts` is therefore
deliberately not used here.
"""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from nucae import preprocess

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "chr21_20000001_25000000.tsv.gz"
SIGNAL_COL = 3        # 0-based; the GC-corrected column, and the script's default


def _load_reference():
    """Import the vendored copy without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location(
        "reference_verbatim", HERE / "reference_verbatim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def signal() -> np.ndarray:
    """The fixture slice as float64, parsed exactly as the reference parses it.

    Same three lines as `reference_verbatim.process_chromosome`: split on tab,
    `float()` the GC-corrected column, then one `np.array(..., dtype=np.float64)`.
    """
    values = []
    with gzip.open(FIXTURE, "rt") as fh:
        for line in fh:
            values.append(float(line.split("\t")[SIGNAL_COL]))
    return np.array(values, dtype=np.float64)


@pytest.fixture(scope="module")
def outputs(signal) -> tuple[np.ndarray, np.ndarray]:
    """Both chains, on copies of the one parsed array.

    Copies because the reference is free to smooth in place; the array itself is
    asserted unchanged afterwards, which is what proves both sides saw the same
    input rather than one seeing the other's leftovers.
    """
    before = signal.copy()
    theirs = _load_reference().smooth_and_normalise(signal.copy())
    mine = preprocess.smooth(signal.copy(), np.zeros(signal.shape, dtype=bool))
    assert np.array_equal(signal, before), "the parsed array was mutated in place"
    return theirs, mine


def test_the_fixture_is_the_slice_it_claims(signal):
    assert signal.shape == (5_000_000,)
    assert signal.dtype == np.float64


def test_both_chains_return_float32_of_the_same_length(outputs, signal):
    theirs, mine = outputs
    assert theirs.dtype == np.float32 and mine.dtype == np.float32
    assert theirs.shape == mine.shape == signal.shape


def test_smooth_is_bit_identical_to_the_production_reference(outputs):
    """Bitwise, not approximate. Compared as int32 so NaN compares by bits."""
    theirs, mine = outputs
    n_diff = int(np.count_nonzero(theirs.view(np.int32) != mine.view(np.int32)))
    assert n_diff == 0, (
        f"{n_diff:,} of {theirs.size:,} positions differ from "
        f"04_smooth_and_normalize.py; max|delta| = "
        f"{np.abs(theirs.astype(np.float64) - mine.astype(np.float64)).max():.3e}"
    )
