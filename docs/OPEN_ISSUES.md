# OPEN ISSUES

Things deliberately left undecided, with what it would take to decide them.
An entry here is a measurement someone has to run, not a bug to fix quietly.

---

## 1. Does the model benefit from seeing the mask? — DEFERRED EXPERIMENT

The dataloader returns the mask as a **loss weight** (`valid`), not as a model
input channel. Input is 6 channels: coverage plus one-hot A,C,G,T,N.

A 7th channel carrying validity is a plausible design — it would let the model
know which of its inputs are real rather than inferring it from a flat run of
zeros. It is **not** assumed to help, and it is not free: it changes the input
contract, so every checkpoint trained one way is incompatible with the other.

**To settle it:** train two arms differing only in that channel, same seed, same
recipe, and compare on held-out chromosomes. Until then the input stays 6
channels.

---

## 2. "Beyond the last written record" is not part of `valid` — BY DESIGN

`WindowDataset._valid` excludes positions that are masked, positions within
120 bp of a masked region, and positions within 400 bp of a chromosome
terminus. It does **not** exclude positions past the end of the counts file.

Two reasons:

- The HDF5 does not store where the counts file stopped, and recovering it
  means asking where coverage ends. For a sparse writer — which
  `02_fcc_count.sh` is — that boundary IS the last non-zero position, so using
  it would make validity a function of coverage. CLAUDE.md: a position that is
  zero at 1x and non-zero at 45x is empty, not missing.
- It buys nothing measurable. On the dense-era chr21 counts, the 10,077 bp
  between the file's last record (46,699,906) and the .fai length (46,709,983)
  is **100% masked already** by the mappable complement.

**To change it:** record the file's extent as a level attr at build time
(`load_counts` knows `pos[-1]`), and OR it in here. That is a Task 5 change and
a rebuild. Worth doing only if a real case is found where the tail is mappable.

---

## 3. Two `nucae` packages are installed under the same import name

The conda env `nucae` has `nucae` installed editable from
`/Users/idonof/PhD/projects/NucAE`, not from this repository. `import nucae`
from any directory outside this repo loads the other project. Inside the repo
it works by cwd shadowing, which is why it went unnoticed.

**To fix:** re-run `pip install -e .` in this repository. Until then every
command here needs `PYTHONPATH` set to the repo root.

---

## 4. Preprocessing differs from the model repo at `PhD/projects/NucAE`

That repo's chain is `blacklist -> whittaker -> mean-normalise -> gaussian ->
median`. This repo's is `blacklist -> whittaker -> gaussian -> median`, with no
per-window mean-normalisation, matching production; `tests/reference/` proves
the match bit-for-bit.

Any checkpoint trained there saw windows divided by their own mean, so tensors
from this dataloader are **not in the same units**. That repo also records its
Whittaker step as unvalidated, and a known ~6 bp disagreement between two 5x
signals attributed to more than one preprocessing code path.

**To settle it:** run both chains on identical input and diff, before any
checkpoint from there is evaluated on data from here.

---

## 5. Bit-identity of a haloed window is unattainable, and peak recovery does not reproduce

**Settled part.** CLAUDE.md previously asserted that a haloed window "gives
values identical to smoothing the whole chromosome". That is true for the
Gaussian (reach 120 bp) and the running median (reach 187 bp), both inside the
400 bp halo -- measured on chr21 with the Whittaker disabled, 0 differing
positions over 50 windows.

It is false for the full chain. Whittaker-Eilers is a global banded solve with
no compact support, so no finite halo reaches bit-identity:

| halo | windows differing (of 20) | positions differing |
|---|---|---|
| 400 | 7 | 3,591 |
| 1,000 | 5 | 2,450 |
| 2,000 | 4 | 267 |
| 5,000 | 5 | 268 |
| 20,000 | 3 | 4 |

The residual is bounded far below float32 storage precision (max|delta| ~6e-11
vs a float32 ULP of ~3e-8 at the signal's scale), so `tests/unit/test_halo.py`
gates on exactness with the Whittaker off and on that bound with it on.
CLAUDE.md's windowing section has been corrected.

**UNRESOLVED part.** CLAUDE.md quoted "peak recovery 1.0000 (median and
minimum over 200 windows)". Measured here on BH01 chr21 with a 400 bp halo,
MaxFinder at `window=375, max_gap=35, min_len=10`, over the 163 interior
windows that contain peaks:

```
median 1.0000    min 0.9412    mean 0.9982
identical peak sets: 151/163 (92.64%)
```

The median reproduces. **The minimum does not.**

The provenance differs and the difference is not resolved: these counts are
BH01 from the DENSE-era writer via `pipeline_audit/data/`, and the sample,
coverage level, window selection and peak-caller parameters behind the original
measurement are not recorded anywhere findable. So this is not evidence that
the original number was wrong -- only that it does not reproduce under the one
setup that can be run here.

**To settle it:** recover the original measurement's inputs, or re-measure on
the intended production data once a sparse-writer HDF5 exists, and record the
parameters alongside the number this time.

---

## 6. The training path applies no validity mask — MATCHES WHAT RAN

`nucae/module.py` computes MSE over all 25,000 positions of every window.
`WindowDataset` returns a `valid` mask marking positions within 120 bp of a
masked region and within 400 bp of a chromosome terminus; the training contract
in `nucae/prewindowed.py` is a 2-tuple and has nowhere to put it. The
pre-windowed files carry no equivalent information at all.

This is **deliberate and it is a defect**. Masked positions carry values
computed from a partial kernel, and the loss currently treats them as real
signal. It is retained because it is what the cluster trained on: changing it
would mean the reproduction gate no longer compares like with like.

**To fix, and this is the first planned improvement:** carry `valid` through as
a third element, weight the MSE by it, and compute Pearson over valid positions
only. That changes every number, so it must be a single deliberate change made
against a reproduced baseline, never folded in alongside anything else.

---

## 7. The pre-windowed files' normalisation is unknown

`old_nucae/convert_to_hdf5.py` writes `metadata/normalised = True`
unconditionally. It performs no normalisation itself, and neither does
`cfdna_dataset_hdf5.py`; both simply pass through signals its docstring calls
"already normalised". The step happened in a pipeline that predates every file
we hold, so the attribute records a belief rather than a measurement.

This matters beyond bookkeeping. If each window was divided by its own mean, the
windows are dimensionless and amplitude no longer tracks sequencing depth --
which would make any MSE quoted on them incomparable with an MSE from the
counts path, and would void any `alpha` chosen for a future composite loss
(entry 3 of `PhD/projects/NucAE/docs/OPEN_ISSUES.md`, where alpha is shown to
scale as 1/var).

**To settle it:** `python old_nucae/inspect_h5.py --hdf5 <cfdna_*.h5>`. Both
arms of a window are the same locus at different depths, so
`std(full)/std(under)` near 1 indicates a per-window rescaling and a ratio near
the depth ratio indicates none. Until it has been run, do not describe these
windows as normalised in any particular way.

---

## 8. Two silent defects in the pre-windowed files themselves

Both originate in `old_nucae/convert_to_hdf5.py` and affect data already on
disk. Neither is repaired by reading; the files would have to be rebuilt.

- **A failed line becomes an all-zero window rather than a dropped one.** The
  arrays are pre-allocated with `np.zeros` and the `except` branch never writes
  or removes row `i`. The count is printed once and discarded. Such a window
  then trains: MSE pushes the model toward predicting zero there.
- **A failed line shifts every subsequent chromosome LABEL by one.**
  `chroms.append` runs before the statements that can raise, so a failure
  appends a second entry for the same line. `start` and `end` are indexed by `i`
  and stay aligned, so the label decouples from its own coordinates. The
  script's `chroms[:N]` truncates the tail rather than correcting the offset.
  Consequence is narrow but real: the bedGraph exported by
  `scripts/04_evaluate_reconstruction.py` would carry wrong chromosome names.
  Per-window metrics are unaffected.

**To check:** `inspect_h5.py` counts all-zero windows and reports any
`chrom == b'unknown'`. A single `unknown` means at least one line failed; if it
failed after the chrom append, everything after it is shifted.
