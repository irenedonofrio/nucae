# PORT_INVENTORY

Inventory of `../nucae_wip/` — PLAN.md Task 1. Nothing was moved, run, imported,
or modified.

All Python in `nucae_wip` lives under `input_processing/`: 30 files, 3,478 lines.
There is one loose notebook at the top level. Everything below is a path relative
to `../nucae_wip/`.

Read this as: **most of the package is wanted, but almost none of it is wanted
verbatim.** The five modules that do the science are built around the
*replacement* preprocessing chain, not the inherited one this repo is rebuilding.

---

## 1. Summary — what goes where

| Task it serves | Files | Verdict |
|---|---|---|
| HDF5 building (Task 5) | `parse.py`, `sequence.py`, `scalar.py`, `build.py`, 3 scripts | Port the logic, review the mask and M semantics |
| Dataloader (Task 6) | `window.py`, `dataset.py`, `mask.py` | **Do not port as-is** — wrong smoothing chain |
| Preprocessing (Task 2) | `fcc_smooth.py` | **Do not port** — PLAN Task 2 says write fresh |
| Evaluation (Task 4/7) | `MaxFinder.py` | Identical to the pipeline_audit copy; see §4 |
| Model | — | **Nothing.** Confirms PLAN's note |
| Training | — | **Nothing.** Confirms PLAN's note |
| None / outside the repo boundary | `config.py`, `check_gc.py`, `make_counts.sh` + its tests | See §5 |

---

## 2. Every Python file

### `input_processing/src/` — the package

| File | Lines | What it does |
|---|---:|---|
| `fcc_smooth.py` | 402 | Vendored copy of `pipeline_audit/fcc_smooth.py`. The whole smoothing engine: four blacklist modes, mask-aware filters, a chunked variant, plus `load_counts` and `load_blacklist_mask`. |
| `build.py` | 237 | HDF5 writer — root attrs, `/index` rows, per-level and per-chrom datasets, window tiling, `keep`/`valid_frac`. |
| `config.py` | 233 | Site YAML loader: resolves paths, reads the samples and metadata sheets, `--check` path verifier. |
| `scalar.py` | 139 | The normalising scalar M: trimmed mean, a pooled definition and a two-pass streaming version that matches it in bounded memory. |
| `dataset.py` | 127 | `torch` `Dataset` over the HDF5 — assembles one window into `x`/`y`/`cond` tensors. Also `masked_mse`. |
| `parse.py` | 117 | Counts TSV → dense 0-based arrays on the `.fai` axis, read in chunks, column-selectable. |
| `MaxFinder.py` | 96 | Vendored peak caller (`get_peaks`, `running_median`). |
| `mask.py` | 64 | BEDs → per-position validity mask (True = invalid), plus the single dilation function. |
| `window.py` | 62 | Window extraction: slice with halo, smooth, crop, divide by M. Owns `HALO`, `SIGMA`, `TRUNCATE`. |
| `sequence.py` | 37 | Reference FASTA → uint8 base codes, A=0 C=1 G=2 T=3 N=4, soft-mask aware. |
| `__init__.py` | 0 | Empty. |

### `input_processing/scripts/`

| File | Lines | What it does |
|---|---:|---|
| `make_scalar.py` | 38 | Per (sample, level): compute M, write JSON. |
| `build_h5.py` | 35 | Per sample: counts + scalars + refs cache → one HDF5. |
| `make_refs.py` | 33 | Cache `{chrom}.mask.npy` and `{chrom}.sequence.npy` once per (chrom, BEDs, FASTA). |
| `check_gc.py` | 32 | Diagnostic: confirm a GC-bias table's column layout before counting. |
| `make_counts.sh` | 357 | *(shell)* Vendored `02_fcc_count.sh` — BAM → sparse counts TSV. |

### `input_processing/tests/` — 1,555 lines over 13 files

| File | Lines | Covers |
|---|---:|---|
| `test_build.py` | 212 | Root attrs, window tiling, partial-window drop, chunk = one window, fixed-width index strings, splits, `keep == valid_frac > 0`, dilation, required level attrs. |
| `test_config.py` | 164 | Path anchoring, env expansion, sheet schema, duplicate detection, merged-sample flag, `method_constants` not settable from YAML. |
| `test_scalar.py` | 163 | Zeros included, mask exclusion, trim behaviour, pooled ≠ mean-of-per-chrom, streaming matches the oracle bitwise across trims and slice sizes. |
| `test_parse.py` | 160 | `.fai` length, 1-based → 0-based boundary, dtypes, gap and chunk-boundary detection, column selection. |
| `test_mask.py` | 158 | Mappability complemented, blacklist literal, BED half-open, chromosome leakage, dilation properties and monotonicity. |
| `test_dataset.py` | 146 | Tensor shapes, no NaN, one-hot exactly one per position, validity is the AND of both terms, `cond`, `masked_mse`, split/keep filtering. |
| `test_make_counts_run.py` | 135 | `make_counts.sh` against the `chrT` BAM fixture: contig naming, empty-output guard. |
| `test_window.py` | 132 | **Haloed window vs whole-chromosome smoothing**, support radius = 120, edge flags, scalar division, float32. |
| `test_sequence.py` | 97 | Base-for-base encoding, soft-masked bases are real, composition and GC against the generator. |
| `test_counts_seam.py` | 84 | The writer/reader seam: output is sparse, every row lands at position−1, omitted positions read back as zero. |
| `test_make_counts.py` | 65 | The vendored shell script differs from its source only by the documented patch. |
| `test_maxfinder.py` | 39 | Vendored copy matches its pin and its source; `get_peaks` default ≠ what the pipeline uses; scale invariance. |
| `__init__.py` | 0 | Empty. |

### `input_processing/tests/fixtures/`

| File | Lines | What it does |
|---|---:|---|
| `make_counts/generate.py` | 254 | Builds the synthetic `chrT` BAM/FASTA/BED fixture. Seeded `default_rng`, reproducible byte-for-byte. |
| `make_counts/regenerate_patch.py` | 17 | Rewrites `vendored.patch` when the divergence changes on purpose. |

### Not Python

`fcc_pipeline_stages.ipynb` (43 KB, repo top level) — walkthrough notebook, not
part of the package. `CLAUDE.md` (102 lines), `PLAN.md` (412), `README.md` (298),
`RESULTS.md` (497) — `RESULTS.md` is the measurement record behind these modules
and is worth reading before porting any of them. Two `.sbatch` files, YAML configs,
sample sheets, a `chr21.h5`, and cached `_refs` arrays.

---

## 3. FLAG — smoothing, masking, normalisation

This is the code PLAN says must not be ported as-is. It is not a corner of the
package; it is its centre.

### `window.py:prepare` — the replacement chain, in one function

```python
fcc_smooth.smooth_fcc(..., blacklist_mode="nan", whittaker=False,
                      gaussian_sigma=30, medfilt_kernel=None)
values = smoothed[...] / np.float32(scalar)
```

Three differences from the inherited chain this repo is rebuilding:

1. **No Whittaker step.** CLAUDE.md: "Do not remove the Whittaker step."
2. **No running-median subtraction.** CLAUDE.md: "Do not move the median."
3. **Blacklist handled as NaN, not zeroed**, and the Gaussian becomes a
   normalised convolution. CLAUDE.md: "Do not change the blacklist handling."
4. **Divides by the per-sample scalar M.** PLAN Task 6's chain does not include
   this step, and PLAN Task 6's channel spec has no place for it.

`fcc_smooth.py` labels this mode "NEW NUMERICAL BEHAVIOUR relative to the current
pipeline" in its own docstring.

### `fcc_smooth.py` — the new numerics live here

`_masked_chain`, `_gaussian_masked`, `_running_median_masked`, and the
`"nan"`/`"zero"` blacklist modes are all replacement behaviour. `_reference_chain`
*is* the inherited chain and is the only part that reproduces production. PLAN
Task 2 says to write the simple version fresh rather than copy this file — that
is the right call, and §4 gives a second reason.

### `mask.py` — masking, and a semantic that differs from PLAN Task 2

`build_mask` returns the union of *unmappable* and *blacklisted*. PLAN Task 2's
`load_blacklist_mask` is blacklist only; the mappability keep-list enters at
Task 5, where PLAN defines the mask as a three-way union including "beyond the
last written record". These are compatible but not the same function — decide at
Task 5 which one is canonical, and do not end up with both.

`dilated_valid` is deliberately the single dilation shared by `build.py` and
`dataset.py`, with a measured justification for `binary_dilation` over
`maximum_filter1d` (37 s / +0.4 GB vs 3.7 s / +4.2 GB on chr1). Worth keeping
that measurement; the note says the choice should flip on the cluster.

### `scalar.py` — normalisation, but PLAN asks for it

PLAN Task 5 defines M exactly as `scalar.py` computes it: trimmed mean over
non-masked positions of the uncorrected counts, zeros included, top 0.1%
trimmed, pooled across autosomes, before smoothing. This module is wanted.

Two cautions. `scalar.py:sample_scalar` is documented as computed on
`gc_corrected` in practice (`make_scalar.py` defaults to `--column gc_corrected`),
while PLAN Task 5 says **uncorrected**. `build_h5.py` reads `M_{level}_gc_corrected.json`
into the `M` attr and the uncorrected one into `M_raw` — the opposite assignment
to PLAN's. Resolve which column M is defined on before Task 5, because the gate
("within a factor of two of 0.0531 for BH01") depends on the answer.

**Deferred 2026-08-27.** M was dropped from PLAN Task 5 entirely. It belongs to
the proposed preprocessing chain; the inherited chain this repo rebuilds uses no
normalising scalar, so there is no scalar pre-pass and neither `M` nor `M_raw` is
written. Task 5's level attrs are now `nominal_depth`, `measured_depth`,
`gc_table_md5`, `bam_path`, `n_fragments`, and its gate dropped the `0.0531`
check.

The gc-corrected-vs-uncorrected question above is therefore **deferred, not
answered.** It comes back the moment a chain that needs a scalar is measured, and
the discrepancy it records — sheet default, `build_h5.py` assignment, and PLAN
disagreeing three ways — will still be there to resolve. `scalar.py` (139 lines)
and `test_scalar.py` (163 lines) stay parked in `nucae_wip`; nothing in phase one
needs them.

### `dataset.py` — extra channels not in PLAN Task 6

Produces a `cond` tensor of `[log M, log input_mean]` and a `row` index alongside
`x` and `y`. PLAN Task 6's spec is channels 0–6 and a target, nothing else.
`masked_mse` is a loss function — model-side, out of scope for phase one.

---

## 4. FLAG — duplicated primitives

CLAUDE.md: "This repository has been broken three times by two implementations of
the same primitive drifting apart." Here is where that risk sits.

### Two counts readers, with different contracts

| | `fcc_smooth.load_counts` | `parse.load_dense` |
|---|---|---|
| Returns | `pd.DataFrame` | tuple of dense `np.ndarray` |
| Requires | positions **dense and contiguous from 1** | positions **strictly increasing**, sparse allowed |
| Absent positions | rejected | left as zero |

Both assert exactly 4 columns for the same documented reason (a legacy script
rewrote these files in place, and binding the wrong column silently yields a
median-subtracted signal).

**This is the one substantive decision in this inventory.** PLAN Task 2 specifies
`load_counts(path) -> pd.DataFrame` asserting "dense-contiguous-from-1". But
`nucae_wip`'s own `make_counts.sh` emits **sparse** output — only covered
positions — and `test_counts_seam.py` exists specifically to prove that the
reader accepts what the writer produces. `RESULTS.md §3` records the counting
step as BLOCKED, and `build.py` carries a long note on `n_counts` explaining that
the dense-writer era is over.

So: if the counts files you will feed this repo came from the sparse writer,
PLAN Task 2's `load_counts` will reject them on the first file. If they came from
`../cfdna_tfp_profiling/` in its dense form, PLAN is right as written. I have not
inspected any real counts file, so I cannot tell you which. **Worth settling
before Task 2, not during it.**

### `MaxFinder.py` — and an md5 that does not match

`nucae_wip/input_processing/src/MaxFinder.py` is byte-identical to
`pipeline_audit/MaxFinder.py`. Both are md5 `1e3e84a422f41fc9e79cf6cb4590b24f`.

**CLAUDE.md and PLAN Task 4 both record the canonical md5 as
`1fbe12050e295ce7fa641d231ee49482`.** Those are the only two copies of the file
anywhere under `PhD/projects/github/`, and neither matches. **Task 4's gate will
fail as written.** Either the recorded hash is wrong, or the canonical file is
somewhere I have not looked. This needs resolving before Task 4 — do not "fix" it
by editing the file to match a hash.

**Resolved 2026-08-27.** The recorded hash was wrong; no third copy of the file
exists. The measured hash `1e3e84a422f41fc9e79cf6cb4590b24f` was adopted as
canonical, and `CLAUDE.md` and `PLAN.md` Task 4 were both corrected to match it.
`MaxFinder.py` itself was not modified — both copies still hash to the measured
value. Task 4's gate now passes as written. The finding above is left as found,
because "the pinned hash did not match the file" is worth being able to find
again.

The `min_len` note in CLAUDE.md checks out: `get_peaks(arr, window, max_gap=35, min_len=30)`
in the signature, and `test_maxfinder.py` has a test named
`test_get_peaks_signature_default_is_not_what_the_pipeline_uses`.

### One BED reader used for two different meanings

`mask.py` calls `fcc_smooth.load_blacklist_mask` for **both** the blacklist and
the mappability BED, complementing the latter at exactly one place. That is the
right pattern — one reader, one complement — and PLAN Task 2's `load_blacklist_mask`
should stay general enough to serve both at Task 5.

### Constants that will exist twice

`window.py` owns `HALO = 400`, `SIGMA = 30.0`, `TRUNCATE = 4.0` and derives
`support_radius() == 120`. PLAN Task 2 puts `SIGMA`, `HALO` and the rest at the
top of `preprocess.py`. `config.py:method_constants` imports them from four
modules to stamp them into the HDF5. One authoritative home, imported everywhere
else.

Note `window.py` has `SIGMA = 30.0` (float) where PLAN Task 2 writes `SIGMA = 30`
(int). No numerical consequence for `gaussian_filter1d`, but they should not be
two different literals in two files.

---

## 5. FLAG — outside this repo's boundary

CLAUDE.md: "Input to this repo is `chr*_counts.tsv.gz` ... produced by
`../cfdna_tfp_profiling/`. This repo starts from the counts files on disk."

`scripts/make_counts.sh` (357 lines) is the BAM → counts producer — the vendored
`02_fcc_count.sh`. It sits on the far side of that boundary, and it brings with
it `test_make_counts.py`, `test_make_counts_run.py`, `regenerate_patch.py`,
`vendored.patch`, and a BAM/FASTA/BED fixture. `check_gc.py` inspects a GC-bias
table used by that script, not by this repo.

Recommendation: **do not port any of it.** Roughly 850 lines of code, tests, and
fixtures that belong to `cfdna_tfp_profiling`.

`config.py` is not assigned to any task in PLAN, though PLAN's skeleton has a
`config/` directory and Task 7's script "loads config". It is 233 lines with 164
lines of tests for a feature set — YAML sites, samples sheets, metadata sheets,
merged-library flags — that may exceed what phase one needs. Flagging as
undecided, not as something to port by default.

---

## 6. The existing test suite

13 files, 1,555 lines, plus 271 lines of fixture generators. Fixtures are
synthetic and committed; the docstrings say "No real data" and that holds
everywhere I read.

**What it covers well:** the numerics with hand-computable answers (`scalar`),
coordinate-system boundaries (`parse`, `mask`, `sequence`), and the HDF5's
structural claims (`build`).

**The one test that maps directly onto a PLAN gate:**
`test_window.py:test_haloed_window_matches_whole_chromosome` is PLAN Task 6's
non-optional gate, already written. It asserts against whole-chromosome smoothing
on synthetic arrays, including a window overlapping a masked interval. **The
test's structure ports; its chain does not** — it exercises Gaussian-only
NaN-mode, and Task 6 needs the full inherited chain.

**What is missing, and matters most:** there is **no bit-identity test against the
production reference** anywhere in `nucae_wip`. `fcc_smooth.py`'s docstring points
at `check_reference.py`, which lives in `pipeline_audit/`, not here. PLAN Task 3 —
"the most important test in the repo" — has no ancestor to port. It must be
written fresh. I confirmed the assets it needs exist:
`pipeline_audit/fixture/reference_verbatim.py`,
`pipeline_audit/fixture/chr21_slice5000000.tsv.gz`, and
`pipeline_audit/check_reference.py` to read for reference.

`pipeline_audit/` also holds the Task 7 assets as PLAN describes them: both
`metric.py` and `metrics.py` (port the first, not the second), and both
`peak_calling.py` and `peak_calling_comparison.py` (same).

---

## 7. What I did not verify

- I did not run anything in `nucae_wip`, import from it, or open `chr21.h5`.
- I did not read any real counts file, so §4's dense-vs-sparse question is
  raised from source code, not settled from data.
- Line counts are `wc -l`. File descriptions are from reading the source and its
  docstrings; where a docstring states a measurement (dilation timings, chunking
  equivalence, halo recovery) I have repeated it as a claim made there, not
  independently checked it.
