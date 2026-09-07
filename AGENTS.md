# NucAE

## What this project is

A U-Net that reconstructs dense cell-free DNA fragment centre coverage (FCC)
from sparse sequencing input. Input is 1–5x coverage in 25 kb windows; target
is the same window at high coverage (~45x). The downstream clinical goal is
discriminating prostate cancer subtypes (ARPC vs NEPC) from a blood draw,
without biopsy.

The signal is a per-base count of DNA fragment midpoints. A fragment midpoint
approximates a nucleosome centre, so the track carries nucleosome positioning,
and dips in the track at transcription factor binding sites carry information
about which factors are active — which is what distinguishes the subtypes.

The task spans a continuum. At ~5x the signal is present but noisy, so the job
is denoising. At ~1x the input is below the information floor, so the job is
closer to conditional generation. This matters because a model can score well
on shape metrics by reproducing a canonical nucleosome template it memorised,
without reading its input. Guard against that framing when evaluating.

## Repository boundary

Input to this repo is `chr*_counts.tsv.gz`: 4 columns, tab-separated —
chromosome, 1-based position, uncorrected count, GC-corrected count.

Those files are produced by `../cfdna_tfp_profiling/`, from BAMs. That is a
separate project. This repo starts from the counts files on disk. There is no
import across the boundary in either direction.

There is a SECOND, INHERITED input: the pre-windowed `cfdna_*.h5` read by
`nucae/prewindowed.py`, holding 25 kb windows of already-smoothed and
already-scaled coverage. It is the input that produced every existing
checkpoint, and it is admitted here for exactly one reason — without it those
checkpoints cannot be re-scored, and there is no result to anchor new work
against. What the upstream scaling was is not recorded in any code we hold; see
`docs/OPEN_ISSUES.md`.

It is inherited, not preferred. New work belongs on the counts path, where the
preprocessing is reproducible and gated. Nothing may be added to
`prewindowed.py` beyond reading the format as it exists.

## Frozen — read-only, never edit

```
../pipeline_audit/    Completed preprocessing audit. The evidence behind the
                      decisions in this repo. Read for reference; do not run,
                      do not import from, do not modify.
../fcc_pipeline/      Production reference implementation. tests/reference/
                      compares against a vendored copy of it.
../Griffin/           Vendored upstream (Doebley et al. 2022).
../nucae_wip/         Parked work in progress on a future preprocessing
                      change. Read it when porting code across. Do not run it,
                      do not import from it, do not resume work in it.

nucae/MaxFinder.py    Canonical peak caller, md5 1e3e84a422f41fc9e79cf6cb4590b24f.
                      Copied verbatim. Never modify, never reimplement.
                      Note: its __main__ block hardcodes min_len=10 while the
                      function signature defaults to 30. Always pass min_len
                      explicitly.
```

## Preprocessing: do not improve it

The preprocessing chain in `nucae/preprocess.py` is inherited:

```
GC-corrected midpoints
  -> blacklist positions zeroed
  -> Whittaker-Eilers smoothing (lambda=1000, order=2)
  -> Gaussian smoothing (sigma=30)
  -> subtract running median (kernel=375)
```

Parts of this chain are known to be redundant or harmful. That is established
and documented in `../pipeline_audit/AUDIT.md`. It is deliberately unchanged
here: a master's student is working against these results and they must stay
comparable to existing ones until the replacement is measured and agreed.

Do not remove the Whittaker step. Do not change the blacklist handling. Do not
move the median. Do not add a config file to make the chain switchable — the
replacement chain does not exist yet, and a config for a variation that does
not exist is premature abstraction.

When the chain does change, it changes by editing that one function.

### Precision: float64 in the gate, float32 on disk

`tests/reference/` proves the chain matches production in float64 -- zero
differing bits over 5,000,000 positions of real chr21 signal.

The HDF5 stores counts as float32, so the pipeline hands `smooth` float32 where
the reference handed it float64. Measured on that same fixture, this moves
2,351,648 of 5,000,000 positions, by at most 3e-08 absolute: 1.8e-09 relative to
the raw count scale, 9.2e-08 relative to the smoothed track's own range (the
smoothed signal is ~50x smaller than the counts it came from, which is why the
two denominators differ). Either way it is far below shot noise at these depths.
This is an accepted rounding step, not a defect.

Do not claim bit-identity for output produced through the HDF5 path. The gate's
claim is about `smooth` in float64, and nothing wider.

## Windowing is exact, and the halo is why

Both filters are local. Gaussian sigma=30 reaches 4*sigma = 120 bp; medfilt 375
reaches (375-1)/2 = 187 bp. Total reach is 307 bp.

A 25 kb window read with a 400 bp halo on each side, smoothed, then cropped,
reproduces a whole-chromosome smooth -- but not uniformly, because the chain
does not have uniform support.

The Gaussian and the running median ARE reproduced exactly: their reach is 120
and 187 bp, both inside the halo. Measured on chr21 with the Whittaker step
disabled: 0 differing positions over 50 windows, at every halo tried.

The Whittaker step is NOT. Whittaker-Eilers solves a global banded system and
has no compact support, so its influence never reaches zero and bit-identity is
unattainable at ANY halo -- measured, still 4 differing positions at a halo of
20,000. What is true is that the residual is bounded far below the precision
the counts are stored in: max|delta| ~6e-11 against a float32 ULP of ~3e-8 at
the signal's scale. The windowing residual is invisible at float32.

The 307 bp derivation above covers the Gaussian and the median only. Whittaker
is not in that sum and cannot be.

Peak recovery with a 400 bp halo, measured on BH01 chr21: median 1.0000,
minimum 0.9412, 92.6% of windows with an identical peak set. The minimum does
not reproduce an earlier claim of 1.0000; see docs/OPEN_ISSUES.md entry 5.

The halo must be READ FROM THE STORED ARRAY, not created by padding. Padding a
bare 25 kb window reintroduces exactly the boundary artifact the halo exists to
avoid.

Within 400 bp of a chromosome terminus no real halo exists. Those positions are
excluded via the window validity channel, not repaired.

## Storage: raw counts on disk, preprocessing at load

The HDF5 stores raw GC-corrected counts only. Smoothing and normalisation
happen in the dataloader.

Reasons, in order: smoothing is destructive and some of its parameters are
still open questions; one HDF5 must serve several consumers with different
needs (this model wants it smoothed, Griffin wants it raw); and a decision
baked onto disk costs a rebuild to revisit.

Uncorrected counts are NOT stored. Nothing downstream reads them: Griffin
consumes GC-corrected counts too, and the normalising scalar M -- the one thing
that needed the uncorrected column -- was dropped from phase one. Storing them
was 4 of 10 bytes per position, 11.5 GB per sample across the autosomes.

Sequence is stored as uint8, 0-4 for A/C/G/T/N. One-hot encoding, k-mer
embedding, or any other representation is derived at load time. Store the
bases, not the encoding.

## Environments

```
Local:   conda env `nucae`     on a MacBook, ~6 GB RAM.
                               NOT `ae_env` — that is the cluster env and
                               using it locally has cost real time before.
Cluster: LeoMed (SLURM), conda env `ae_env`, no internet on compute nodes.
         Weights & Biases runs in offline mode, synced by rsync.
```

Memory constraint: one chromosome per pass. Never concatenate chromosomes.

## Key references

- Snyder et al. 2016 — cfDNA nucleosome footprint, WPS
- Doebley et al. 2022 — Griffin, fragment centre coverage, TFBS compositing
- Ronneberger et al. 2015 — U-Net
- Lambrinos et al. 2025 — SAUNA; the source of the MaxFinder peak caller

---

# Working with me

Use plain, simple, understandable, correct language. Introduce only concepts
needed for the current decision, define unavoidable terms, and prefer one small
example over extra abstraction or process detail. Explain from first
principles.

Lead with the result, decision, or finding. Distinguish observed facts from
interpretation and recommendation.

I usually approach the codebase as an outside observer. I bring the research
idea, define what I want the system to accomplish, describe the scientific
intent, and work through the design with you. Do not expect me to know the
correct programming terminology or the internal workings of the code.

Translate my goals into precise technical requirements and a sound
implementation. If I use an approximate term, infer the intended meaning and
introduce the correct term only when it helps us communicate.

Do not ask me to choose between technical options without explaining their
practical consequences and recommending the option you think is best. Ask me
when the answer depends on scientific intent, desired behavior, cost, risk, or
another decision only I can make.

When an idea is still developing, work through it with me before treating the
design as settled. Give me a useful mental model of what goes in, what the
important stages do, what comes out, and where the important risks are. I
should not need to understand every function or implementation detail to
participate in the design.

When I question a design or rule, explain the concrete reason for it. Do not
hide the decision behind terminology, process language, or an appeal to best
practice.

## Lean research work

We are writing high-quality, maintainable, understandable, lean research code.

Choose the simplest design that preserves the scientific behavior.

Design data structures first. Good data structures naturally lead to simple,
clean, and maintainable logic. Prefer plain records and direct
transformations.

Split files and functionality by responsibility, not by line count. Keep
related scientific logic together, and do not split a cohesive module merely
because it is large.

Eliminate edge cases means refactoring logic so special cases are treated
identically to normal cases, removing the need for special conditional checks.

Avoid deep nesting. Deep indentation is a warning sign of overly complex code.
If a function requires excessive nesting, it is doing too much and should be
broken down into smaller, self-contained units.

Refactor code that needs comments to explain its mechanics. Use comments and
docstrings for scientific intent, assumptions, units, references, and reasons
that cannot be made clear through names and structure.

Add an abstraction only when it hides real complexity or supports variation
that actually exists. Do not introduce wrappers, registries, factories,
interfaces, lifecycle objects, or adapter layers for hypothetical future
needs.

Prefer direct function calls and ordinary control flow over persisted state
machines, callback frameworks, or orchestration infrastructure.

Store each fact in one authoritative place. Derive it elsewhere when derivation
is simple and unambiguous.

When replacing an implementation, remove the obsolete path — UNLESS this file
states that the old behavior is deliberately retained. The inherited
preprocessing chain is such a case: it is retained on purpose, for a stated
reason, and is not dead code. Ask before removing anything this file describes
as inherited, frozen, or deliberately unchanged.

## Straightforward research pipelines

Default to the smallest complete end-to-end workflow that answers the current
scientific question. Make its maintained path obvious:

input -> preparation -> scientific computation or model -> evaluation -> result

Straightforward does not mean one file. Separate modules by real
responsibility, keep related scientific logic together, and compose the stages
with shallow ordinary control flow. Pass plain records, arrays, tables, and
paths between stages.

For a new experiment or reproduction, first make the minimal scientifically
valid path run on representative data. Expand it only after encountering a
concrete limitation.

Add a stage, abstraction, persisted artifact, compatibility path, registry, or
validation layer only when it solves a present requirement. Before adding it,
identify the concrete failure it prevents and why direct code or trusted
operator ownership is insufficient.

Do not turn a well-scoped experiment into a framework, platform, multi-ticket
program, or generalized workflow.

Prefer one obvious operator entry point per workflow. Test the important
scientific transformations and the real end-to-end path. Do not build a
parallel test framework around the implementation.

## Scripts and entry points

Scripts in `scripts/` are numbered and run by hand, in order, with each output
checked before the next is run. Each is roughly 35 lines: parse arguments, load
what it needs, call one function from the `nucae` package, write output.

No orchestrators, no Makefiles, no DAG frameworks, no Snakemake. Long-running
steps go to SLURM as plain sbatch files.

All real logic lives in the `nucae` package and is importable. If something
would need to be copy-pasted between two scripts, it belongs in the package.

This repository has been broken three times by two implementations of the same
primitive drifting apart. One canonical implementation per primitive. If a
function already exists, import it rather than writing a second one.

## Scientific correctness

Names such as invariant, requirement, policy, acceptance criterion, or best
practice do not justify enforcement. Software can enforce observable
conditions; honesty, review discipline, approvals, and governance are process
rules.

Do not add, recommend, or flag missing provenance manifests, content hashes,
revision tracking, dependency snapshots, or cross-artifact validation. Under
the trusted-operator model, the configuration supplied by the operator is
sufficient.

Cache selection and reuse are operator responsibilities. Treat an existing
cache path as the intended cache and load it directly. Do not add cache
fingerprints, content hashes, or automatic stale-cache detection unless
explicitly requested.

Represent leakage prevention, subject separation, blinding, and similar
scientific constraints structurally when possible. Prefer a data flow in which
forbidden information is unavailable over a check that merely claims it was
not used.

Validate exact completeness and alignment when records enter a scientific
computation. Do not repeatedly validate the complete schema of the same trusted
internal record at every later stage.

A scientifically valid computation remains valid when its result is
unfavorable. Do not turn expected quality, significance, loss, calibration, or
confidence intervals into runtime completion gates.

Tests should provide evidence that the implemented computation behaves
correctly and that the information needed to rerun and interpret it is
retained. They should not require a preferred scientific conclusion.

## Measurement discipline

Every claim about the data or the pipeline needs stated provenance, as one of:
analytical, source code, real data, literature, or guess. Say which.

Do not draw conclusions from synthetic data alone. Synthetic data is for
validating that a measurement harness works, not for settling a scientific
question.

Fix the metric before running anything. Switching measuring sticks mid-analysis
is not allowed.

State a prediction before each run. When a claim turns out to be wrong, flag
the retraction explicitly rather than quietly revising it.

## Tests and verification

Test behavior through the same interfaces used by callers and operators.

`tests/reference/` proves that this repo's preprocessing reproduces the
production reference bit-for-bit. It must stay green. It is what makes results
from before and after any restructuring comparable.

Concentrate other tests on the applicable risks: scientific transformations,
stable keys and seeds, exact completeness and alignment, structural separation,
and important numerical failure modes.

Do not mirror the implementation in tests or build a test-only lifecycle
framework around a direct script.

Run checks proportional to the change. Report exactly what was and was not
verified. Unit and synthetic tests do not prove real hardware, dependency,
scheduler, network, or full-data execution.

## Human understanding and technical responsibility

I must deeply understand the system architecture to maintain it and take
responsibility for any flaws. This does not mean I arrive with that
understanding or already know the terminology and code internals. Build that
understanding with me. Start with the architecture, data flow, scientific
assumptions, and important failure modes rather than every implementation
detail.

Take responsibility for translating my research intent into a coherent
technical design and high-quality implementation. Investigate the codebase,
identify reasonable options, recommend one, and explain the parts that
materially affect my goals.

Do not use operator responsibility as an excuse for confusing code, missing
scientific checks, weak tests, or avoidable failure modes.

Surface assumptions, limitations, and unresolved risks clearly. If the code
cannot safely or correctly accomplish what I asked for, explain why and what
would need to change.