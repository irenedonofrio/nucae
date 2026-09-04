"""The bookkeeping in nucae.data and the two readers it depends on.

NOT THE MATHS. `preprocess.smooth` is gated bit-identically against production
in tests/reference/; nothing here re-checks it. What is checked here is
coordinates, mask polarity, column binding and window boundaries -- the code
that decides WHICH numbers the chain is applied to.

That distinction is the point. Wrong maths tends to look wrong. Wrong
bookkeeping returns a plausible signal from the wrong place, and every summary
metric downstream is computed happily on it. Every test below fixes a value
that a plausible-looking mistake would change:

    an interval at 10..20 must mask 10..20, not 11..21
    a MAPPABLE region must come back VALID, not masked
    60,000 bp must give two windows, not three
    the gc_corrected column must land in `gc_corrected`

Synthetic fixtures throughout, because these are questions with exact answers.
A real slice can only be checked for plausibility, which is what the code
already achieves when it is wrong.
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from nucae import data, preprocess

LINE = 60          # FASTA line width, as samtools writes it


def write_fasta(directory, chroms: dict[str, str]):
    """A FASTA and the .fai samtools would produce for it. Returns the path."""
    path = directory / "ref.fa"
    text, index, offset = [], [], 0
    for name, seq in chroms.items():
        header = f">{name}\n"
        body = "".join(seq[i:i + LINE] + "\n" for i in range(0, len(seq), LINE))
        index.append(f"{name}\t{len(seq)}\t{offset + len(header)}\t{LINE}\t{LINE + 1}")
        offset += len(header) + len(body)
        text.append(header + body)
    path.write_text("".join(text))
    (directory / "ref.fa.fai").write_text("\n".join(index) + "\n")
    return path


def write_bed(directory, intervals, name="regions.bed"):
    path = directory / name
    pd.DataFrame(intervals).to_csv(path, sep="\t", header=False, index=False)
    return path


def write_counts(directory, chrom, rows, name=None):
    """rows: (position, uncorrected, gc_corrected), positions 1-based."""
    path = directory / (name or f"{chrom}_counts.tsv.gz")
    with gzip.open(path, "wt") as fh:
        for position, uncorrected, corrected in rows:
            fh.write(f"{chrom}\t{position}\t{uncorrected}\t{corrected}\n")
    return path


# ── the .fai reader ───────────────────────────────────────────────────────────

def test_chrom_length_reads_the_declared_length(tmp_path):
    write_fasta(tmp_path, {"chr1": "A" * 130, "chr7": "C" * 70})
    assert data.chrom_length(tmp_path / "ref.fa.fai", "chr1") == 130
    assert data.chrom_length(tmp_path / "ref.fa.fai", "chr7") == 70


def test_chrom_length_raises_rather_than_guessing(tmp_path):
    write_fasta(tmp_path, {"chr1": "A" * 130})
    with pytest.raises(KeyError):
        data.chrom_length(tmp_path / "ref.fa.fai", "chr21")


# ── BED coordinates ───────────────────────────────────────────────────────────

def test_a_bed_interval_covers_exactly_its_own_positions(tmp_path):
    """BED is 0-based half-open: [10, 20) is array indices 10..19 and no others.

    The single most consequential line in the file. One base either way and
    every mask in every window is shifted for the life of the dataset.
    """
    bed = write_bed(tmp_path, [("chr1", 10, 20)])
    mask = preprocess.load_blacklist_mask(bed, "chr1", 30)

    assert mask[10:20].all()
    assert not mask[:10].any()
    assert not mask[20:].any()
    assert int(mask.sum()) == 10


def test_intervals_on_other_chromosomes_are_ignored(tmp_path):
    bed = write_bed(tmp_path, [("chr1", 0, 30), ("chr2", 0, 30)])
    assert int(preprocess.load_blacklist_mask(bed, "chr2", 30).sum()) == 30
    assert int(preprocess.load_blacklist_mask(bed, "chr3", 30).sum()) == 0


def test_an_interval_running_past_the_end_is_clipped(tmp_path):
    bed = write_bed(tmp_path, [("chr1", 25, 999)])
    mask = preprocess.load_blacklist_mask(bed, "chr1", 30)
    assert int(mask.sum()) == 5 and mask[25:].all()


# ── mask polarity ─────────────────────────────────────────────────────────────

def test_mappable_regions_come_back_valid_not_masked(tmp_path):
    """build_mask takes a KEEP-list and returns True = INVALID.

    The polarity is inverted exactly once, in build_mask. Inverted twice or not
    at all, training proceeds on precisely the regions meant to be excluded and
    nothing anywhere raises.
    """
    bed = write_bed(tmp_path, [("chr1", 10, 20)])          # 10..19 are MAPPABLE
    mask = data.build_mask(bed, "chr1", 30)                 # True = invalid

    assert not mask[10:20].any(), "mappable positions were marked invalid"
    assert mask[:10].all() and mask[20:].all(), "unmappable positions were marked valid"


def test_dilated_valid_shrinks_the_valid_region_by_the_radius(tmp_path):
    """A position within `radius` of a masked base has a kernel that touched it."""
    mask = np.zeros(100, dtype=bool)
    mask[50] = True                                          # one bad base

    valid = data.dilated_valid(mask, radius=3)
    assert not valid[47:54].any(), "the neighbourhood of a masked base stayed valid"
    assert valid[:47].all() and valid[54:].all(), "dilation reached too far"


def test_dilated_valid_with_no_radius_is_the_plain_complement():
    mask = np.zeros(10, dtype=bool)
    mask[4] = True
    assert np.array_equal(data.dilated_valid(mask, radius=0), ~mask)


# ── window boundaries ─────────────────────────────────────────────────────────

def test_a_trailing_partial_window_is_not_emitted():
    """60,000 bp is two windows of 25,000 and 10,000 bp left over, not three.

    The model takes a fixed input length, so a short window is not a window.
    """
    starts, ends = data.window_bounds(60_000, window=25_000)
    assert starts.tolist() == [0, 25_000]
    assert ends.tolist() == [25_000, 50_000]


def test_an_exact_multiple_uses_every_base():
    starts, ends = data.window_bounds(50_000, window=25_000)
    assert len(starts) == 2 and int(ends[-1]) == 50_000


def test_a_chromosome_shorter_than_one_window_yields_none():
    starts, _ = data.window_bounds(24_999, window=25_000)
    assert starts.size == 0


# ── the index table ───────────────────────────────────────────────────────────

def test_index_rows_bounds_are_contiguous_and_window_wide(tmp_path):
    rows = data.index_rows("S1", "full", "chr1", 75_000,
                           np.zeros(75_000, dtype=bool), window=25_000)
    assert rows["start"].tolist() == [0, 25_000, 50_000]
    assert (rows["end"] - rows["start"] == 25_000).all()


def test_index_rows_take_the_split_from_the_chromosome(tmp_path):
    """SPLITS is the only thing that decides a split, and unlisted means train."""
    mask = np.zeros(50_000, dtype=bool)
    for chrom, expected in (("chr1", b"train"), ("chr7", b"val"), ("chr8", b"test")):
        rows = data.index_rows("S1", "full", chrom, 50_000, mask, window=25_000)
        assert set(rows["split"]) == {expected}, chrom


def test_valid_frac_is_measured_on_the_dilated_mask(tmp_path):
    """Not on the raw mask: the dataloader marks the dilated region unusable too.

    One masked base with a 120 bp dilation costs 241 positions, so the two
    measurements differ by a factor of 241 and cannot be confused for each other.
    """
    mask = np.zeros(25_000, dtype=bool)
    mask[10_000] = True
    rows = data.index_rows("S1", "full", "chr1", 25_000, mask, window=25_000)

    expected = data.dilated_valid(mask).mean()
    assert rows["valid_frac"][0] == pytest.approx(expected, abs=1e-6)
    assert rows["valid_frac"][0] < 1.0, "dilation was not applied"


def test_a_fully_masked_window_is_flagged_not_dropped(tmp_path):
    """`keep` is a flag. Nothing is ever removed from the index."""
    rows = data.index_rows("S1", "full", "chr1", 25_000,
                           np.ones(25_000, dtype=bool), window=25_000)
    assert len(rows["start"]) == 1 and not rows["keep"][0]


# ── the counts reader ─────────────────────────────────────────────────────────

def test_counts_positions_are_one_based(tmp_path):
    """Position 1 in the file is index 0 in the array."""
    path = write_counts(tmp_path, "chr1", [(1, 7, 7.5), (10, 3, 3.5)])
    counts = preprocess.load_counts(path, 20)
    assert counts.gc_corrected[0] == pytest.approx(7.5)
    assert counts.gc_corrected[9] == pytest.approx(3.5)
    assert counts.gc_corrected[1] == 0.0


def test_the_two_count_columns_are_not_transposed(tmp_path):
    """Distinct values per column: a swap is silent and changes every number."""
    path = write_counts(tmp_path, "chr1", [(5, 11, 22.0)])
    counts = preprocess.load_counts(path, 10)
    assert counts.uncorrected[4] == pytest.approx(11.0)
    assert counts.gc_corrected[4] == pytest.approx(22.0)


def test_absent_positions_are_zero_not_missing(tmp_path):
    path = write_counts(tmp_path, "chr1", [(3, 1, 1.0)])
    counts = preprocess.load_counts(path, 10)
    assert counts.gc_corrected.shape == (10,)
    assert int(np.count_nonzero(counts.gc_corrected)) == 1


def test_unsorted_positions_are_refused(tmp_path):
    """Two chromosomes concatenated restart the positions; that must not load."""
    path = write_counts(tmp_path, "chr1", [(5, 1, 1.0), (2, 1, 1.0)])
    with pytest.raises(ValueError, match="strictly increasing"):
        preprocess.load_counts(path, 10)


def test_positions_past_the_chromosome_are_refused(tmp_path):
    """The counts file and the reference disagreeing is a wrong-genome-build error."""
    path = write_counts(tmp_path, "chr1", [(50, 1, 1.0)])
    with pytest.raises(ValueError, match="coordinate system"):
        preprocess.load_counts(path, 10)


def test_a_file_with_extra_columns_is_refused(tmp_path):
    """A legacy script rewrote it in place; it is no longer raw counts."""
    path = tmp_path / "rewritten.tsv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("chr1\t1\t2\t3.0\t99\n")
    with pytest.raises(ValueError, match="columns"):
        preprocess.load_counts(path, 10)


# ── the sequence reader ───────────────────────────────────────────────────────

def test_bases_map_to_their_codes_and_soft_masking_is_kept(tmp_path):
    """hg38 is soft-masked: lowercase acgt are REAL bases, not N.

    Letting them fall through to N would zero every repeat-masked region --
    which is most of the genome's repetitive content.
    """
    fasta = write_fasta(tmp_path, {"chr1": "ACGTacgtN"})
    codes = data.load_sequence(fasta, "chr1", 9)
    assert codes.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4]


def test_a_fasta_that_disagrees_with_the_fai_is_refused(tmp_path):
    fasta = write_fasta(tmp_path, {"chr1": "ACGT"})
    with pytest.raises(ValueError, match="disagree"):
        data.load_sequence(fasta, "chr1", 8)


# ── regressions: the two defects found while wiring the counts path ───────────

def _build_two_levels(tmp_path, second_bed=None, window=25_000):
    """One sample, two levels, one chromosome. Returns the HDF5 path."""
    import h5py                                              # noqa: F401  (h5py via data)
    n = 2 * window
    fasta = write_fasta(tmp_path, {"chr1": "ACGT" * (n // 4)})
    bed = write_bed(tmp_path, [("chr1", 0, n)])
    counts = write_counts(tmp_path, "chr1", [(i, 1, 1.0) for i in range(1, n, 7)])

    attrs = {"nominal_depth": float("nan"), "measured_depth": float("nan"),
             "gc_table_md5": "", "bam_path": "", "n_fragments": -1}
    h5 = data.open_h5(tmp_path / "s.h5", reference_genome="ref.fa",
                      sample_attrs={"subtype": "", "tumour_fraction": float("nan"),
                                    "sex": "", "cohort": ""}, window=window)
    data.build_chromosome(h5, "S1", "full", "chr1", counts, fasta, bed, attrs,
                          window=window, log=lambda *_: None)
    data.build_chromosome(h5, "S1", "under", "chr1", counts, fasta,
                          second_bed or bed, attrs, window=window, log=lambda *_: None)
    h5.close()
    return tmp_path / "s.h5"


def test_a_second_level_reuses_the_shared_mask_and_sequence(tmp_path):
    """shared/ depends on the chromosome, never on the level.

    Writing it unconditionally made the SECOND level of a sample fail outright,
    which is every two-level file -- i.e. every file that can train anything.
    """
    path = _build_two_levels(tmp_path)
    import h5py
    with h5py.File(path) as h5:
        assert set(h5["levels"]) == {"full", "under"}
        assert set(h5["shared/chr1"]) == {"mask", "sequence"}


def test_a_second_level_built_against_a_different_bed_is_refused(tmp_path):
    """Silently keeping the first mask would leave it not describing the counts."""
    other = write_bed(tmp_path, [("chr1", 0, 100)], name="other.bed")
    with pytest.raises(ValueError, match="same reference genome"):
        _build_two_levels(tmp_path, second_bed=other)


def test_each_window_is_served_once_per_file_not_once_per_level(tmp_path):
    """THE SILENT ONE. /index carries a row per (level, window).

    Unfiltered, a two-level file serves every window twice: the epoch is double
    its true size and so is the reported test-set count. Nothing raises, and the
    duplicated items are identical, so no metric looks wrong either.
    """
    path = _build_two_levels(tmp_path)
    dataset = data.WindowDataset(path, "under", "full", split="train")

    assert len(dataset) == 2, "windows are duplicated once per level"
    assert [dataset.metadata(i)["start"] for i in range(len(dataset))] == [0, 25_000]


def test_an_unknown_level_is_refused_with_the_levels_that_exist(tmp_path):
    path = _build_two_levels(tmp_path)
    with pytest.raises(KeyError):
        data.WindowDataset(path, "under", "nonexistent", split="train")
