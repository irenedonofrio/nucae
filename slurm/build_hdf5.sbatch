#!/bin/bash
#SBATCH --job-name=nucae_build
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --partition=cpu
#
# >>> CHECK BEFORE SUBMITTING <<<
#   --mem        48G       load_counts reads a whole chromosome's TSV AT ONCE, by
#                          PLAN Task 2. chr1 at full depth is the case that
#                          decides this number: ~249 M positions, most of them
#                          covered, held as a DataFrame and then two float32
#                          arrays. Peak is chr1 and nothing else comes close.
#   --cpus-per-task 2      SINGLE THREADED. pandas, pysam and h5py here are not
#                          parallel and asking for more CPUs would not make this
#                          faster. The parallelism in this pipeline was in the
#                          FCC step, upstream.
#   --time    24:00:00     NOT MEASURED, and this one is a real guess: 22
#                          chromosomes x 2 levels = 44 passes, each reading a
#                          large gzipped TSV. Run one chromosome first and
#                          multiply -- chr1 is ~5.4% of the genome by length.
#
# Both coverage levels of one sample into ONE HDF5. Both, because WindowDataset
# reads its input and its target from the same file; two files cannot be paired.
#
#   mkdir -p slurm/logs
#   sbatch slurm/build_hdf5.sbatch <sample_id> <fcc_full_dir> <fcc_under_dir> \
#                                  <out.h5> <hg38.fa> <mappable.bed> \
#                                  [full_depth] [under_depth]
#
# THE SAME hg38.fa AND THE SAME .bed THE FCC STEP USED. The counts carry 1-based
# positions checked against the .fai length, so a different build fails loudly --
# but a different mappability BED does not, and would leave a mask that does not
# describe the counts beside it.
#
# RESUMABLE, unlike the gate. Levels are built in order and each chromosome is
# skipped if it is already in the file, so a job killed at the walltime is
# resubmitted with the same command and continues where it stopped.
#
# THAT SAFETY HAS A SHARP EDGE: --skip-existing cannot tell "already built" from
# "built earlier from a different BED". If any input changed, DELETE THE FILE and
# start again. Resuming onto a stale file would report success and leave arrays
# that do not describe each other.
#
# WRITES <out.h5>: /levels/{full,under}/{chrom}/counts_gc, /shared/{chrom}/{mask,
# sequence}, /index. Roughly 29 GB for two levels genome-wide, uncompressed.
# CHECK THE QUOTA FIRST -- running out at chr19 wastes the whole job.

set -euo pipefail

SAMPLE="${1:?usage: sbatch slurm/build_hdf5.sbatch <sample> <fcc_full> <fcc_under> <out.h5> <hg38.fa> <mappable.bed> [full_x] [under_x]}"
FCC_FULL="${2:?missing full-depth counts dir}"
FCC_UNDER="${3:?missing undersampled counts dir}"
OUT="${4:?missing output .h5}"
FASTA="${5:?missing hg38.fa}"
MAPPABLE="${6:?missing mappable .bed}"
DEPTH_FULL="${7:-nan}"
DEPTH_UNDER="${8:-nan}"

for f in "${FASTA}" "${FASTA}.fai" "${MAPPABLE}"; do
    [ -f "$f" ] || { echo "ERROR: missing reference input: $f" >&2; exit 2; }
done
# Fail now on a missing counts file rather than at chr22 after twenty hours.
for chrom in chr{1..22}; do
    for dir in "${FCC_FULL}" "${FCC_UNDER}"; do
        [ -f "${dir}/${chrom}_counts.tsv.gz" ] || {
            echo "ERROR: missing ${dir}/${chrom}_counts.tsv.gz" >&2
            echo "       The FCC step did not finish, or wrote a different name." >&2
            exit 2; }
    done
done

source ~/.bashrc
conda activate ae_env      # NEVER base; `nucae` is the MacBook name

echo "job:    ${SLURM_JOB_ID}"
echo "node:   $(hostname)"
echo "start:  $(date -Is)"
echo "sample: ${SAMPLE}"
echo "out:    ${OUT}"
echo "ref:    ${FASTA}"
echo "bed:    ${MAPPABLE}"
echo "depths: full=${DEPTH_FULL}x  under=${DEPTH_UNDER}x   <- MEASURED, not targeted"
echo "space:  $(df -h "$(dirname "${OUT}")" | tail -1)"
echo

# Two plain calls rather than a loop over packed strings: the levels differ in
# their counts directory and their measured depth, and nothing is shared but the
# references. --skip-existing makes a resubmission continue rather than restart.
echo "== level full =="
python scripts/01_build_hdf5.py \
    --sample "${SAMPLE}" --level full \
    --counts-dir "${FCC_FULL}" \
    --fasta "${FASTA}" --mappable "${MAPPABLE}" \
    --measured-depth "${DEPTH_FULL}" \
    --out "${OUT}" --skip-existing
echo

echo "== level under =="
python scripts/01_build_hdf5.py \
    --sample "${SAMPLE}" --level under \
    --counts-dir "${FCC_UNDER}" \
    --fasta "${FASTA}" --mappable "${MAPPABLE}" \
    --measured-depth "${DEPTH_UNDER}" \
    --out "${OUT}" --skip-existing
echo

echo "-- what came out --"
python - "${OUT}" <<'PY'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1]) as h5:
    print("levels    :", sorted(h5["levels"]))
    print("chromosomes:", len(h5["shared"]))
    idx = h5["index"]
    for level in sorted(h5["levels"]):
        rows = idx["level"][:] == level.encode()
        splits = {s.decode(): int(((idx["split"][:] == s) & rows).sum())
                  for s in np.unique(idx["split"][:])}
        keep = int((idx["keep"][:] & rows).sum())
        print(f"  {level:6s}: {int(rows.sum()):,} windows  {splits}  keep={keep:,}")
PY
echo "end:    $(date -Is)"
