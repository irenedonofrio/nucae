#!/bin/bash
#SBATCH --job-name=nucae_coverage
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=8G
#SBATCH --partition=cpu
#
# >>> CHECK BEFORE SUBMITTING <<<
#   --cpus-per-task 11  one `samtools coverage` per chromosome, 11 at a time, so
#                       22 autosomes are two passes. 22 would be one pass; 11 is
#                       the cheaper thing to get scheduled. Any value works.
#   --time    02:00:00  NOT MEASURED. Each job seeks with the .bai and reads only
#                       its own chromosome, so this is roughly one pass over the
#                       BAM divided by the CPU count. A deep WGS BAM is the case
#                       to watch.
#   --mem         8G    samtools streams a region at a time.
#
# Mean depth of a BAM, and the samtools fraction that reaches a target depth.
#
#   mkdir -p slurm/logs
#   sbatch slurm/coverage.sbatch <in.bam> <target_depth> [seed]
#
# WHY THIS IS A JOB AND NOT A ONE-LINER: `samtools coverage` with no region
# streams the whole file on one core. Split by chromosome it seeks with the
# index, so the work divides cleanly across CPUs.
#
# The mean is weighted BY CHROMOSOME LENGTH, not a plain average of the 22
# per-chromosome depths. An unweighted mean gives chr21 the same say as chr1,
# which is nearly six times its size, and reports a depth the library does not
# have.
#
# AUTOSOMES ONLY. chrX and chrY depth depends on the donor's sex, and chrM sits
# orders of magnitude above everything else; either would drag the figure that
# sets the downsampling fraction. The model trains on autosomes regardless.
#
# READS the BAM and its index. Writes nothing but the log.

set -euo pipefail

BAM="${1:?usage: sbatch slurm/coverage.sbatch <in.bam> <target_depth> [seed]}"
TARGET="${2:?missing target depth, e.g. 5}"
SEED="${3:-42}"

[ -f "${BAM}.bai" ] || [ -f "${BAM%.bam}.bai" ] || {
    echo "ERROR: no .bai beside ${BAM}. Per-chromosome seeking needs the index;" >&2
    echo "       without it every job would stream the whole file." >&2
    exit 2; }

source ~/.bashrc
conda activate ae_env      # samtools; NEVER base

echo "job:    ${SLURM_JOB_ID}"
echo "node:   $(hostname)"
echo "start:  $(date -Is)"
echo "bam:    ${BAM}  ($(du -h "${BAM}" | cut -f1))"
echo "target: ${TARGET}x"
echo

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

# One file per chromosome rather than one shared stream: concurrent appends from
# 11 processes can interleave mid-line, and a corrupted row here would silently
# skew the depth that sets the downsampling fraction.
printf 'chr%s\n' {1..22} \
  | xargs -P "${SLURM_CPUS_PER_TASK}" -I{} \
      sh -c 'samtools coverage -r "$1" "$2" | grep -v "^#" > "$3/$1.cov"' _ {} "${BAM}" "${work}"

cat "${work}"/chr*.cov > "${work}/all.cov"
n=$(wc -l < "${work}/all.cov")
[ "${n}" -eq 22 ] || { echo "ERROR: ${n} chromosomes reported, expected 22" >&2; exit 3; }

echo "-- per chromosome --"
printf '%-8s %14s %10s\n' chrom length meandepth
sort -V "${work}/all.cov" | awk '{printf "%-8s %14d %10.2f\n", $1, $3, $7}'
echo

awk -v target="${TARGET}" -v seed="${SEED}" '
    { depth += $7 * $3; length_total += $3 }
    END {
        mean = depth / length_total
        fraction = target / mean
        printf "mean depth (length-weighted) : %.3fx\n", mean
        printf "target depth                 : %.3fx\n", target
        if (fraction >= 1) {
            printf "\nfraction would be %.4f: the BAM is ALREADY at or below the\n", fraction
            printf "target. There is nothing to downsample.\n"
            exit 0
        }
        printf "fraction to keep             : %.4f\n", fraction
        # Built digit by digit, NOT as seed followed by %.4f: that prints the
        # fractions own leading "0" and turns seed 42 into 420. samtools would
        # accept it and downsample correctly under a seed nobody recorded.
        printf "\npass this to downsample.sbatch (seed %d):\n\n    %d.%04d\n\n", \
               seed, seed, int(fraction * 10000 + 0.5)
        printf "Subsampling is random, so the result lands NEAR the target, not on\n"
        printf "it. downsample.sbatch reports what you actually got -- that measured\n"
        printf "number is the one for the methods section, not this one.\n"
    }' "${work}/all.cov"

echo "end:    $(date -Is)"
