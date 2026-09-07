#!/bin/bash
#SBATCH --job-name=nucae_downsample
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=cpu
#
# >>> CHECK BEFORE SUBMITTING <<<
#   --partition  cpu       no GPU is used here; confirm the name with `sinfo`
#   --time       04:00:00  NOT MEASURED. Decompress + recompress of the whole BAM,
#                          I/O bound rather than CPU bound. Scale from `ls -lh`:
#                          roughly 10 min per 10 GB at 8 threads, then double it.
#   --mem        8G        samtools streams; memory is not the constraint.
#
# Uniform read-level downsampling, the INPUT arm of the reconstruction task.
#
#   mkdir -p slurm/logs
#   sbatch slurm/downsample.sbatch <in.bam> <out.bam> <SEED.FRACTION>
#
# SEED.FRACTION is ONE number: the integer part is the seed, the decimal part is
# the fraction kept. `42.10` is seed 42, keep 10%. BOTH HALVES ARE THE EXPERIMENT
# -- the fraction sets the depth being simulated and the seed is the only thing
# that makes this subsample reproducible. Record the whole string.
#
# -s subsamples by READ NAME, so both mates of a pair are kept or dropped
# together. That is not a nicety: the FCC step counts fragment midpoints, and a
# half-pair has no midpoint.
#
# WRITES <out.bam> and <out.bam>.bai. Reads the input; never modifies it.

set -euo pipefail

IN="${1:?usage: sbatch slurm/downsample.sbatch <in.bam> <out.bam> <SEED.FRACTION>}"
OUT="${2:?missing output bam}"
FRAC="${3:?missing SEED.FRACTION, e.g. 42.10}"

case "$FRAC" in
    *.*) : ;;
    *)   echo "ERROR: '${FRAC}' has no decimal part. Pass SEED.FRACTION as one" >&2
         echo "       number, e.g. 42.10 = seed 42, keep 10%. A bare integer" >&2
         echo "       would be read by samtools as a seed with fraction 0." >&2
         exit 2 ;;
esac

# `set -u` off just for the environment setup: /etc/profile.d scripts and conda's
# own activate hooks reference unbound variables, and with -u active that kills
# the job before any work starts. Restored immediately after -- the point of -u
# is to catch OUR unset variables, not the system's.
set +u
source ~/.bashrc
conda activate ae_env      # samtools; NEVER base
set -u

echo "job:   ${SLURM_JOB_ID}"
echo "node:  $(hostname)"
echo "start: $(date -Is)"
echo "in:    ${IN}  ($(du -h "${IN}" | cut -f1))"
echo "out:   ${OUT}"
echo "-s:    ${FRAC}   <- seed.fraction, RECORD THIS"

samtools view -@ "${SLURM_CPUS_PER_TASK}" -b -s "${FRAC}" "${IN}" > "${OUT}"
samtools index -@ "${SLURM_CPUS_PER_TASK}" "${OUT}"

echo
echo "-- depth check: what you actually got --"
samtools coverage "${OUT}" | awk 'NR==1 || $1 ~ /^chr[0-9]+$/ {print}' | head -25
echo "end:   $(date -Is)"
