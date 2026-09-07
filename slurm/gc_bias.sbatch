#!/bin/bash
#SBATCH --job-name=nucae_gcbias
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu
#
# >>> CHECK BEFORE SUBMITTING <<<
#   --cpus-per-task 8   MUST MATCH `threads:` in the gc config. The driver reads
#                       its thread count from the YAML, not from Slurm, so a
#                       mismatch either wastes an allocation or oversubscribes.
#   --mem        16G    config_gc.yaml: "prefix-sum peaks ~3 GB at 8 threads;
#                       request --mem>=8G". Doubled, because that note predates
#                       this sample.
#   --time    08:00:00  NOT MEASURED. Samples run SEQUENTIALLY inside the driver,
#                       so a two-row sample sheet is two full passes, not two
#                       parallel ones. Budget accordingly.
#
# STAGE 1 of cfdna_tfp_profiling: GC bias tables. This wrapper only supplies the
# cluster resources; the pipeline itself lives in that repository and is not
# copied here, so the two cannot drift apart.
#
#   mkdir -p slurm/logs
#   sbatch slurm/gc_bias.sbatch <cfdna_tfp_profiling dir> <config_gc.yaml> <results_root>
#
# ONE TABLE PER ARM, and they are not interchangeable: the full-depth and the
# downsampled BAM are separate rows in the sample sheet and get separate tables.
# Uniform downsampling does not change a library's true GC bias, only how
# precisely it can be estimated, so the two tables SHOULD agree closely --
# compare them rather than assuming it. A large disagreement means the low-depth
# arm carries correction noise the model will have to learn to undo.
#
# DO NOT edit map_q or size_range in the config. They are 20 and [15, 500] here,
# deliberately NOT the 30 and [120, 200] the FCC step uses, and the config says
# changing them breaks byte-identity with the validated output.
#
# WRITES <results_root>/{GC_counts,GC_bias}/ and gc_correction_manifest.tsv

set -euo pipefail

REPO="${1:?usage: sbatch slurm/gc_bias.sbatch <cfdna_tfp_profiling> <config.yaml> <results_root>}"
CONFIG="${2:?missing gc config}"
RESULTS="${3:?missing results root}"

# `set -u` off just for the environment setup: /etc/profile.d scripts and conda's
# own activate hooks reference unbound variables, and with -u active that kills
# the job before any work starts. Restored immediately after -- the point of -u
# is to catch OUR unset variables, not the system's.
set +u
source ~/.bashrc
conda activate griffin     # run_gc_correction.sh activates NOTHING; that is the caller's job
set -u

echo "job:    ${SLURM_JOB_ID}"
echo "node:   $(hostname)"
echo "start:  $(date -Is)"
echo "config: ${CONFIG}"
echo "out:    ${RESULTS}"
echo
grep -E "^\s+(map_q|size_range|threads):" "${CONFIG}" || true
echo

mkdir -p "${RESULTS}"
bash "${REPO}/gc_correction/run_gc_correction.sh" \
     -c "${CONFIG}" --results-root "${RESULTS}"

echo
echo "-- tables written --"
ls -la "${RESULTS}/GC_bias/"
echo "end:    $(date -Is)"
