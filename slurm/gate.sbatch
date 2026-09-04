#!/bin/bash
#SBATCH --job-name=nucae_gate
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
# >>> CHECK BEFORE SUBMITTING <<<
#   --partition  gpu       (as given; confirm with `sinfo -o "%P %l %G"`)
#   --time       08:00:00  NOT MEASURED. Inference is one forward pass per window,
#                          but the bedGraph export writes 25,000 lines per window
#                          PER FILE, twice, and that dominates. Run the --limit 50
#                          smoke test first and extrapolate before trusting this.
#   --mem        16G       one window at a time, batch size 1; the model is 8.5M
#                          parameters. Generous on purpose, it is not the constraint.
#
# THE REPRODUCTION GATE. Re-scores an existing checkpoint on the test split so the
# result can be compared against the test_summary.txt saved when it was trained.
# Nothing new should be trained until this reproduces.
#
#   mkdir -p slurm/logs
#   sbatch slurm/gate.sbatch <cfdna.h5> <checkpoint.ckpt> <out_dir> [--dual-stream]
#
# THE HDF5 MUST BE THE ONE THAT CHECKPOINT WAS TRAINED ON. A model trained on
# cfdna_fullgenome.h5 scored against cfdna_chr89.h5 is a different window set and
# the numbers will not correspond to the saved summary. Check which file the old
# summary came from before comparing against it.
#
# Pass --dual-stream for a V3 checkpoint: the module that wrote it had no such
# hyperparameter, so it cannot be recovered from the file.
#
# READS   the HDF5 (test split only) and the checkpoint. Neither is modified.
# WRITES  <out_dir>/{test_summary.txt, test_signal_metrics.csv,
#         test_peak_distances.csv, predictions.tsv.gz, ground_truth.tsv.gz}
#
# IF IT DIES HALFWAY: NOT RESUMABLE, and it starts over. The script deletes the
# two .tsv.gz files at startup so a rerun cannot append to a previous run's
# output. Nothing is corrupted by a kill; you simply lose the elapsed time.

set -euo pipefail          # fail loudly and early. A silent partial failure on a
                           # cluster means finding out three days later.

HDF5="${1:?usage: sbatch slurm/gate.sbatch <cfdna.h5> <checkpoint.ckpt> <out_dir> [--dual-stream]}"
CKPT="${2:?missing checkpoint}"
OUT="${3:?missing output directory}"
shift 3

source ~/.bashrc
conda activate ae_env      # NEVER base; `nucae` is the MacBook name

echo "job:   ${SLURM_JOB_ID}"
echo "node:  $(hostname)"
echo "start: $(date -Is)"
echo "hdf5:  ${HDF5}"
echo "ckpt:  ${CKPT}"

# ---- work ----
python scripts/04_evaluate_reconstruction.py \
    --data "${HDF5}" \
    --ckpt "${CKPT}" \
    --out "${OUT}" \
    "$@"

echo "end:   $(date -Is)"
echo
echo "Compare ${OUT}/test_summary.txt against the summary saved with the checkpoint:"
echo "  mean Pearson (recon), mean Pearson (input), median peak distance, % beats input"
