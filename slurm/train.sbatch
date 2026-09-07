#!/bin/bash
#SBATCH --job-name=nucae_train
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
# >>> CHECK BEFORE SUBMITTING <<<
#   --partition  gpu       (as given; confirm with `sinfo -o "%P %l %G"`)
#   --time       24:00:00  The thesis records ~3 min/epoch, so 40 epochs is ~2h --
#                          but that figure came from the SMALL chr8/9 dataset. On
#                          the full genome it will be longer, by an unmeasured
#                          factor. Read the first epoch's time in the log and
#                          resubmit with a real number if it is far off.
#   --cpus       4         matches --workers 4 below; one worker per core.
#   --mem        32G       from the thesis's SLURM header. Not measured here.
#
# DO NOT SUBMIT THIS UNTIL slurm/gate.sbatch HAS REPRODUCED THE OLD NUMBERS.
# A new training result is uninterpretable while the port is unverified.
#
#   mkdir -p slurm/logs
#   sbatch slurm/train.sbatch <cfdna.h5> <out_dir> [--no-sequence|--dual-stream|...]
#
# Variants: no flag is V1 / V2-with-sequence, --no-sequence is the V2 ablation,
# --dual-stream is V3. Everything else defaults to the recipe that ran: MSE only,
# AdamW 1e-3, weight decay 1e-2, ReduceLROnPlateau, gradient clipping 1.0.
#
# READS   the HDF5 (train and val splits).
# WRITES  <out_dir>/run_config.json         every argument plus library versions
#         <out_dir>/checkpoints/            top 3 by val_loss, plus last.ckpt
#         <out_dir>/logs/                   CSV metrics per step and per epoch
#
# IF IT DIES HALFWAY: RESUMABLE. ModelCheckpoint writes last.ckpt every epoch and
# --resume restores optimiser and scheduler state, not just weights. Resubmit as
#
#   sbatch slurm/train.sbatch <cfdna.h5> <out_dir> --resume <out_dir>/checkpoints/last.ckpt
#
# The run then continues to the ORIGINAL --epochs total; it does not add 40 more.
# A job killed at the walltime loses at most one epoch.
#
# WATCH grad_norm IN THE CSV LOG. It is recorded BEFORE clipping. Well below 1.0
# means the clip never fires and is free insurance. Near or above 1.0 means it is
# firing most steps and acting as a silent learning-rate cut -- which would not
# show up anywhere in the LR logs.

set -euo pipefail          # fail loudly and early. A silent partial failure on a
                           # cluster means finding out three days later.

HDF5="${1:?usage: sbatch slurm/train.sbatch <cfdna.h5> <out_dir> [extra args]}"
OUT="${2:?missing output directory}"
shift 2

# `set -u` off just for the environment setup: /etc/profile.d scripts and conda's
# own activate hooks reference unbound variables, and with -u active that kills
# the job before any work starts. Restored immediately after -- the point of -u
# is to catch OUR unset variables, not the system's.
set +u
source ~/.bashrc
conda activate ae_env      # NEVER base; `nucae` is the MacBook name
set -u

# ONE THREAD PER WORKER. The counts-format dataloader smooths two windows per
# item, and numpy/scipy would each take every core they can see: 8 workers x 8
# threads on 8 CPUs is 64-way contention, and the workers spend their time
# descheduling each other. The parallelism that matters here is ACROSS workers,
# not inside one. Harmless for the pre-windowed format, which does no smoothing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "job:   ${SLURM_JOB_ID}"
echo "node:  $(hostname)"
echo "gpu:   ${CUDA_VISIBLE_DEVICES:-none}"
echo "start: $(date -Is)"
echo "hdf5:  ${HDF5}"
echo "out:   ${OUT}"

# ---- work ----
# --workers matches --cpus-per-task. --seed is written to run_config.json; change
# it per run rather than editing this file, so the file stays the record of one recipe.
python scripts/03_train.py \
    --data "${HDF5}" \
    --out "${OUT}" \
    --epochs 40 \
    --batch-size 16 \
    --workers 4 \
    --seed 42 \
    "$@"

echo "end:   $(date -Is)"
