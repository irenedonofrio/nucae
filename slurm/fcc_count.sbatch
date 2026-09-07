#!/bin/bash
#SBATCH --job-name=nucae_fcc
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu
#
# >>> CHECK BEFORE SUBMITTING <<<
#   --cpus-per-task 8   passed on as --threads. 02_fcc_count.sh fans out over
#                       chromosomes with GNU parallel, so this is the number of
#                       chromosomes in flight at once, not threads within one.
#   --mem        32G    NOT MEASURED. Eight concurrent chromosome jobs, and chr1
#                       is 249 Mbp. If it OOMs, lower --cpus-per-task before
#                       raising memory: the two are coupled here.
#   --time    12:00:00  NOT MEASURED. Full-depth BAM; the downsampled arm is far
#                       quicker. Run the downsampled one first and scale.
#
# STAGE 2 counting from cfdna_tfp_profiling: GC-corrected fragment midpoints,
# SPARSE, one file per chromosome. This is the handoff into scripts/01_build_hdf5.py,
# which expects exactly {chrom}_counts.tsv.gz with four columns.
#
#   mkdir -p slurm/logs
#   sbatch slurm/fcc_count.sbatch <cfdna_tfp_profiling> <in.bam> <GC_bias.txt> \
#                                 <out_dir> <hg38.fa> <mappable.bed>
#
# The reference and the mappability BED are passed SEPARATELY rather than derived
# from one refs directory. They do not have to live together, and a script that
# assumes they do fails by looking in the wrong place rather than by saying so.
#
# EACH ARM USES ITS OWN GC BIAS TABLE. Pairing the downsampled BAM with the
# full-depth table (or the reverse) applies a correction estimated from a
# different read set: it will run, and every count will be quietly wrong.
#
# frag-min/max and mapq are left at the script's defaults (120/200/30). Those are
# the Stage-2 values the rest of this repository assumes; they are NOT the GC
# stage's 15-500/20.
#
# WRITES <out_dir>/{chrom}_counts.tsv.gz and _joblog.tsv

set -euo pipefail

REPO="${1:?usage: sbatch slurm/fcc_count.sbatch <repo> <in.bam> <gc_bias.txt> <out_dir> <hg38.fa> <mappable.bed>}"
BAM="${2:?missing bam}"
GCBIAS="${3:?missing GC_bias.txt}"
OUT="${4:?missing output dir}"
FASTA="${5:?missing hg38.fa}"
MAPPABLE="${6:?missing mappable .bed}"

for f in "${FASTA}" "${FASTA}.fai" "${MAPPABLE}"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f" >&2; exit 2; }
done

# `set -u` off just for the environment setup: /etc/profile.d scripts and conda's
# own activate hooks reference unbound variables, and with -u active that kills
# the job before any work starts. Restored immediately after -- the point of -u
# is to catch OUR unset variables, not the system's.
set +u
source ~/.bashrc
conda activate tfp_env     # 02_fcc_count.sh no longer activates conda itself
set -u

echo "job:     ${SLURM_JOB_ID}"
echo "node:    $(hostname)"
echo "start:   $(date -Is)"
echo "bam:     ${BAM}"
echo "gc-bias: ${GCBIAS}"
echo "ref:     ${FASTA}"
echo "bed:     ${MAPPABLE}"
echo "out:     ${OUT}"

bash "${REPO}/nucleosome_profiling/02_fcc_count.sh" \
    --bam      "${BAM}" \
    --gc-bias  "${GCBIAS}" \
    --out      "${OUT}" \
    --ref      "${FASTA}" \
    --mappable "${MAPPABLE}" \
    --threads  "${SLURM_CPUS_PER_TASK}"

echo
echo "-- what scripts/01_build_hdf5.py will read --"
ls "${OUT}" | head -5
echo "columns in chr21: $(zcat "${OUT}/chr21_counts.tsv.gz" | head -1 | awk -F'\t' '{print NF}')  (must be 4)"
zcat "${OUT}/chr21_counts.tsv.gz" | head -2
echo "end:     $(date -Is)"
