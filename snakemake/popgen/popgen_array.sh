#!/bin/bash
#SBATCH --job-name=popgen
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# One array task = one chunk of genes, measured serially by gene_popgen.py.
#
# Sizing follows the pi stage, which measures ~1.5 s for its largest gene and is
# essentially all I/O. This stage reads the same files once and adds arithmetic
# on a per-population count tensor that is small next to the read: 21 lineages x
# the widest gene x 5 states is ~25 MB, and each lineage pair costs one pass over
# a few hundred kept columns. Budget ~3-5 s for the largest genes, so chunks the
# same size as pi's with the same resources.
#
# Peak memory is set by --block-cells and the population count, not by gene
# length: 4e6 cells is ~32 MB of int64 index, so 4G is mostly headroom for the
# file buffer. All of these are overridden from the sbatch command line in
# run_popgen.sh.
#
# Usage: sbatch --array=0-N%C popgen_array.sh <outdir>/popgen_params.sh

set -euo pipefail

PARAMS="${1:?usage: popgen_array.sh <popgen_params.sh>}"
# shellcheck source=/dev/null
source "$PARAMS"

echo "[popgen] task ${SLURM_ARRAY_TASK_ID} start: $(date -Is) on $(hostname)"
echo "[popgen] params: $PARAMS"

only_args=()
if [ -n "${POPGEN_ONLY_CLUSTERS:-}" ]; then
    # shellcheck disable=SC2206
    only_args=(--only-clusters ${POPGEN_ONLY_CLUSTERS})
fi

# -u: keep per-gene progress visible in the log while the task is still
# running, instead of block-buffered until it exits
exec "$POPGEN_PYTHON" -u "$POPGEN_DIR/gene_popgen.py" \
    --alignments "$POPGEN_OUTDIR/alignments.tsv" \
    --outdir "$POPGEN_OUTDIR" \
    --chunk-index "$SLURM_ARRAY_TASK_ID" \
    --chunk-size "$POPGEN_CHUNK_SIZE" \
    --clusters "$POPGEN_CLUSTERS" \
    --min-cluster-size "$POPGEN_MIN_CLUSTER_SIZE" \
    --min-occupancy "$POPGEN_MIN_OCCUPANCY" \
    --min-seqs "$POPGEN_MIN_SEQS" \
    --min-sites "$POPGEN_MIN_SITES" \
    --min-pop-seqs "$POPGEN_MIN_POP_SEQS" \
    --block-cells "$POPGEN_BLOCK_CELLS" \
    "${only_args[@]}"
