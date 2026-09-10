#!/bin/bash
# Submit per-gene between-lineage divergence (dXY) and Tajima's D for a finished
# pangenomerge run, and write <outdir>/gene_dxy.tsv and <outdir>/gene_tajima.tsv.
#
# Run this on a login node once the `msa` rule has completed; it submits a SLURM
# job array over chunks of genes plus a dependent collect job, then returns.
#
#   ./run_popgen.sh --results-dir /path/to/project/results
#
# Lineages come from PopPUNK: every cluster with at least --min-cluster-size
# isolates is measured, and dXY is reported for every pair of them. That is
# quadratic in the lineage count, so the banner prints the pair count and the
# projected row count before anything is submitted.

set -euo pipefail

POPGEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_DIR=""
MSA_DIR=""
CLUSTERS=""
FORCE_MSA_DIR=0
OUTDIR=""
MIN_CLUSTER_SIZE=500
MAX_CLUSTERS=40
ONLY_CLUSTERS=""
CHUNK_SIZE=250
MAX_CONCURRENT=50
MIN_OCCUPANCY=0.2
MIN_SEQS=20
MIN_SITES=90
MIN_POP_SEQS=20
BLOCK_CELLS=4000000
CPUS=1
MEM=8G
TIME=04:00:00
PARTITION=standard
SLURM_ACCOUNT=""
PYTHON="$(command -v python3 || true)"
DRY_RUN=0

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<USAGE

Options:
  --results-dir DIR       pipeline results directory (contains msa/ and
                          clusters/); the codon alignments are taken from
                          DIR/msa/codon/aligned_gene_sequences and the lineages
                          from DIR/clusters/combined_clusters.csv
  --msa-dir DIR           use this codon alignment directory instead. Refused
                          unless it sits under codon/, or --force-msa-dir is given
  --clusters FILE         PopPUNK Taxon,Cluster CSV
  --force-msa-dir         allow a --msa-dir outside codon/
  --outdir DIR            output directory (default: <results-dir>/popgen)
  --min-cluster-size N    lineages smaller than this are not measured
                          (default: $MIN_CLUSTER_SIZE)
  --only-clusters "A B"   measure exactly these lineages, ignoring
                          --min-cluster-size
  --max-clusters N        refuse to run above this many lineages; pairs grow as
                          N^2, and all 306 s_pyogenes clusters would be 46,665
                          pairs x 14,247 genes (default: $MAX_CLUSTERS)
  --chunk-size N          genes per array task (default: $CHUNK_SIZE)
  --max-concurrent N      concurrent array tasks (default: $MAX_CONCURRENT)
  --min-occupancy F       keep columns present in >= F of sequences (default: $MIN_OCCUPANCY)
  --min-seqs N            skip genes with fewer sequences (default: $MIN_SEQS)
  --min-sites N           skip genes with fewer retained NUCLEOTIDE columns
                          (default: $MIN_SITES; the pi stage's 30 aa sites x 3)
  --min-pop-seqs N        skip a lineage within a gene when it contributes
                          fewer sequences than this (default: $MIN_POP_SEQS)
  --block-cells N         alignment cells counted per numpy block (default: $BLOCK_CELLS)
  --cpus N                cpus per array task (default: $CPUS)
  --mem SIZE              memory per array task (default: $MEM)
  --time HH:MM:SS         walltime per array task (default: $TIME)
  --partition NAME        SLURM partition (default: $PARTITION)
  --slurm-account NAME    SLURM account
  --python PATH           interpreter with numpy (default: $PYTHON)
  --dry-run               write the inputs and print the sbatch commands only
USAGE
}

die() { echo "error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --results-dir)      RESULTS_DIR="$2"; shift 2 ;;
        --msa-dir)          MSA_DIR="$2"; shift 2 ;;
        --clusters)         CLUSTERS="$2"; shift 2 ;;
        --force-msa-dir)    FORCE_MSA_DIR=1; shift ;;
        --outdir)           OUTDIR="$2"; shift 2 ;;
        --min-cluster-size) MIN_CLUSTER_SIZE="$2"; shift 2 ;;
        --only-clusters)    ONLY_CLUSTERS="$2"; shift 2 ;;
        --max-clusters)     MAX_CLUSTERS="$2"; shift 2 ;;
        --chunk-size)       CHUNK_SIZE="$2"; shift 2 ;;
        --max-concurrent)   MAX_CONCURRENT="$2"; shift 2 ;;
        --min-occupancy)    MIN_OCCUPANCY="$2"; shift 2 ;;
        --min-seqs)         MIN_SEQS="$2"; shift 2 ;;
        --min-sites)        MIN_SITES="$2"; shift 2 ;;
        --min-pop-seqs)     MIN_POP_SEQS="$2"; shift 2 ;;
        --block-cells)      BLOCK_CELLS="$2"; shift 2 ;;
        --cpus)             CPUS="$2"; shift 2 ;;
        --mem)              MEM="$2"; shift 2 ;;
        --time)             TIME="$2"; shift 2 ;;
        --partition)        PARTITION="$2"; shift 2 ;;
        --slurm-account)    SLURM_ACCOUNT="$2"; shift 2 ;;
        --python)           PYTHON="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)                  usage >&2; die "unknown argument: $1" ;;
    esac
done

### resolve the alignment directory

# Same choice the pi stage makes, and the opposite of dN/dS: these are diversity
# measures, so they want every isolate, including the ones --strict-codons drops
# for failing translation QC. N, degenerate bases and part-gapped codons are all
# just missing data here.
if [ -n "$MSA_DIR" ]; then
    case "$(basename "$(dirname "$MSA_DIR")")" in
        codon|codons) ;;
        *)
            [ "$FORCE_MSA_DIR" -eq 1 ] || die \
"--msa-dir is not under codon/: $MSA_DIR
       This stage wants the non-strict --codons alignments, which keep the
       sequences codon_strict/ drops. Pass --force-msa-dir if you are certain
       this is what you want."
            ;;
    esac
else
    [ -n "$RESULTS_DIR" ] || { usage >&2; die "--results-dir or --msa-dir is required"; }
    MSA_DIR="$RESULTS_DIR/msa/codon/aligned_gene_sequences"
fi
[ -d "$MSA_DIR" ] || die "codon alignment directory not found: $MSA_DIR"

### resolve the lineage assignments

# The metadata sqlite has an isolate_names.poppunk_cluster column, but it is
# NULL for every row in every species built so far: add_clusters_to_sqlite()
# matches the CSV's suffix-less Taxon against sample_name, which carries '.fa',
# so no row ever updates. Read the CSV instead.
if [ -z "$CLUSTERS" ]; then
    [ -n "$RESULTS_DIR" ] || die "--clusters is required when --msa-dir is used"
    CLUSTERS="$RESULTS_DIR/clusters/combined_clusters.csv"
fi
[ -f "$CLUSTERS" ] || die "cluster assignments not found: $CLUSTERS"
head -1 "$CLUSTERS" | grep -q '^Taxon,Cluster' \
    || die "$CLUSTERS does not start with the expected 'Taxon,Cluster' header"

if [ -z "$OUTDIR" ]; then
    [ -n "$RESULTS_DIR" ] || die "--outdir is required when --msa-dir is used"
    OUTDIR="$RESULTS_DIR/popgen"
fi

[ -x "$PYTHON" ] || die "python not executable: $PYTHON"
"$PYTHON" -c "import numpy" 2>/dev/null || die "$PYTHON cannot import numpy"

### which lineages, and how many pairs

read -r NPOPS PAIRS POPLIST <<<"$("$PYTHON" - "$CLUSTERS" "$MIN_CLUSTER_SIZE" "$ONLY_CLUSTERS" <<'PYEOF'
import csv, sys
path, min_size, only = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
sizes = {}
with open(path, newline="") as fh:
    for row in csv.DictReader(fh):
        sizes[row["Cluster"]] = sizes.get(row["Cluster"], 0) + 1
if only:
    keep = [c for c in only if c in sizes]
else:
    keep = [c for c, n in sizes.items() if c != "novel" and n >= min_size]
keep.sort(key=lambda c: (-sizes[c], c))
n = len(keep)
print(n, n * (n - 1) // 2,
      ",".join(f"{c}:{sizes[c]}" for c in keep[:8]) + (",..." if n > 8 else ""))
PYEOF
)"
[ "$NPOPS" -ge 2 ] || die \
"fewer than two lineages selected from $CLUSTERS
       --min-cluster-size $MIN_CLUSTER_SIZE is too high, or --only-clusters
       named clusters that do not exist."

[ "$NPOPS" -le "$MAX_CLUSTERS" ] || die \
"$NPOPS lineages selected, above --max-clusters $MAX_CLUSTERS.
       That is $PAIRS pairs per gene. Raise --min-cluster-size, name lineages
       with --only-clusters, or raise --max-clusters deliberately."

### build the gene list

mkdir -p "$OUTDIR" "$OUTDIR/logs" "$OUTDIR/stats" "$OUTDIR/dxy" "$OUTDIR/tajima"
ALIGNMENTS="$OUTDIR/alignments.tsv"

# node name is the alignment basename, per get_alignment_basename() in
# pangenomerge/alignment_functions/generate_alignments.py.
# -L, as in the pi stage: a symlinked alignment (or a symlinked directory of
# them) is a normal way to assemble an ad-hoc gene set, and -type f alone would
# silently skip every one of them.
find -L "$MSA_DIR" -maxdepth 1 -name '*.aln.fas' -type f -size +0 -print \
    | sort \
    | awk -F/ '{name=$NF; sub(/\.aln\.fas$/, "", name); print name "\t" $0}' \
    > "$ALIGNMENTS"

NGENES=$(wc -l < "$ALIGNMENTS")
[ "$NGENES" -gt 0 ] || die "no non-empty *.aln.fas files in $MSA_DIR"
NCHUNKS=$(( (NGENES + CHUNK_SIZE - 1) / CHUNK_SIZE ))

### record every setting, so the array (and any resubmit) is reproducible

PARAMS="$OUTDIR/popgen_params.sh"
cat > "$PARAMS" <<PARAMEOF
# written by run_popgen.sh on $(date -Is)
# source of codon alignments: $MSA_DIR
POPGEN_DIR=$(printf '%q' "$POPGEN_DIR")
POPGEN_OUTDIR=$(printf '%q' "$OUTDIR")
POPGEN_PYTHON=$(printf '%q' "$PYTHON")
POPGEN_CLUSTERS=$(printf '%q' "$CLUSTERS")
POPGEN_MIN_CLUSTER_SIZE=$(printf '%q' "$MIN_CLUSTER_SIZE")
POPGEN_ONLY_CLUSTERS=$(printf '%q' "$ONLY_CLUSTERS")
POPGEN_CHUNK_SIZE=$(printf '%q' "$CHUNK_SIZE")
POPGEN_MIN_OCCUPANCY=$(printf '%q' "$MIN_OCCUPANCY")
POPGEN_MIN_SEQS=$(printf '%q' "$MIN_SEQS")
POPGEN_MIN_SITES=$(printf '%q' "$MIN_SITES")
POPGEN_MIN_POP_SEQS=$(printf '%q' "$MIN_POP_SEQS")
POPGEN_BLOCK_CELLS=$(printf '%q' "$BLOCK_CELLS")
PARAMEOF

echo "codon alns  : $MSA_DIR"
echo "lineages    : $CLUSTERS"
echo "selected    : $NPOPS lineages, $PAIRS pairs  [$POPLIST]"
echo "genes       : $NGENES"
echo "dxy rows    : up to $((NGENES * PAIRS))"
echo "outdir      : $OUTDIR"
echo "array       : 0-$((NCHUNKS - 1))%$MAX_CONCURRENT ($CHUNK_SIZE genes per task)"
echo "resources   : $CPUS cpus, $MEM, $TIME, partition $PARTITION"
echo "python      : $PYTHON"

### submit

acct_args=()
[ -n "$SLURM_ACCOUNT" ] && acct_args=(--account "$SLURM_ACCOUNT")

array_args=(
    --array="0-$((NCHUNKS - 1))%$MAX_CONCURRENT"
    --job-name=popgen
    --partition="$PARTITION"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME"
    --output="$OUTDIR/logs/popgen_%A_%a.out"
    --error="$OUTDIR/logs/popgen_%A_%a.err"
    "${acct_args[@]}"
    --parsable
    "$POPGEN_DIR/popgen_array.sh" "$PARAMS"
)

collect_cmd="$(printf '%q ' "$PYTHON" "$POPGEN_DIR/collect_popgen.py" --outdir "$OUTDIR")"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "dry run, would submit:"
    printf '  sbatch'; printf ' %q' "${array_args[@]}"; printf '\n'
    printf '  sbatch --dependency=afterany:<jobid> ... --wrap %q\n' "$collect_cmd"
    exit 0
fi

ARRAY_JOB=$(sbatch "${array_args[@]}")

# afterany, not afterok: genes that fail are meant to land in the tables as
# blanks, so they should still be written when some chunks die
COLLECT_JOB=$(sbatch --parsable \
    --dependency=afterany:"$ARRAY_JOB" \
    --job-name=popgen_collect \
    --partition="$PARTITION" \
    --cpus-per-task=1 \
    --mem=16G \
    --time=02:00:00 \
    --output="$OUTDIR/logs/popgen_collect_%j.out" \
    --error="$OUTDIR/logs/popgen_collect_%j.err" \
    "${acct_args[@]}" \
    --wrap "$collect_cmd")

echo
echo "submitted array   : $ARRAY_JOB"
echo "submitted collect : $COLLECT_JOB (afterany:$ARRAY_JOB)"
echo "between lineages  : $OUTDIR/gene_dxy.tsv"
echo "within lineages   : $OUTDIR/gene_tajima.tsv"
echo "per-gene detail   : $OUTDIR/popgen_stats.tsv"
