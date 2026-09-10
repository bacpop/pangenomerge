#!/usr/bin/env python3
"""Between-lineage divergence (dXY) and Tajima's D for one chunk of genes.

Run by popgen_array.sh, once per SLURM array task. Each task takes a contiguous
slice of the alignment list written by run_popgen.sh and, for every gene in it:

  1. streams the alignment once, accumulating per-column state counts SEPARATELY
     for each selected PopPUNK lineage,
  2. drops columns occupied by fewer than --min-occupancy of the sequences,
  3. writes dXY, pi, d_a and Hudson's F_ST for every pair of lineages, and
     Tajima's D within each lineage and over the whole sample.

Why the counts are enough: pi and dXY are both functions of per-column allele
frequencies, so neither needs the pairwise comparisons their definitions imply.
A 46k-sequence gene has ~10^9 pairs within one lineage alone; the frequency form
is algebraically identical when data are complete and is what the pi stage
already relies on.

Lineages come from the PopPUNK CSV, not from the metadata database: the sqlite
isolate_names.poppunk_cluster column is NULL for every row in every species
(add_clusters_to_sqlite() matches the CSV's suffix-less Taxon against
sample_name, which carries '.fa', so no row ever updates). The alignment headers
are ">SAMPLE.fa;<seqid>", so the join key is the header sample with '.fa'
stripped.

An isolate carrying two copies of a gene contributes two records to the
alignment. Population-genetic statistics assume one allele per individual, so by
default only the first copy per isolate is counted (--paralogs first); the
alternative is to count every record (--paralogs all). This matters more than
its rarity suggests: in group_10045 two extra copies out of 7,786 raise the
lineage's pi by 8%, because a divergent paralog adds pairwise differences at
columns that are otherwise invariant.

Missing data are handled by pairwise deletion, as in the pi stage: every column
carries its own sample size n_j. That is exact for pi and dXY, but Tajima's
variance is defined for a single n - see tajima() for the approximation made and
the n_eff it reports alongside D.
"""
import argparse
import csv
import itertools
import math
import os
import sys
import time

import numpy as np

# iter_fasta, the byte lookup table and the missing-state convention are worth
# sharing with the pi stage; count_states is not, because this stage needs a
# per-population variant of it. snakemake/pi is not a package, hence the path
# insert rather than an import.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pi"))
from gene_pi import AlignmentError, BASE_LUT, N_BASES, iter_fasta  # noqa: E402

WHOLE_SAMPLE = "all"

STATS_HEADER = [
    "node",
    "n_seqs",
    "n_seqs_counted",
    "n_seqs_assigned",
    "n_duplicate_copies",
    "n_sites_total",
    "n_sites_kept",
    "frac_kept",
    "n_populations",
    "n_pairs",
    "status",
    "seconds",
    "message",
]

DXY_HEADER = [
    "node",
    "pop1",
    "pop2",
    "n_seqs_1",
    "n_seqs_2",
    "n_sites",
    "n_sites_shared",
    "dxy",
    "dxy_shared",
    "pi_1",
    "pi_2",
    "da",
    "fst_hudson",
]

TAJIMA_HEADER = [
    "node",
    "population",
    "n_seqs",
    "n_sites",
    "n_segregating",
    "pi_total",
    "pi_per_site",
    "theta_w",
    "tajima_d",
    "n_eff",
]


def load_clusters(path, min_size, only=None):
    """Read the PopPUNK CSV into {sample: population index} plus the labels.

    `only` overrides the size threshold when given. The 'novel' catch-all is
    always dropped: it is not a lineage, just everything PopPUNK could not
    assign.
    """
    members = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Taxon" not in reader.fieldnames \
                or "Cluster" not in reader.fieldnames:
            raise SystemExit(f"{path}: expected columns Taxon,Cluster, got "
                             f"{reader.fieldnames}")
        for row in reader:
            members.setdefault(row["Cluster"], []).append(row["Taxon"])

    if only:
        wanted = [c for c in only if c in members]
        missing = [c for c in only if c not in members]
        if missing:
            raise SystemExit(f"{path}: no such cluster(s): {', '.join(missing)}")
    else:
        wanted = [c for c, m in members.items()
                  if c != "novel" and len(m) >= min_size]
    # largest first, so the pair table reads in a stable, meaningful order
    wanted.sort(key=lambda c: (-len(members[c]), c))

    sample_to_pop = {}
    for index, cluster in enumerate(wanted):
        for sample in members[cluster]:
            sample_to_pop[sample] = index
    sizes = [len(members[c]) for c in wanted]
    return sample_to_pop, wanted, sizes


def sample_of(header):
    """Alignment header -> the PopPUNK Taxon it belongs to."""
    sample = header.split(";", 1)[0]
    return sample[:-3] if sample.endswith(".fa") else sample


def count_states_by_pop(path, sample_to_pop, n_pops, block_cells,
                        dedup=True):
    """One pass over an alignment, returning per-population column counts.

    Returns (counts, n_seqs, n_counted, n_assigned, n_duplicates) where counts
    has shape
    (n_pops + 1, n_columns, N_BASES + 1): the last population slot collects
    sequences from unselected lineages so the whole-sample totals stay exact
    (counts.sum(axis=0)), and the last state column is the missing tally.

    This is count_states() from the pi stage with a population stride added to
    the bincount index, so the extra grouping costs one more array add per block
    rather than a second pass over the file.
    """
    n_states = N_BASES
    stride = None
    flat = None
    width = 0
    offsets = None
    rows_per_block = 0
    n_seqs = 0
    n_counted = 0
    n_assigned = 0
    n_duplicates = 0
    seen = set()
    buf = []
    pops = []
    unassigned = n_pops

    def flush():
        block = np.frombuffer(b"".join(buf), dtype=np.uint8).reshape(-1, width)
        codes = BASE_LUT[block].astype(np.int64)
        codes += offsets
        codes += np.asarray(pops, dtype=np.int64)[:, None] * stride
        flat[...] += np.bincount(codes.ravel(), minlength=(n_pops + 1) * stride)
        buf.clear()
        pops.clear()

    for header, seq in iter_fasta(path):
        if flat is None:
            width = len(seq)
            if width == 0:
                raise AlignmentError(f"{path}: first record is empty")
            stride = width * (n_states + 1)
            flat = np.zeros((n_pops + 1) * stride, dtype=np.int64)
            offsets = np.arange(width, dtype=np.int64) * (n_states + 1)
            rows_per_block = max(1, block_cells // width)
        elif len(seq) != width:
            raise AlignmentError(
                f"{path}: record {header!r} has length {len(seq)}, expected "
                f"{width}; this is not an alignment")
        n_seqs += 1
        sample = sample_of(header)
        if dedup:
            if sample in seen:
                n_duplicates += 1
                continue
            seen.add(sample)
        pop = sample_to_pop.get(sample, unassigned)
        if pop != unassigned:
            n_assigned += 1
        buf.append(seq.encode())
        pops.append(pop)
        n_counted += 1
        if len(buf) == rows_per_block:
            flush()
    if buf:
        flush()
    if flat is None:
        raise AlignmentError(f"{path}: file contains no records")
    return (flat.reshape(n_pops + 1, width, n_states + 1),
            n_seqs, n_counted, n_assigned, n_duplicates)


def column_frequencies(counts):
    """(real counts, per-column n) for one population's count matrix."""
    real = counts[:, :N_BASES].astype(np.float64)
    return real, real.sum(axis=1)


def column_pi(real, n):
    """Per-column pi = n/(n-1) * (1 - sum p^2), NaN where n < 2.

    Returned per column rather than averaged: Tajima's D needs the sum over
    sites, dXY needs the columns lined up with another population's.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = real / n[:, None]
        pi = (n / (n - 1.0)) * (1.0 - (freq ** 2).sum(axis=1))
    return np.where(n >= 2, pi, np.nan)


# a1(n) is needed once per segregating column, with n running to the tens of
# thousands: computed naively that is O(S x n) per gene and dominates the stage.
# Cache the prefix sums instead, so a1(n) is a lookup: _A1[k] = sum_{i=1}^{k} 1/i,
# hence a1(n) = _A1[n - 1].
_A1 = np.zeros(1)
_A2 = np.zeros(1)


def _grow_harmonic(n):
    global _A1, _A2
    if len(_A1) > n:
        return
    idx = np.arange(1, n + 2, dtype=np.float64)
    _A1 = np.concatenate(([0.0], np.cumsum(1.0 / idx)))
    _A2 = np.concatenate(([0.0], np.cumsum(1.0 / idx ** 2)))


def harmonic(n):
    """a1 = sum_{i=1}^{n-1} 1/i, and a2 = sum 1/i^2."""
    _grow_harmonic(n)
    return float(_A1[n - 1]), float(_A2[n - 1])


def harmonic_a1(n_values):
    """Vectorised a1 for an array of per-column sample sizes."""
    n_values = np.asarray(n_values, dtype=np.int64)
    _grow_harmonic(int(n_values.max()))
    return _A1[n_values - 1]


def tajima(real, n, keep):
    """Tajima's D for one population over the kept columns.

    pi is the per-gene SUM of per-column pi and theta_W is accumulated per
    column as sum over segregating columns of 1/a1(n_j), so both estimators
    respect the per-column sample size that pairwise deletion produces.

    The variance does not decompose that way: e1 and e2 are functions of a
    single n. They are evaluated at n_eff, the mean n_j over the segregating
    columns, which is exact when occupancy is uniform and an approximation
    otherwise. n_eff is reported so the reader can see how far from uniform a
    given gene was.
    """
    usable = keep & (n >= 2)
    if not usable.any():
        return None
    pi_cols = column_pi(real[usable], n[usable])
    n_used = n[usable]
    segregating = (real[usable] > 0).sum(axis=1) >= 2
    n_sites = int(usable.sum())
    S = int(segregating.sum())
    pi_total = float(np.nansum(pi_cols))
    result = {
        "n_sites": n_sites,
        "n_segregating": S,
        "pi_total": pi_total,
        "pi_per_site": pi_total / n_sites if n_sites else float("nan"),
        "theta_w": float("nan"),
        "tajima_d": float("nan"),
        "n_eff": float("nan"),
    }
    if S == 0:
        result["theta_w"] = 0.0
        return result

    n_seg = n_used[segregating]
    theta_w = float((1.0 / harmonic_a1(n_seg)).sum())
    result["theta_w"] = theta_w

    n_eff = int(round(float(n_seg.mean())))
    result["n_eff"] = n_eff
    if S < 2 or n_eff < 4:
        # the variance needs S(S-1) > 0, and c2 is undefined below n = 4
        return result
    a1, a2 = harmonic(n_eff)
    b1 = (n_eff + 1) / (3.0 * (n_eff - 1))
    b2 = 2.0 * (n_eff ** 2 + n_eff + 3) / (9.0 * n_eff * (n_eff - 1))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n_eff + 2) / (a1 * n_eff) + a2 / a1 ** 2
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)
    variance = e1 * S + e2 * S * (S - 1)
    if variance > 0:
        result["tajima_d"] = (pi_total - theta_w) / math.sqrt(variance)
    return result


def dxy_pair(real_x, n_x, real_y, n_y, keep):
    """dXY and the matching within-population pi, over the shared columns.

    Per column dXY = 1 - sum_a p_a^X p_a^Y. Every cross-population pair is
    distinct, so unlike pi this needs no n/(n-1) correction.

    pi_X and pi_Y are averaged over the SAME columns as dXY rather than over
    each population's own usable set, so that d_a and F_ST are internally
    consistent.
    """
    usable = keep & (n_x >= 1) & (n_y >= 1)
    if not usable.any():
        return None
    px = real_x[usable] / n_x[usable, None]
    py = real_y[usable] / n_y[usable, None]
    dxy = float((1.0 - (px * py).sum(axis=1)).mean())

    # pi needs n >= 2, dXY only needs n >= 1, so d_a and F_ST are formed on the
    # narrower shared set and dxy is recomputed there. Mixing the two column
    # sets would subtract a pi measured over one region from a dXY measured over
    # another; the sets differ by a handful of columns in practice, and
    # reporting both site counts makes the gap visible rather than assumed away.
    shared = keep & (n_x >= 2) & (n_y >= 2)
    result = {
        "n_sites": int(usable.sum()),
        "n_sites_shared": int(shared.sum()),
        "dxy": dxy,
        "dxy_shared": float("nan"),
        "pi_1": float("nan"),
        "pi_2": float("nan"),
        "da": float("nan"),
        "fst_hudson": float("nan"),
    }
    if not shared.any():
        return result
    sx = real_x[shared] / n_x[shared, None]
    sy = real_y[shared] / n_y[shared, None]
    dxy_shared = float((1.0 - (sx * sy).sum(axis=1)).mean())
    pi_x = float(np.nanmean(column_pi(real_x[shared], n_x[shared])))
    pi_y = float(np.nanmean(column_pi(real_y[shared], n_y[shared])))
    pi_within = 0.5 * (pi_x + pi_y)
    result.update({
        "dxy_shared": dxy_shared,
        "pi_1": pi_x,
        "pi_2": pi_y,
        "da": dxy_shared - pi_within,
        "fst_hudson": (1.0 - pi_within / dxy_shared
                       if dxy_shared > 0 else float("nan")),
    })
    return result


def num(value, digits=8):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def process_gene(node, path, args, sample_to_pop, labels, writers):
    """Measure one gene. Returns its stats row."""
    started = time.time()
    row = dict.fromkeys(STATS_HEADER, "")
    row["node"] = node

    def finish(status, message=""):
        row["status"] = status
        row["message"] = message
        row["seconds"] = f"{time.time() - started:.1f}"
        return row

    n_pops = len(labels)
    try:
        counts, n_seqs, n_counted, n_assigned, n_duplicates = count_states_by_pop(
            path, sample_to_pop, n_pops, args.block_cells,
            dedup=(args.paralogs == "first"))
    except AlignmentError as err:
        return finish("read_error", str(err))
    except OSError as err:
        return finish("read_error", str(err))

    row["n_seqs"] = n_seqs
    row["n_seqs_counted"] = n_counted
    row["n_seqs_assigned"] = n_assigned
    row["n_duplicate_copies"] = n_duplicates
    row["n_sites_total"] = counts.shape[1]

    if n_counted < args.min_seqs:
        return finish("too_few_sequences",
                      f"{n_counted} < --min-seqs {args.min_seqs}")

    total = counts.sum(axis=0)
    occupancy = total[:, :N_BASES].sum(axis=1)
    keep = occupancy >= args.min_occupancy * n_counted
    row["n_sites_kept"] = int(keep.sum())
    row["frac_kept"] = f"{keep.sum() / counts.shape[1]:.4f}"
    if keep.sum() < args.min_sites:
        return finish("too_few_sites",
                      f"{int(keep.sum())} < --min-sites {args.min_sites}")

    # per-population frequency views, computed once and reused by every pair
    per_pop = []
    for index in range(n_pops):
        real, n = column_frequencies(counts[index])
        present = int(n.max()) if n.size else 0
        per_pop.append((real, n, present))

    usable_pops = [i for i in range(n_pops)
                   if per_pop[i][2] >= args.min_pop_seqs]
    row["n_populations"] = len(usable_pops)

    # Tajima's D: whole sample first, then each usable lineage
    whole_real, whole_n = column_frequencies(total)
    stats = tajima(whole_real, whole_n, keep)
    if stats:
        writers["tajima"].writerow({
            "node": node, "population": WHOLE_SAMPLE, "n_seqs": n_counted,
            "n_sites": stats["n_sites"], "n_segregating": stats["n_segregating"],
            "pi_total": num(stats["pi_total"]),
            "pi_per_site": num(stats["pi_per_site"]),
            "theta_w": num(stats["theta_w"]),
            "tajima_d": num(stats["tajima_d"], 6),
            "n_eff": num(stats["n_eff"]),
        })
    for index in usable_pops:
        real, n, present = per_pop[index]
        stats = tajima(real, n, keep)
        if stats:
            writers["tajima"].writerow({
                "node": node, "population": labels[index], "n_seqs": present,
                "n_sites": stats["n_sites"],
                "n_segregating": stats["n_segregating"],
                "pi_total": num(stats["pi_total"]),
                "pi_per_site": num(stats["pi_per_site"]),
                "theta_w": num(stats["theta_w"]),
                "tajima_d": num(stats["tajima_d"], 6),
                "n_eff": num(stats["n_eff"]),
            })

    n_pairs = 0
    for i, j in itertools.combinations(usable_pops, 2):
        real_x, n_x, present_x = per_pop[i]
        real_y, n_y, present_y = per_pop[j]
        pair = dxy_pair(real_x, n_x, real_y, n_y, keep)
        if pair is None:
            continue
        n_pairs += 1
        writers["dxy"].writerow({
            "node": node, "pop1": labels[i], "pop2": labels[j],
            "n_seqs_1": present_x, "n_seqs_2": present_y,
            "n_sites": pair["n_sites"],
            "n_sites_shared": pair["n_sites_shared"],
            "dxy": num(pair["dxy"]), "dxy_shared": num(pair["dxy_shared"]),
            "pi_1": num(pair["pi_1"]),
            "pi_2": num(pair["pi_2"]), "da": num(pair["da"]),
            "fst_hudson": num(pair["fst_hudson"], 6),
        })
    row["n_pairs"] = n_pairs

    if len(usable_pops) < 2:
        return finish("too_few_populations",
                      f"{len(usable_pops)} lineage(s) with >= "
                      f"--min-pop-seqs {args.min_pop_seqs} sequences")
    return finish("computed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alignments", required=True,
                        help="TSV of node<TAB>path written by run_popgen.sh")
    parser.add_argument("--outdir", required=True,
                        help="popgen output directory")
    parser.add_argument("--chunk-index", type=int, required=True,
                        help="0-based chunk to process (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--chunk-size", type=int, required=True,
                        help="number of genes per chunk")
    parser.add_argument("--clusters", required=True,
                        help="PopPUNK Taxon,Cluster CSV")
    parser.add_argument("--min-cluster-size", type=int, default=500,
                        help="lineages smaller than this are not measured "
                             "(default: %(default)s)")
    parser.add_argument("--only-clusters", nargs="*", default=None,
                        help="measure exactly these lineages, ignoring "
                             "--min-cluster-size")
    parser.add_argument("--min-occupancy", type=float, default=0.2,
                        help="keep columns present in at least this fraction "
                             "of sequences (default: %(default)s)")
    parser.add_argument("--min-seqs", type=int, default=20,
                        help="skip genes with fewer sequences "
                             "(default: %(default)s)")
    parser.add_argument("--min-sites", type=int, default=90,
                        help="skip genes with fewer retained columns. These are "
                             "NUCLEOTIDE columns, so the default is the pi "
                             "stage's 30 amino-acid sites x 3 "
                             "(default: %(default)s)")
    parser.add_argument("--min-pop-seqs", type=int, default=20,
                        help="skip a lineage in a gene when it contributes "
                             "fewer sequences than this (default: %(default)s)")
    parser.add_argument("--paralogs", choices=("first", "all"), default="first",
                        help="an isolate with two copies of a gene contributes "
                             "one allele ('first') or both records ('all') "
                             "(default: %(default)s)")
    parser.add_argument("--block-cells", type=int, default=4000000,
                        help="alignment cells counted per numpy block "
                             "(default: %(default)s)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    with open(args.alignments) as handle:
        genes = [line.rstrip("\n").split("\t")
                 for line in handle if line.strip()]
    start = args.chunk_index * args.chunk_size
    chunk = genes[start:start + args.chunk_size]
    if not chunk:
        print(f"chunk {args.chunk_index}: nothing to do "
              f"({len(genes)} genes total)")
        return 0

    sample_to_pop, labels, sizes = load_clusters(
        args.clusters, args.min_cluster_size, args.only_clusters)
    if len(labels) < 2:
        raise SystemExit(f"{args.clusters}: fewer than two lineages selected "
                         f"(--min-cluster-size {args.min_cluster_size})")

    print(f"chunk {args.chunk_index}: genes {start}-{start + len(chunk) - 1} "
          f"of {len(genes)}, {len(labels)} lineages "
          f"({len(labels) * (len(labels) - 1) // 2} pairs), "
          f"min-occupancy {args.min_occupancy}")
    print("lineages: " + ", ".join(f"{c} (n={s})"
                                   for c, s in zip(labels, sizes)))

    for sub in ("stats", "dxy", "tajima", "logs"):
        os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)

    stats_path = os.path.join(args.outdir, "stats", f"chunk_{args.chunk_index}.tsv")
    dxy_path = os.path.join(args.outdir, "dxy", f"chunk_{args.chunk_index}.tsv")
    taj_path = os.path.join(args.outdir, "tajima", f"chunk_{args.chunk_index}.tsv")

    with open(stats_path, "w", newline="") as fs, \
            open(dxy_path, "w", newline="") as fd, \
            open(taj_path, "w", newline="") as ft:
        stats_writer = csv.DictWriter(fs, fieldnames=STATS_HEADER, delimiter="\t")
        stats_writer.writeheader()
        writers = {
            "dxy": csv.DictWriter(fd, fieldnames=DXY_HEADER, delimiter="\t"),
            "tajima": csv.DictWriter(ft, fieldnames=TAJIMA_HEADER, delimiter="\t"),
        }
        writers["dxy"].writeheader()
        writers["tajima"].writeheader()
        for entry in chunk:
            node, path = entry[0], entry[1]
            row = process_gene(node, path, args, sample_to_pop, labels, writers)
            stats_writer.writerow(row)
            # flush after every gene: a walltime kill should still leave a
            # usable partial chunk, as in the pi stage
            fs.flush()
            fd.flush()
            ft.flush()
            print(f"[{node}] {row['status']} "
                  f"pops={row['n_populations']} pairs={row['n_pairs']} "
                  f"{row['seconds']}s")

    print(f"chunk {args.chunk_index} done: {len(chunk)} genes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
