#!/usr/bin/env python3
"""Gather the per-chunk popgen fragments into three tables.

Runs once, after the array finishes, as a dependent job submitted by
run_popgen.sh. The node list comes from alignments.tsv rather than from whatever
happens to be in stats/, so every gene that went in comes out: genes with no
usable measurement appear with status "not_run".

The dXY table gains its normalisation here rather than in the worker, because it
needs every gene at once: a lineage pair's absolute divergence is set by how far
apart the two lineages are overall, so the interesting quantity per gene is its
dXY relative to that pair's genome-wide median.
"""
import argparse
import array
import csv
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_popgen import DXY_HEADER, STATS_HEADER, TAJIMA_HEADER  # noqa: E402

DXY_OUT_HEADER = DXY_HEADER + ["dxy_median_pair", "dxy_rel"]


def fragment_paths(directory):
    """chunk_*.tsv in numeric chunk order.

    sorted() on the glob would be lexical, putting chunk_10 before chunk_2. The
    stats and tajima tables are assembled through dicts so order does not matter
    there, but the dxy pass streams and must not reorder genes.
    """
    paths = glob.glob(os.path.join(directory, "chunk_*.tsv"))
    paths.sort(key=lambda p: int(os.path.basename(p)[len("chunk_"):-len(".tsv")]))
    return paths


def stream_fragments(directory, header):
    """Yield rows one at a time, so a table larger than memory can be merged.

    gene_dxy.tsv is genes x lineage pairs: 190 pairs is ~320k rows for
    s_pyogenes, but 1,176 pairs over 25,883 genes is ~30M, which cannot be held
    as dicts. Everything that touches the dxy table therefore streams.
    """
    for path in fragment_paths(directory):
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = [c for c in header if c not in (reader.fieldnames or [])]
            if missing:
                print(f"warning: {path} is missing column(s) "
                      f"{', '.join(missing)}; skipped", file=sys.stderr)
                continue
            for row in reader:
                yield row


def read_fragments(directory, header):
    """Every chunk_*.tsv in `directory`, in chunk order."""
    return list(stream_fragments(directory, header))


def write_table(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    alignments = os.path.join(args.outdir, "alignments.tsv")
    with open(alignments) as handle:
        nodes = [line.split("\t", 1)[0] for line in handle if line.strip()]

    # ---- stats: one row per gene, in alignments.tsv order
    stats = {r["node"]: r for r in read_fragments(
        os.path.join(args.outdir, "stats"), STATS_HEADER)}
    stats_rows = []
    for node in nodes:
        row = stats.get(node)
        if row is None:
            row = dict.fromkeys(STATS_HEADER, "")
            row["node"] = node
            row["status"] = "not_run"
        stats_rows.append(row)
    write_table(os.path.join(args.outdir, "popgen_stats.tsv"),
                STATS_HEADER, stats_rows)

    status_counts = {}
    for row in stats_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print(f"nodes: {len(nodes)}")
    for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {count}")

    # ---- tajima: one row per gene per population, as produced
    tajima_rows = read_fragments(os.path.join(args.outdir, "tajima"),
                                 TAJIMA_HEADER)
    write_table(os.path.join(args.outdir, "gene_tajima.tsv"),
                TAJIMA_HEADER, tajima_rows)
    with_d = [r for r in tajima_rows if r["tajima_d"]]
    if with_d:
        values = sorted(float(r["tajima_d"]) for r in with_d)
        print(f"Tajima's D: {len(values)} values, median "
              f"{statistics.median(values):.3f}, "
              f"{sum(1 for v in values if v > 0)} positive")

    # ---- dxy: two streaming passes. Pass 1 collects the per-pair dxy values
    # (190-1,176 pairs x at most one float per gene, so tens of MB at worst);
    # pass 2 re-streams and writes, appending the normalisation. Never holds the
    # whole table.
    dxy_dir = os.path.join(args.outdir, "dxy")
    by_pair = {}
    n_rows = n_with_dxy = 0
    for row in stream_fragments(dxy_dir, DXY_HEADER):
        n_rows += 1
        if row["dxy"]:
            n_with_dxy += 1
            key = (row["pop1"], row["pop2"])
            if key not in by_pair:
                by_pair[key] = array.array("d")
            by_pair[key].append(float(row["dxy"]))
    medians = {pair: statistics.median(values)
               for pair, values in by_pair.items() if len(values)}

    summary_path = os.path.join(args.outdir, "pair_summary.tsv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["pop1", "pop2", "n_genes", "dxy_median",
                         "dxy_q1", "dxy_q3"])
        for (x, y), values in sorted(by_pair.items()):
            ordered = sorted(values)
            n = len(ordered)
            writer.writerow([x, y, n, f"{medians[(x, y)]:.8g}",
                             f"{ordered[n // 4]:.8g}", f"{ordered[3 * n // 4]:.8g}"])
    print(f"wrote {summary_path} ({len(by_pair)} pairs)")

    out_path = os.path.join(args.outdir, "gene_dxy.tsv")
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DXY_OUT_HEADER,
                                delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in stream_fragments(dxy_dir, DXY_HEADER):
            median = medians.get((row["pop1"], row["pop2"]))
            row["dxy_median_pair"] = f"{median:.8g}" if median else ""
            row["dxy_rel"] = (f"{float(row['dxy']) / median:.6g}"
                              if median and row["dxy"] else "")
            writer.writerow(row)
    print(f"wrote {out_path} ({n_rows} rows, {n_with_dxy} with dxy)")
    print(f"lineage pairs with data: {len(medians)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
