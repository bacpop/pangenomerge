#!/usr/bin/env python3
"""Figures for a finished popgen run: between-lineage dXY and Tajima's D.

Writes three PNGs (no PDFs, by project convention):

  <prefix>_dxy_matrix.png   lineage x lineage median dXY, the population
                            structure the per-gene values are measured against
  <prefix>_tajima.png       Tajima's D within lineages and over the whole
                            sample, with the neutral expectation marked
  <prefix>_dxy_outliers.png dxy_rel (each gene against its own pair's median)
                            and Hudson's F_ST

Usage:
  plot_popgen.py --popgen-dir <results>/popgen --out-prefix <results>/plots/popgen \\
                 --title "Streptococcus pyogenes" [--paper]

Reading these: dXY between two PopPUNK lineages is dominated by how diverged the
lineages are overall, so the per-gene signal lives in dxy_rel, not in dXY. Genes
far above their pair's median are candidate recombination barriers or
lineage-specific adaptations; genes far below are candidates for recent
horizontal exchange between the two lineages. Tajima's D on a collection of this
size and structure is descriptive, not a neutrality test - clonal expansion and
convenience sampling both push it strongly negative.
"""
import argparse
import array
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#d62728"
INK, INK2, INK3, GRID = "#14181c", "#4d565e", "#7b858d", "#e6eaed"
FS = 8  # points added to every font size, matching the other plot scripts

# sequential blue, floored at step 250 so the palest cells stay visible on white
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])

ap = argparse.ArgumentParser()
ap.add_argument("--popgen-dir", required=True)
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--title", default="")
ap.add_argument("--paper", action="store_true")
ap.add_argument("--width", type=float, default=6.3)
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()

if a.paper:
    SZ = dict(tick=7, label=8, ann=7, title=12)
    DPI = 600 if a.dpi == 200 else a.dpi
    GRID_LW, TITLE_Y, TITLE_BOLD = 0.5, 1.01, "normal"
    SCALE = a.width / 13.0
else:
    SZ = dict(tick=9 + FS, label=11 + FS, ann=9 + FS, title=15 + FS)
    DPI = a.dpi
    GRID_LW, TITLE_Y, TITLE_BOLD = 0.8, 1.03, "bold"
    SCALE = 1.0

os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)) or ".", exist_ok=True)


def rows(name):
    """Whole table as a list. Only for tables known to be small."""
    path = os.path.join(a.popgen_dir, name)
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def stream(name):
    """Row-at-a-time. gene_dxy.tsv is genes x lineage pairs and reaches 29M
    rows / 280 MB for s_pneumoniae's 1,128 pairs, so nothing here holds it."""
    path = os.path.join(a.popgen_dir, name)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            yield row


def fnum(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def style(ax, xlabel, ylabel, axis="y"):
    ax.set_xlabel(xlabel, fontsize=SZ["label"], color=INK2)
    ax.set_ylabel(ylabel, fontsize=SZ["label"], color=INK2)
    ax.grid(True, axis=axis, color=GRID, lw=GRID_LW, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8ced3")
    ax.tick_params(colors=INK3, labelsize=SZ["tick"])


def finish(fig, path):
    if a.title:
        fig.suptitle(a.title, fontsize=SZ["title"], color=INK, y=TITLE_Y,
                     ha="center", style="italic", fontweight=TITLE_BOLD)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print("wrote", path)
    plt.close(fig)


taj_rows = rows("gene_tajima.tsv")

# ---------------------------------------------------------------- figure 1
# lineage x lineage median dXY. The per-pair medians come from the collect
# step's pair_summary.tsv when it exists, so the 29M-row table is streamed once
# for the histograms rather than aggregated here.
order, seen = [], set()
pair_median = {}
summary_path = os.path.join(a.popgen_dir, "pair_summary.tsv")
if os.path.exists(summary_path):
    for r in rows("pair_summary.tsv"):
        for pop in (r["pop1"], r["pop2"]):
            if pop not in seen:
                seen.add(pop)
                order.append(pop)
        if r["dxy_median"]:
            pair_median[(r["pop1"], r["pop2"])] = float(r["dxy_median"])
    print(f"read {len(pair_median):,} pair medians from pair_summary.tsv")

rel_values = array.array("d")
fst_values = array.array("d")
by_pair = defaultdict(lambda: array.array("d"))
n_dxy_rows = 0
for r in stream("gene_dxy.tsv"):
    n_dxy_rows += 1
    if not order or not pair_median:
        for pop in (r["pop1"], r["pop2"]):
            if pop not in seen:
                seen.add(pop)
                order.append(pop)
    value = fnum(r, "dxy")
    if value is not None and not pair_median:
        by_pair[(r["pop1"], r["pop2"])].append(value)
    rel = fnum(r, "dxy_rel")
    if rel is not None and rel > 0:
        rel_values.append(rel)
    fst = fnum(r, "fst_hudson")
    if fst is not None:
        fst_values.append(fst)
print(f"streamed {n_dxy_rows:,} dxy rows and {len(taj_rows):,} tajima rows")

if not pair_median:
    pair_median = {pair: float(np.median(values))
                   for pair, values in by_pair.items() if len(values)}

index = {pop: i for i, pop in enumerate(order)}
size = len(order)
matrix = np.full((size, size), np.nan)
for (x, y), median in pair_median.items():
    if x in index and y in index:
        i, j = index[x], index[y]
        matrix[i, j] = matrix[j, i] = median

fig, ax = plt.subplots(figsize=(9.5 * SCALE, 8.2 * SCALE), constrained_layout=True)
image = ax.imshow(matrix * 1e3, cmap=SEQ_CMAP, interpolation="nearest")
ax.set_xticks(range(size)); ax.set_yticks(range(size))
tick_size = SZ["tick"] if size <= 30 else max(5, SZ["tick"] - 4)
ax.set_xticklabels(order, fontsize=tick_size, color=INK3, rotation=90)
ax.set_yticklabels(order, fontsize=tick_size, color=INK3)
ax.set_xlabel("PopPUNK lineage", fontsize=SZ["label"], color=INK2)
ax.set_ylabel("PopPUNK lineage", fontsize=SZ["label"], color=INK2)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
bar.set_label(r"median $d_{XY}$ across genes ($\times 10^{-3}$)",
              fontsize=SZ["label"], color=INK2)
bar.ax.tick_params(colors=INK3, labelsize=SZ["tick"])
bar.outline.set_visible(False)
# annotate only when the grid is coarse enough for the numbers to fit
if size <= 12:
    for i in range(size):
        for j in range(size):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j] * 1e3:.1f}", ha="center",
                        va="center", fontsize=SZ["ann"] - 2,
                        color="white" if matrix[i, j] > np.nanmedian(matrix) else INK)
finish(fig, f"{a.out_prefix}_dxy_matrix.png")

# ---------------------------------------------------------------- figure 2
within = np.array([v for v in (fnum(r, "tajima_d") for r in taj_rows
                               if r["population"] != "all") if v is not None])
whole = np.array([v for v in (fnum(r, "tajima_d") for r in taj_rows
                              if r["population"] == "all") if v is not None])

fig, axes = plt.subplots(1, 2, figsize=(13.0 * SCALE, 5.2 * SCALE),
                         constrained_layout=True, sharey=True)
edges = np.linspace(min(within.min(), whole.min()) - 0.1,
                    max(within.max(), whole.max()) + 0.1, 70)
for ax, values, colour, label in ((axes[0], within, BLUE, "within lineage"),
                                  (axes[1], whole, ORANGE, "whole sample")):
    height, _ = np.histogram(values, bins=edges)
    ax.stairs(100 * height / len(values), edges, color=colour, lw=1.8,
              fill=False, zorder=3)
    ax.stairs(100 * height / len(values), edges, color=colour, alpha=0.13,
              fill=True, zorder=2)
    ax.axvline(0.0, color=RED, lw=1.4, ls="--", dashes=(6, 4), zorder=4)
    median = float(np.median(values))
    ax.axvline(median, color=INK2, lw=1.2, zorder=4)
    ax.text(0.02, 0.97,
            f"{label}\nn = {len(values):,}\nmedian {median:+.2f}\n"
            f"{100 * (values > 0).mean():.1f}% above 0",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=SZ["ann"], color=INK2, linespacing=1.5)
    style(ax, r"Tajima's $D$", "measurements (%)" if ax is axes[0] else "")
axes[0].text(0.0, axes[0].get_ylim()[1], "  neutral", color=RED,
             fontsize=SZ["ann"], va="top", ha="left")
finish(fig, f"{a.out_prefix}_tajima.png")

# ---------------------------------------------------------------- figure 3
rel = np.frombuffer(rel_values, dtype=np.float64)
fst = np.frombuffer(fst_values, dtype=np.float64)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0 * SCALE, 5.2 * SCALE),
                               constrained_layout=True)
log_rel = np.log10(rel)
edges = np.linspace(log_rel.min(), log_rel.max(), 80)
height, _ = np.histogram(log_rel, bins=edges)
ax1.stairs(100 * height / len(log_rel), edges, color=BLUE, lw=1.8, fill=False, zorder=3)
ax1.stairs(100 * height / len(log_rel), edges, color=BLUE, alpha=0.13, fill=True, zorder=2)
ax1.axvline(0.0, color=RED, lw=1.4, ls="--", dashes=(6, 4), zorder=4)
lo, hi = np.percentile(log_rel, [1, 99])
for edge in (lo, hi):
    ax1.axvline(edge, color=INK3, lw=1.0, ls=":", zorder=4)
ax1.text(0.02, 0.97,
         f"n = {len(rel):,} gene-pair values\n"
         f"dotted: 1st and 99th percentile\n"
         f"({10 ** lo:.2f}x and {10 ** hi:.2f}x the pair median)",
         transform=ax1.transAxes, va="top", ha="left",
         fontsize=SZ["ann"], color=INK2, linespacing=1.5)
style(ax1, r"$\log_{10}(d_{XY}$ / pair median$)$", "gene-pair values (%)")

edges = np.linspace(min(0.0, fst.min()), 1.0, 60)
height, _ = np.histogram(fst, bins=edges)
ax2.stairs(100 * height / len(fst), edges, color=AQUA, lw=1.8, fill=False, zorder=3)
ax2.stairs(100 * height / len(fst), edges, color=AQUA, alpha=0.13, fill=True, zorder=2)
ax2.text(0.02, 0.97,
         f"n = {len(fst):,}\nmedian {np.median(fst):.3f}",
         transform=ax2.transAxes, va="top", ha="left",
         fontsize=SZ["ann"], color=INK2, linespacing=1.5)
style(ax2, r"Hudson's $F_{ST}$", "gene-pair values (%)")
finish(fig, f"{a.out_prefix}_dxy_outliers.png")
