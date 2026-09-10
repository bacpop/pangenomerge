#!/usr/bin/env python3
"""One figure per statistic, with a panel per species.

Companion to plot_popgen.py, which draws a single species. This walks a set of
results directories and lays them out on a shared figure so the species can be
compared directly:

  combined_dxy_matrix.png    lineage x lineage median dXY, one heatmap each
  combined_tajima.png        Tajima's D within lineages vs whole sample
  combined_dxy_outliers.png  dxy_rel and Hudson's F_ST, species across columns

A species whose popgen stage has not been run keeps its place in the layout and
is labelled "unfinished", so the comparison shows what is missing rather than
silently rearranging itself.

Usage:
  plot_popgen_combined.py --root <species_pangenomes> --outdir <dir> \\
      --species s_aureus s_pyogenes m_tuberculosis s_pneumoniae
"""
import argparse
import array
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#d62728"
INK, INK2, INK3, GRID = "#14181c", "#4d565e", "#7b858d", "#e6eaed"
FS = 4  # smaller bump than the single-species plots: four panels, less room

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])

GENUS = {"s": "Streptococcus", "m": "Mycobacterium", "k": "Klebsiella",
         "e": "Escherichia", "p": "Pseudomonas"}
SPECIES_NAMES = {
    "s_aureus": "Staphylococcus aureus",
    "s_pyogenes": "Streptococcus pyogenes",
    "s_pneumoniae": "Streptococcus pneumoniae",
    "m_tuberculosis": "Mycobacterium tuberculosis",
}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="species_pangenomes directory")
ap.add_argument("--species", nargs="+", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--paper", action="store_true")
ap.add_argument("--width", type=float, default=6.3)
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()

if a.paper:
    SZ = dict(tick=6, label=7, ann=6, title=9, panel=9)
    DPI = 600 if a.dpi == 200 else a.dpi
    GRID_LW, SCALE = 0.5, a.width / 15.0
else:
    SZ = dict(tick=8 + FS, label=10 + FS, ann=8 + FS, title=13 + FS, panel=12 + FS)
    DPI, GRID_LW, SCALE = a.dpi, 0.8, 1.0

os.makedirs(a.outdir, exist_ok=True)


def display_name(key):
    if key in SPECIES_NAMES:
        return SPECIES_NAMES[key]
    head, _, tail = key.partition("_")
    return f"{GENUS.get(head, head.upper())} {tail}".strip()


def read_table(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fnum(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def load(key):
    """Everything the three figures need from one species, or None."""
    popgen = os.path.join(a.root, key, "results", "popgen")
    dxy_path = os.path.join(popgen, "gene_dxy.tsv")
    taj_path = os.path.join(popgen, "gene_tajima.tsv")
    if not (os.path.isfile(dxy_path) and os.path.isfile(taj_path)):
        return None

    order, seen, pair_median = [], set(), {}
    summary = os.path.join(popgen, "pair_summary.tsv")
    if os.path.isfile(summary):
        for row in read_table(summary):
            for pop in (row["pop1"], row["pop2"]):
                if pop not in seen:
                    seen.add(pop)
                    order.append(pop)
            if row["dxy_median"]:
                pair_median[(row["pop1"], row["pop2"])] = float(row["dxy_median"])

    # A run collected before pair_summary.tsv existed has no medians on disk;
    # accumulate them from the table itself rather than drawing an empty matrix.
    by_pair = {}
    need_medians = not pair_median
    rel, fst = array.array("d"), array.array("d")
    with open(dxy_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if need_medians:
                for pop in (row["pop1"], row["pop2"]):
                    if pop not in seen:
                        seen.add(pop)
                        order.append(pop)
                value = fnum(row, "dxy")
                if value is not None:
                    key_pair = (row["pop1"], row["pop2"])
                    if key_pair not in by_pair:
                        by_pair[key_pair] = array.array("d")
                    by_pair[key_pair].append(value)
            value = fnum(row, "dxy_rel")
            if value is not None and value > 0:
                rel.append(value)
            value = fnum(row, "fst_hudson")
            if value is not None:
                fst.append(value)

    within, whole = array.array("d"), array.array("d")
    for row in read_table(taj_path):
        value = fnum(row, "tajima_d")
        if value is None:
            continue
        (whole if row["population"] == "all" else within).append(value)

    if need_medians:
        pair_median = {pair: float(np.median(values))
                       for pair, values in by_pair.items() if len(values)}
        print(f"{key}: no pair_summary.tsv, computed {len(pair_median):,} "
              f"pair medians from gene_dxy.tsv")

    index = {pop: i for i, pop in enumerate(order)}
    matrix = np.full((len(order), len(order)), np.nan)
    for (x, y), median in pair_median.items():
        if x in index and y in index:
            i, j = index[x], index[y]
            matrix[i, j] = matrix[j, i] = median

    print(f"{key}: {len(order)} lineages, {len(rel):,} dxy_rel, "
          f"{len(within):,} within-lineage D")
    return dict(order=order, matrix=matrix,
                rel=np.frombuffer(rel, dtype=np.float64),
                fst=np.frombuffer(fst, dtype=np.float64),
                within=np.frombuffer(within, dtype=np.float64),
                whole=np.frombuffer(whole, dtype=np.float64))


data = {key: load(key) for key in a.species}
missing = [k for k, v in data.items() if v is None]
if missing:
    print("no popgen results for: " + ", ".join(missing))


def blank(ax, key):
    """Keep the species' place in the layout, and say why it is empty."""
    ax.set_axis_off()
    ax.text(0.5, 0.56, display_name(key), transform=ax.transAxes, ha="center",
            va="center", fontsize=SZ["panel"], color=INK3, style="italic")
    ax.text(0.5, 0.42, "unfinished", transform=ax.transAxes, ha="center",
            va="center", fontsize=SZ["panel"], color=RED)


def panel_title(ax, key):
    ax.set_title(display_name(key), fontsize=SZ["panel"], color=INK,
                 style="italic", pad=6)


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


def save(fig, name):
    path = os.path.join(a.outdir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print("wrote", path)
    plt.close(fig)


rows_n = 2
cols_n = int(np.ceil(len(a.species) / rows_n))

# ---------------------------------------------------------------- dXY matrix
fig, axes = plt.subplots(rows_n, cols_n,
                         figsize=(7.6 * cols_n * SCALE, 7.0 * rows_n * SCALE),
                         constrained_layout=True)
for ax, key in zip(np.ravel(axes), a.species):
    entry = data[key]
    if entry is None or not len(entry["order"]):
        blank(ax, key)
        continue
    size = len(entry["order"])
    image = ax.imshow(entry["matrix"] * 1e3, cmap=SEQ_CMAP, interpolation="nearest")
    step = 1 if size <= 25 else max(1, size // 20)
    ticks = range(0, size, step)
    ax.set_xticks(list(ticks)); ax.set_yticks(list(ticks))
    labels = [entry["order"][i] for i in ticks]
    tick_size = SZ["tick"] if size <= 25 else max(4, SZ["tick"] - 3)
    ax.set_xticklabels(labels, fontsize=tick_size, color=INK3, rotation=90)
    ax.set_yticklabels(labels, fontsize=tick_size, color=INK3)
    ax.set_xlabel(f"PopPUNK lineage ({size})", fontsize=SZ["label"], color=INK2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label(r"median $d_{XY}$ ($\times 10^{-3}$)",
                  fontsize=SZ["label"], color=INK2)
    bar.ax.tick_params(colors=INK3, labelsize=SZ["tick"])
    bar.outline.set_visible(False)
    panel_title(ax, key)
save(fig, "combined_dxy_matrix.png")

# ---------------------------------------------------------------- Tajima's D
fig, axes = plt.subplots(rows_n, cols_n,
                         figsize=(7.0 * cols_n * SCALE, 5.0 * rows_n * SCALE),
                         constrained_layout=True)
for ax, key in zip(np.ravel(axes), a.species):
    entry = data[key]
    if entry is None or not len(entry["within"]):
        blank(ax, key)
        continue
    lo = min(entry["within"].min(), entry["whole"].min()) - 0.1
    hi = max(np.percentile(entry["within"], 99.9),
             np.percentile(entry["whole"], 99.9)) + 0.5
    edges = np.linspace(lo, hi, 70)
    for values, colour, label in ((entry["within"], BLUE, "within lineage"),
                                  (entry["whole"], ORANGE, "whole sample")):
        height, _ = np.histogram(values, bins=edges)
        ax.stairs(100 * height / len(values), edges, color=colour, lw=1.6,
                  fill=False, zorder=3, label=label)
        ax.stairs(100 * height / len(values), edges, color=colour, alpha=0.13,
                  fill=True, zorder=2)
    ax.axvline(0.0, color=RED, lw=1.2, ls="--", dashes=(5, 3), zorder=4)
    ax.text(0.98, 0.97,
            f"within: median {np.median(entry['within']):+.2f}\n"
            f"whole:  median {np.median(entry['whole']):+.2f}",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=SZ["ann"], color=INK2, linespacing=1.4)
    style(ax, r"Tajima's $D$", "measurements (%)")
    panel_title(ax, key)
handles = [Line2D([], [], color=BLUE, lw=2, label="within lineage"),
           Line2D([], [], color=ORANGE, lw=2, label="whole sample"),
           Line2D([], [], color=RED, lw=1.2, ls="--", label=r"$D = 0$ (neutral)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=SZ["label"], labelcolor=INK2, bbox_to_anchor=(0.5, -0.03))
save(fig, "combined_tajima.png")

# ---------------------------------------------------------------- outliers
fig, axes = plt.subplots(2, len(a.species),
                         figsize=(4.6 * len(a.species) * SCALE, 8.4 * SCALE),
                         constrained_layout=True)
axes = np.atleast_2d(axes)
for column, key in enumerate(a.species):
    entry = data[key]
    top, bottom = axes[0, column], axes[1, column]
    if entry is None or not len(entry["rel"]):
        blank(top, key)
        bottom.set_axis_off()
        continue

    log_rel = np.log10(entry["rel"])
    edges = np.linspace(-4.2, 2.5, 80)
    height, _ = np.histogram(log_rel, bins=edges)
    top.stairs(100 * height / len(log_rel), edges, color=BLUE, lw=1.6,
               fill=False, zorder=3)
    top.stairs(100 * height / len(log_rel), edges, color=BLUE, alpha=0.13,
               fill=True, zorder=2)
    top.axvline(0.0, color=RED, lw=1.2, ls="--", dashes=(5, 3), zorder=4)
    top.text(0.03, 0.97, f"n = {len(entry['rel']):,}\n"
                         f"{100 * (entry['rel'] < 0.1).mean():.1f}% below 0.1x",
             transform=top.transAxes, va="top", ha="left",
             fontsize=SZ["ann"], color=INK2, linespacing=1.4)
    style(top, r"$\log_{10}(d_{XY}$ / pair median$)$",
          "gene-pair values (%)" if column == 0 else "")
    panel_title(top, key)

    edges = np.linspace(min(0.0, entry["fst"].min()), 1.0, 60)
    height, _ = np.histogram(entry["fst"], bins=edges)
    bottom.stairs(100 * height / len(entry["fst"]), edges, color=AQUA, lw=1.6,
                  fill=False, zorder=3)
    bottom.stairs(100 * height / len(entry["fst"]), edges, color=AQUA,
                  alpha=0.13, fill=True, zorder=2)
    bottom.text(0.03, 0.97, f"median {np.median(entry['fst']):.3f}\n"
                            f"{100 * (entry['fst'] < 0.1).mean():.1f}% below 0.1",
                transform=bottom.transAxes, va="top", ha="left",
                fontsize=SZ["ann"], color=INK2, linespacing=1.4)
    style(bottom, r"Hudson's $F_{ST}$",
          "gene-pair values (%)" if column == 0 else "")
save(fig, "combined_dxy_outliers.png")
