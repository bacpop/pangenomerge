# imports

import argparse
import sys
import re
import networkx as nx
import scipy as scipy
import os
import itertools
import pandas as pd
import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import rand_score,mutual_info_score,adjusted_rand_score,adjusted_mutual_info_score
from pathlib import Path
from Bio import SeqIO
import logging
from itertools import combinations
from edlib import align
from collections import defaultdict
import gc
import multiprocessing as mp
import subprocess
import shutil
import glob as glob_module

# import custom functions
from pangenomerge.custom_functions.manipulate_seqids import indSID_to_allSID, get_seqIDs_in_nodes, dict_to_2d_array
from pangenomerge.custom_functions.graph_similarity import graph_similarity_scores
from pangenomerge.custom_functions.run_mmseqs import run_mmseqs_search, mmseqs_createdb, mmseqs_concatdbs
from pangenomerge.panaroo_functions.load_graphs import load_graphs
from pangenomerge.panaroo_functions.write_gml_metadata import format_metadata_for_gml
from pangenomerge.panaroo_functions.context_search import collapse_families, single_linkage, collapse_spurious_paralogs
from pangenomerge.panaroo_functions.merge_nodes import merge_node_cluster, gen_edge_iterables, gen_node_iterables, iter_del_dups, initial_graph_merge
from pangenomerge.custom_functions.relabel_nodes import relabel_nodes_preserve_attrs,sync_names
from pangenomerge.custom_functions.sqlite import sqlite_connect, sqlite_init_schema, sqlite_create_indexes, add_metadata_to_sqlite, add_isolate_names_to_sqlite, add_clusters_to_sqlite, load_metadata_from_sqlite
from pangenomerge.generate_output import generate_summary_statistics
from pangenomerge.custom_functions.representative_selection import build_allele_lookup
from pangenomerge.panaroo_functions.context_search import final_representative_merge

from .__init__ import __version__

# MUST USE FORK TO ENSURE PARALLEL COMPUTATION OF COLLAPSE SCORES DOESNT COPY GRAPH OBJECT -- LINUX DEFAULT; WINDOWS/MAC BEWARE!

def get_options():
    description = 'Merges two or more Panaroo pangenome gene graphs, or iteratively updates an existing graph.'
    parser = argparse.ArgumentParser(description=description,
                                    prog='pangenomerge')

    IO = parser.add_argument_group('Input and output options')
    IO.add_argument('--mode',
                    default='run',
                    choices=['run', 'test'],
                    help='Run pangenome gene graph merge ("run") or calculate clustering accuracy metrics for merge ("test"). '
                        '[Default = Run] ')
    IO.add_argument('--outdir',
                    required=True,
                    default=None,
                    help='Output directory.')
    IO.add_argument('--component-graphs',
                    dest='component_graphs',
                    default=None,
                    required=False,
                    help='Path to a text file with one Panaroo output directory path per line. \
                    Each directory must contain final_graph.gml and pan_genome_reference.fa. If running in test mode, \
                    must also contain gene_data.csv. Graphs will be merged in the order presented in the file.')
    IO.add_argument('--iterative',
                    dest='iterative',
                    default=None,
                    required=False,
                    help='Tab-separated list of GFFs and their sample IDs for iterative updating of the graph. \
                    Use only for single samples or sets of samples too diverse to create an initial pangenome. \
                    Samples will be merged in the order presented in the file.')
    IO.add_argument('--fast',
                    dest='fast',
                    default=False,
                    action='store_true',
                    help='Test mode only: skip AMI. Its expected-mutual-information term \
                    dominates the cost on large gene sets; ARI, RI, the post-clustering and \
                    attainability-corrected scores, homogeneity and completeness are all \
                    still reported.')
    IO.add_argument('--graph-all',
                    dest='graph_all',
                    default=None,
                    help='Path to Panaroo output directory of pangenome gene graph created from all samples in component-graphs. \
                    Only required for the test case, where it is used as the ground truth.')
    IO.add_argument('--metadata-in-graph',
                    dest='keep_metadata_in_graph',
                    default=False,
                    type=bool,
                    required=False,
                    help='Retains metadata in the final graph GML (in addition to the SQLite database). Dramatically increases \
                    runtime and memory consumption. Not recommended with >10k isolates.')
    IO.add_argument('--include-clusters',
                    dest='include_clusters',
                    default=None,
                    required=False,
                    help='Path to a PopPUNK clusters CSV (header: Taxon,Cluster). \
                    If provided, populates isolate_names.poppunk_cluster in the SQLite. \
                    Sample names in the CSV must match those in the input graphs; isolates \
                    without a match are warned and stored as NULL.')

    parameters = parser.add_argument_group('Parameters')
    parameters.add_argument('--family-threshold',
                    dest='family_threshold',
                    default=0.7,
                    type=float,
                    required=False,
                    help='Sequence identity threshold for putative spurious paralogs. Default: 0.7')
    parameters.add_argument('--context-threshold',
                    dest='context_threshold',
                    default=0.7,
                    type=float,
                    required=False,
                    help='Sequence identity threshold for neighbors of putative spurious paralogs. Default: 0.7')
    parameters.add_argument('--frameshift-coverage',
                    dest='frameshift_coverage',
                    default=0.25,
                    type=float,
                    required=False,
                    help='Coverage threshold for frameshift second-pass merge (catches truncated/frameshifted variants). Default: 0.25')
    parameters.add_argument('--frameshift-identity',
                    dest='frameshift_identity',
                    default=0.90,
                    type=float,
                    required=False,
                    help='Sequence identity threshold for frameshift second-pass merge. Default: 0.90')
    parameters.add_argument('--context-search-iterations',
                    dest='context_search_iterations',
                    default=-1,
                    type=int,
                    required=False,
                    help='Max outer rounds of the alternating context/frameshift merge loop (family inner loop always runs to its own fixed point). -1 = run until no new pairs merge. Default: -1')

    parameters.add_argument('--rep-merge-base',
                    dest='rep_merge_base',
                    default=None,
                    choices=['modal', 'stratified', 'allunique'],
                    required=False,
                    help='Enable the representative-based cross-graph rescue step using this '
                        'allele-selection strategy for BASE-graph nodes. Requires --rep-merge-query. '
                        'modal = most common allele; stratified = most common allele per >15%% length bin; '
                        'allunique = all distinct alleles. Default: off.')
    parameters.add_argument('--rep-merge-query',
                    dest='rep_merge_query',
                    default=None,
                    choices=['modal', 'stratified', 'allunique'],
                    required=False,
                    help='Allele-selection strategy for QUERY-graph nodes in the representative '
                        'rescue step. Requires --rep-merge-base. Default: off.')
    parameters.add_argument('--rep-merge-position',
                    dest='rep_merge_position',
                    default='loop',
                    choices=['loop', 'final', 'both'],
                    required=False,
                    help='When to run the representative rescue step. "loop" (default) runs it '
                        'inside each pairwise merge, where the graph holds only the graphs merged '
                        'so far. "final" runs it once after every component graph is in, so the '
                        'synteny gate sees complete neighbourhoods -- a pair whose shared '
                        'neighbours arrive with a later graph cannot pass in loop mode. "both" '
                        'runs it in each pass and again at the end.')
    parameters.add_argument('--rep-merge-context',
                    dest='rep_merge_context',
                    default='representative',
                    choices=['representative', 'centroid'],
                    required=False,
                    help='Which sequences the representative step\'s synteny gate scores '
                        'neighbours with. "representative" (default) uses the same alleles the '
                        'step proposes pairs from, under the same --rep-merge-base/query '
                        'strategy and seqtype, taking each node pair\'s best-matching '
                        'representative. "centroid" reuses the longest-protein centroid lookup '
                        'from the main pass, which makes the gate identical for every strategy '
                        '(the behaviour before this option existed; kept for ablation).')
    parameters.add_argument('--rep-merge-seqtype',
                    dest='rep_merge_seqtype',
                    default='protein',
                    choices=['protein', 'dna'],
                    required=False,
                    help='Sequence type for the representative rescue step. Default: protein.')

    other = parser.add_argument_group('Other options')
    other.add_argument('--threads',
                    dest="threads",
                    default=1,
                    type=int,
                    help='Number of threads')
    other.add_argument('--sqlite-cache',
                    dest="sqlite_cache",
                    default=2000,
                    type=int,
                    required=False,
                    help='Desired size of SQLite cache expressed in KB. Diminishing returns above 1 GB (1048576 KB). Defaults to 2000 KB.')
    other.add_argument('--debug',
                    action='store_true',
                    help="Set logging to 'debug' instead of 'info' (default)")
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    # parse command line arguments
    options = get_options()

    if options.component_graphs is None and options.iterative is None:
        logging.critical("Specifying either --component-graphs or --iterative is required!")
    if options.mode == 'test' and options.graph_all is None:
        logging.critical("Specifying --graph-all is required for test mode!")
    if options.frameshift_identity < options.family_threshold:
        logging.warning(f"--frameshift-identity ({options.frameshift_identity}) is below --family-threshold ({options.family_threshold}); "
                        f"frameshift pass would accept weaker hits than the family pass.")
    if options.include_clusters is not None and not Path(options.include_clusters).exists():
        logging.critical(f"--include-clusters file does not exist: {options.include_clusters}")
    if (options.rep_merge_base is None) != (options.rep_merge_query is None):
        logging.critical("--rep-merge-base and --rep-merge-query must be specified together!")

    # check whether metadata should be left in merged graph and provide warning
    if options.keep_metadata_in_graph is True:
        logging.warning(f"Metadata will be retained in the final graph GML (in addition to the SQLite database). \
            Dramatically increases runtime and memory consumption. Not recommended with >10k isolates. \
            To remove this warning, run without '--metadata-in-graph'.")
    if options.mode == 'test' and options.keep_metadata_in_graph is not True:
        logging.warning(f"Keeping metadata in graph is required for test mode! Running with --metadata-in-graph...")
        options.keep_metadata_in_graph = True

    # set logging to 'debug' or 'info' (default)
    if options.debug:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ### create outdir

    # first remove any existing files in mmseqs outdir (can cause problems)
    subprocess.run(f'rm -rf {str(options.outdir)}/mmseqs_tmp/*', shell=True, check=True, capture_output=True)

    mmseqs_dir = Path(options.outdir) / "mmseqs_tmp"
    mmseqs_dir.mkdir(parents=True, exist_ok=True)

    intermediate_dir = Path(options.outdir) / "intermediate_graphs"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # this will always point to the current combined pangenome db
    base_db = None

    # define sqlite database path
    sqlite_path = Path(options.outdir) / "pangenome_metadata.sqlite"

    # delete any existing sqlite database
    if sqlite_path.exists():
        logging.info(f"Removing existing SQLite database: {sqlite_path}")
        sqlite_path.unlink()

    # create new sqlite database
    con = sqlite_connect(database=sqlite_path, sqlite_cache=options.sqlite_cache)
    sqlite_init_schema(con)

    ### read in two graphs

    graph_files = pd.read_csv(options.component_graphs, sep='\t', header=None)
    n_graphs = int(len(graph_files))
    graph_count = 0

    # component-graph clustering (C) for reference-corrected ARI/AMI (test mode):
    # seqID (in graph_all clustering namespace) -> unique panaroo cluster of origin
    component_cluster_of_seqid = {}

    # build allele lookup once for the representative-based rescue step (if enabled)
    allele_lookup = None
    if options.rep_merge_base is not None and options.rep_merge_query is not None:
        logging.info(
            f"Representative rescue step enabled (base={options.rep_merge_base}, "
            f"query={options.rep_merge_query}, seqtype={options.rep_merge_seqtype}, "
            f"context={options.rep_merge_context}).")
        graph_dirs = [str(graph_files.iloc[i][0]) for i in range(n_graphs)]
        allele_lookup = build_allele_lookup(
            graph_dirs=graph_dirs,
            mode=options.mode,
            graph_all_dir=options.graph_all,
            seqtype=options.rep_merge_seqtype,
        )
        logging.info(f"Allele lookup built: {len(allele_lookup)} sequences.")

    for graph in range(1, int(n_graphs)):
        
        if graph_count == 0:
            graph_file_1 = str(Path(graph_files.iloc[0][0]) / "final_graph.gml")
            graph_file_2 = str(Path(graph_files.iloc[1][0]) / "final_graph.gml")
        else:
            graph_file_1 = str(intermediate_dir / f"merged_graph_{graph_count}.gml")
            graph_file_2 = str(Path(graph_files.iloc[int(graph_count+1)][0]) / "final_graph.gml")

        logging.info(f"Beginning iteration {graph_count+1} of {n_graphs-1}...")
        logging.info(f"graph_file_1: {graph_file_1}")
        logging.info(f"graph_file_2: {graph_file_2}")

        # keep merged graph in memory instead of repeatedly reading in
        if graph_count == 0:
            graph_1, isolate_names, id_mapping = load_graphs([graph_file_1])
            graph_1 = graph_1[0]
            add_isolate_names_to_sqlite(con, graph_id=graph_count+1, isolate_names=isolate_names)
        else:
            graph_1 = merged_graph

        graph_2, isolate_names, id_mapping = load_graphs([graph_file_2])
        graph_2 = graph_2[0]
        add_isolate_names_to_sqlite(con, graph_id=graph_count+2, isolate_names=isolate_names)

        if options.mode == 'test':

            logging.info(f"Relabeling seqIDs to enable ARI/AMI calculation...")

            ### match clustering_ids from overall run to clustering_ids from individual runs using annotation_ids (test only)

            gene_data_all = pd.read_csv(str(Path(options.graph_all) / "gene_data.csv"))
            gene_data_g2 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count+1)][0]) / "gene_data.csv"))

            if graph_count == 0:
                logging.debug("Applying gene data...")
                gene_data_g1 = pd.read_csv(str(Path(graph_files.iloc[int(graph_count)][0]) / "gene_data.csv"))
            else:
                gene_data_g1 = None
                # not necessary because merged graph already has gene_all seqIDs mapped

            # rename column
            gene_data_all = gene_data_all.rename(columns={'clustering_id': 'clustering_id_all'})
            if graph_count == 0:
                logging.debug("Applying rename...")
                gene_data_g1 = gene_data_g1.rename(columns={'clustering_id': 'clustering_id_indiv'})
            
            gene_data_g2 = gene_data_g2.rename(columns={'clustering_id': 'clustering_id_indiv'})

            # first match by annotation ids:
            if graph_count == 0:
                logging.debug("Applying match...")
                matches_g1 = gene_data_all[['annotation_id', 'clustering_id_all']].merge(
                    gene_data_g1[['annotation_id', 'clustering_id_indiv']],
                    on='annotation_id',
                    how='left'
                )

            matches_g2 = gene_data_all[['annotation_id', 'clustering_id_all']].merge(
                gene_data_g2[['annotation_id', 'clustering_id_indiv']],
                on='annotation_id',
                how='left'
            )

            # now drop rows where the individual seqID wasn't observed (or there's no corresponding seqID from all)
            if graph_count == 0:
                logging.debug("Applying dropna...")
                matches_g1 = matches_g1.dropna()
            matches_g2 = matches_g2.dropna()

            # convert to dict for faster lookup than with loc
            if graph_count == 0:
                logging.debug("Applying gidmap...")
                gid_map_g1 = dict(zip(matches_g1['clustering_id_indiv'], matches_g1['clustering_id_all']))
            gid_map_g2 = dict(zip(matches_g2['clustering_id_indiv'], matches_g2['clustering_id_all']))

            # apply to graphs:
            if graph_count == 0:
                logging.debug("Applying ind...")
                graph_1 = indSID_to_allSID(graph_1, gid_map_g1)
            graph_2 = indSID_to_allSID(graph_2, gid_map_g2)

            # record the panaroo cluster each seqID came from (baseline clustering C):
            # each component-graph node is its own cluster, made globally unique by graph index
            if graph_count == 0:
                for node in graph_1.nodes():
                    for sid in graph_1.nodes[node].get("seqIDs", []):
                        component_cluster_of_seqid[sid] = f"{node}_g{graph_count+1}"
            for node in graph_2.nodes():
                for sid in graph_2.nodes[node].get("seqIDs", []):
                    component_cluster_of_seqid[sid] = f"{node}_g{graph_count+2}"

        # debug statement...
        logging.debug(f"--- MERGE {graph_count+1} ---")
        logging.debug(f"graph_1 nodes: {list(graph_1.nodes())[:20]} ... total {len(graph_1.nodes())}")
        logging.debug(f"graph_2 nodes: {list(graph_2.nodes())[:20]} ... total {len(graph_2.nodes())}")

        ### map nodes from ggcaller graphs to the COG labels in the centroid from pangenome

        ### run mmseqs2 to identify matching COGs

        # read in pangenome reference from graph 1
        if graph_count == 0:
            pangenome_reference_g1 = str(Path(graph_files.iloc[0][0]) / "pan_genome_reference.fa")
        else:
            pangenome_reference_g1 = str(Path(options.outdir) / f"pan_genome_reference_{graph_count}.fa")

        # read in pangenome reference from graph 2
        pangenome_reference_g2 = str(Path(graph_files.iloc[int(graph_count+1)][0]) / "pan_genome_reference.fa")

        # Some nodes in panaroo's final_graph.gml are absent from pan_genome_reference.fa.
        # Re-read the raw (un-relabeled) gml and append the missing nodes' longest DNA
        # sequence so the mmseqs database covers the full graph.
        Path(options.outdir, "mmseqs_tmp").mkdir(parents=True, exist_ok=True)

        if graph_count == 0:
            raw_gml_g1 = str(Path(graph_files.iloc[0][0]) / "final_graph.gml")
            raw_g1 = nx.read_gml(raw_gml_g1)
            present_g1 = {rec.id for rec in SeqIO.parse(pangenome_reference_g1, "fasta")}
            augmented_g1 = str(Path(options.outdir) / "mmseqs_tmp" / "pan_genome_reference_g1_augmented.fa")
            with open(pangenome_reference_g1) as src, open(augmented_g1, "w") as dst:
                shutil.copyfileobj(src, dst)
                for n in raw_g1.nodes():
                    name = raw_g1.nodes[n]['name']
                    if name in present_g1:
                        continue
                    seqs = raw_g1.nodes[n]["dna"]
                    if isinstance(seqs, str):
                        seq = max(seqs.split(";"), key=len)
                    else:
                        seq = max(seqs, key=len)
                    dst.write(f">{name}\n{seq}\n")
            pangenome_reference_g1 = augmented_g1

        raw_gml_g2 = str(Path(graph_files.iloc[int(graph_count+1)][0]) / "final_graph.gml")
        raw_g2 = nx.read_gml(raw_gml_g2)
        present_g2 = {rec.id for rec in SeqIO.parse(pangenome_reference_g2, "fasta")}
        augmented_g2 = str(Path(options.outdir) / "mmseqs_tmp" / f"pan_genome_reference_g2_augmented_{graph_count+1}.fa")
        with open(pangenome_reference_g2) as src, open(augmented_g2, "w") as dst:
            shutil.copyfileobj(src, dst)
            for n in raw_g2.nodes():
                name = raw_g2.nodes[n]['name']
                if name in present_g2:
                    continue
                seqs = raw_g2.nodes[n]["dna"]
                if isinstance(seqs, str):
                    seq = max(seqs.split(";"), key=len)
                else:
                    seq = max(seqs, key=len)
                dst.write(f">{name}\n{seq}\n")
        pangenome_reference_g2 = augmented_g2

        # debug statement...
        logging.debug(f"pangenome reference g1: {pangenome_reference_g1}")
        logging.debug(f"pangenome reference g2: {pangenome_reference_g2}")

        # info statement...
        logging.info("Creating MMSeqs2 database(s)...")

        ### create mmseqs databases for faster search

        # define paths for new databases
        base_db = str(Path(options.outdir) / "mmseqs_tmp" / f"pan_genome_db_{graph_count+1}")
        temp_db = str(Path(options.outdir) / "mmseqs_tmp" / f"temp_db")
        
        # always create new AA database for new graph
        mmseqs_createdb(fasta=pangenome_reference_g2, outdb=temp_db, threads=options.threads, nt2aa=True)

        # create AA database for base graph on first iter only
        if graph_count == 0:
            mmseqs_createdb(fasta=pangenome_reference_g1, outdb=base_db, threads=options.threads, nt2aa=True)

        # info statement...
        logging.info("Running MMSeqs2...")

        ### run mmseqs on the two pangenome references
        run_mmseqs_search(
            targetdb=base_db,
            querydb=temp_db,
            resultdb = str(Path(options.outdir) / "mmseqs_tmp" / "resultdb"),
            resultm8 = str(Path(options.outdir) / "mmseqs_tmp" / "mmseqs_clusters.m8"),
            tmpdir = str(Path(options.outdir) / "mmseqs_tmp"),
            threads=options.threads,
            fident=0.98,
            coverage=0.95
        )
        
        # info statement...
        logging.info("MMSeqs2 complete. Reading and filtering results...")

        # read mmseqs results
        # each "group_" refers to the centroid of that group in the pan_genomes_reference.fa
        mmseqs = pd.read_csv(str(Path(options.outdir) / "mmseqs_tmp" / "mmseqs_clusters.m8"), sep='\t')

        # Test mode only: canonical row order, so the merge is reproducible. mmseqs returns
        # the same hits in a thread-dependent order (verified: two 8-thread searches on
        # identical inputs gave byte-different m8 files that matched once sorted). This
        # search decides which graph_2 nodes are renamed onto graph_1, so its ordering seeds
        # every later step. Run mode skips the sort -- reproducibility is only needed when
        # comparing runs against a ground truth, and the hit table can be very large.
        if options.mode == "test":
            mmseqs = mmseqs.sort_values(["query", "target", "fident"],
                                        kind="mergesort").reset_index(drop=True)

        ### match hits from mmseqs

        # change the second graph node names to the first graph node names for nodes that match according to mmseqs

        # make sure metrics are numeric
        mmseqs["fident"] = pd.to_numeric(mmseqs["fident"], errors='coerce')
        mmseqs["evalue"] = pd.to_numeric(mmseqs["evalue"], errors='coerce')
        mmseqs["tlen"] = pd.to_numeric(mmseqs["tlen"], errors='coerce')
        mmseqs["qlen"] = pd.to_numeric(mmseqs["qlen"], errors='coerce')

        # define length difference
        max_len = np.maximum(mmseqs['tlen'], mmseqs['qlen'])
        mmseqs["len_dif"] = 1 - (np.abs(mmseqs["tlen"] - mmseqs["qlen"]) / max_len)

        # filter for fraction nt identity >= 98% (global) and length difference <= 5%
        mmseqs = mmseqs[(mmseqs["fident"] >= 0.98) & (mmseqs["len_dif"] >= 0.95)].copy()

        ### iterate over query with each unique value of query, and pick the match with the highest fident, then highest len_dif (see calculation)
        # if still multiple matches, pick the first one

        # sort by fident (highest first), len_dif (highest first -- see calculation), and evalue (lowest first)
        mmseqs_sorted = mmseqs.sort_values(by=["fident", "len_dif", "evalue"], ascending=[False, False, True],)

        # debug statement...
        logging.debug(f" {len(mmseqs_sorted)} one-to-one hits.")
        logging.debug(f"{mmseqs_sorted}")

        # only keep the first occurrence per unique query (highest fident, lowest length difference, then smallest evalue if tie)
        mmseqs_filtered = mmseqs_sorted.drop_duplicates(subset=["query"], keep="first")
        mmseqs_filtered = mmseqs_filtered.drop_duplicates(subset=["target"], keep="first") # test if dropping target vs. query duplicates first changes results

        # debug statement...
        logging.debug(f"Filtered to {len(mmseqs_filtered)} one-to-one hits.")
        logging.debug(f"{mmseqs_filtered}")
        dups_query = mmseqs_filtered["query"].duplicated().sum()
        dups_target = mmseqs_filtered["target"].duplicated().sum()
        logging.debug(f"Remaining duplicates — query: {dups_query}, target: {dups_target}")

        # info statement...
        logging.info("Hits filtered. Mapping between graphs...")

        # in mmseqs, the first graph entered (in this case graph_1) is the target and the second entered (in this case graph_2) is the query
        # so graph_1 is our target in mmseqs and the basegraph in the tokenized merge

        # when iterating over graph_2 to append to graph_1, we want to match nodes according to their graph_1 identity
        # so we need to replace all graph_2 nodes with graph_1 node ids

        ### THE NAME ("group_1") AND THE LABEL ('484') ARE DIFFERENT AND A NUMERIC STRING WILL CALL THE LABEL (not index)

        # the groups are not the same across the two graphs! we match by mmseqs (that's the whole point of this)
        # this chunk is just changing the node name from an integer to the group label of that node (which is originally just
        # metadata within the graph)
        # it doesn't map anything between the two graphs

        # change integer node labels into group labels from 'name' attribute (graph 1)
        if graph_count == 0:

            # map 'name' attribute to integer node label
            mapping_groups_1 = dict()
            for node in graph_1.nodes():
                node_group = graph_1.nodes[node].get("name", "error")
                #logging.debug(f"graph: 1, node_index_id: {node}, node_group_id: {node_group}")
                mapping_groups_1[node] = str(node_group)

            # debug statement...
            logging.debug("mapping_groups_1 sample:")
            for k,v in list(mapping_groups_1.items())[:10]:
                logging.debug(f"  {k} to {v}")

            # relabel nodes
            groupmapped_graph_1 = relabel_nodes_preserve_attrs(graph_1, mapping_groups_1)

            # debug statement...
            logging.debug(f"After relabel_nodes_preserve_attrs() graph_1 node sample: {list(groupmapped_graph_1.nodes())[:10]}")
        
        else:

            # map 'name' attribute to integer node label
            mapping_groups_1 = dict()
            for node in graph_1.nodes():
                node_group = graph_1.nodes[node].get("name", "error")
                #logging.debug(f"graph: 1, node_index_id: {node}, node_group_id: {node_group}")
                mapping_groups_1[node] = str(node_group)

            # debug statement...
            logging.debug("mapping_groups_1 sample:")
            for k,v in list(mapping_groups_1.items())[:10]:
                logging.debug(f"  {k} to {v}")
            
            # relabel nodes
            groupmapped_graph_1 = relabel_nodes_preserve_attrs(graph_1, mapping_groups_1)

            # debug statement...
            logging.debug(f"After relabel_nodes_preserve_attrs() graph_1 node sample: {list(groupmapped_graph_1.nodes())[:10]}")

        # change integer node labels into group labels from 'name' attribute (graph 2)
        # map 'name' attribute to integer node label
        mapping_groups_2 = dict()
        for node in graph_2.nodes():
            node_group = graph_2.nodes[node].get("name", "error")
            #logging.debug(f"graph: 2, node_index_id: {node}, node_group_id: {node_group}")
            mapping_groups_2[node] = str(node_group)

        # debug statement...
        logging.debug("mapping_groups_2 sample:")
        for k,v in list(mapping_groups_2.items())[:10]:
            logging.debug(f"  {k} to {v}")

        # relabel nodes
        groupmapped_graph_2 = relabel_nodes_preserve_attrs(graph_2, mapping_groups_2)

        # debug statement...
        logging.debug(f"After relabel_nodes_preserve_attrs() graph_2 node sample: {list(groupmapped_graph_2.nodes())[:10]}")

        ### map filtered mmseqs2 hits to mapping of nodes between graphs

        # mapping format: dictionary with old labels (graph_2/query groups) as keys and new labels (graph_1/target) as values

        # convert df to dictionary with "query" as keys and "target" as values
        # this maps groups from graph_1 to groups from graph_2
        mapping = dict(zip(mmseqs_filtered["query"], mmseqs_filtered["target"]))

        ### to avoid matching nodes from query that have the same group_id but are not the same:
        # append all nodes in target graph with _target
        # append all target nodes in query graph with _target (for later matching)

        # this appends _target to values (graph_1/target groups)
        mapping = {key: f"{value}_target" for key, value in mapping.items()}

        # relabel query graph from old labels (keys) to new labels (values, the _target-appended graph_1 groups)
        # some of these will just be OG group_XXX (not target-appended) from graph 2; the rest will be group_XXX_target from graph 1
        relabeled_graph_2 = relabel_nodes_preserve_attrs(groupmapped_graph_2, mapping)
        relabeled_graph_2 = sync_names(relabeled_graph_2)

        # append _target to ALL nodes in target graph 
        mapping_target = dict(zip(groupmapped_graph_1.nodes, groupmapped_graph_1.nodes))
        mapping_target = {key: f"{value}_target" for key, value in mapping_target.items()}
        relabeled_graph_1 = relabel_nodes_preserve_attrs(groupmapped_graph_1, mapping_target)
        relabeled_graph_1 = sync_names(relabeled_graph_1)

        # reduce memory by removing intermediate files
        for name in [
            "mmseqs", "mmseqs_sorted", "mmseqs_filtered",
            "mapping_groups_1", "mapping_groups_2",
            "groupmapped_graph_1", "groupmapped_graph_2",
            "graph_1", "graph_2",
        ]:
            if name in locals():
                del locals()[name]
        gc.collect()

        # debug statement...
        logging.debug("MMseqs mapping (query to target_target):")
        for k,v in list(mapping.items())[:10]:
            logging.debug(f"  {k} to {v}")

        # debug statement...
        logging.debug(f"Relabeled graph_2 nodes sample: {list(relabeled_graph_2.nodes())[:10]}")
        logging.debug(f"Relabeled graph_1 (_target appended) nodes sample: {list(relabeled_graph_1.nodes())[:10]}")

        # read in graph of all isolates (for ARI/AMI calculation)
        if options.mode == 'test':
            graph_all = [str(Path(options.graph_all) / "final_graph.gml")]
            graph_all, isolate_names, id_mapping = load_graphs(graph_all)
            graph_all = graph_all[0]

        # info statement...
        logging.info("Updating graph metadata to prepare for merge...")
        
        ### add suffix to relevant metadata to be able to identify which graph they refer to later
        # if other bits are too slow, replacing looping over nodes with the nodes.values method shown here

        # NODES

        for node_data in relabeled_graph_2.nodes.values():

            node_data['members'] = [f"{member}_g{graph_count+2}" for member in node_data['members']] # list

            node_data['genomeIDs'] = ";".join(node_data['members']) # str

            if options.mode != 'test':

                seqid_set = {f"{seqid}{f'_g{graph_count+2}'}" for seqid in node_data['seqIDs']}
                node_data['seqIDs'] = seqid_set # set

                geneids = node_data['geneIDs'].split(";")
                geneids = [f"{gid}_g{graph_count+2}" for gid in geneids]
                node_data['geneIDs'] = ";".join(geneids) # str

                node_data['longCentroidID'].append(f'from_g{graph_count+2}') #list

                node_data['maxLenId'] = str(node_data['maxLenId']) + f'_g{graph_count+2}' # int

                node_data['centroid'] = [f"{centroid}_g{graph_count+2}" for centroid in node_data['centroid']] # list

        if graph_count == 0: # if first iteration, do the same to the base graph

            for node_data in relabeled_graph_1.nodes.values():

                node_data['members'] = [f"{member}_g{graph_count+1}" for member in node_data['members']] # list

                node_data['genomeIDs'] = ";".join(node_data['members']) # str

                if options.mode != 'test':

                    seqid_set = {f"{seqid}{f'_g{graph_count+1}'}" for seqid in node_data['seqIDs']}
                    node_data['seqIDs'] = seqid_set # set

                    node_data['longCentroidID'].append(f'from_g{graph_count+1}') #list

                    node_data['centroid'] = [f"{centroid}_g{graph_count+1}" for centroid in node_data['centroid']] # list

                    node_data['maxLenId'] = str(node_data['maxLenId']) + f'_g{graph_count+1}' # int

                    geneids = node_data['geneIDs'].split(";")
                    geneids = [f"{gid}_g{graph_count+1}" for gid in geneids]
                    node_data['geneIDs'] = ";".join(geneids) # str

        # EDGES

        # edge attributes: size (n members), members (list), genomeIDs (semicolon-separated string)

        for edge in relabeled_graph_2.edges:
            
            # members
            relabeled_graph_2.edges[edge]['members'] = [str(member) + f"_g{graph_count+2}" for member in relabeled_graph_2.edges[edge]['members']] # add underscore w graph count+2 to members of g2

            # genome IDs (assuming genomeIDs are always the same as members):
            relabeled_graph_2.edges[edge]['genomeIDs'] = ";".join(relabeled_graph_2.edges[edge]['members'])

            # size
            relabeled_graph_2.edges[edge]['size'] = str(len(relabeled_graph_2.edges[edge]['members']))

        if graph_count == 0: # if first iteration, do the same to the base graph

            for edge in relabeled_graph_1.edges:
                
                # members
                relabeled_graph_1.edges[edge]['members'] = [str(member) + f"_g{graph_count+1}" for member in relabeled_graph_1.edges[edge]['members']] # add underscore with graph count+1 to members of g1

                # genome IDs (assuming genomeIDs are always the same as members):
                relabeled_graph_1.edges[edge]['genomeIDs'] = ";".join(relabeled_graph_1.edges[edge]['members'])

                # size
                relabeled_graph_1.edges[edge]['size'] = str(len(relabeled_graph_1.edges[edge]['members']))

        ### merge graphs

        # info statement...
        logging.info("Beginning graph merge...")

        merged_graph = initial_graph_merge(relabeled_graph_1, relabeled_graph_2, graph_count)

        # reduce memory by removing intermediate files
        del relabeled_graph_1  # delete variable name to ensure it can't be accidentally used in future
        if "relabeled_graph_2" in locals():
            del relabeled_graph_2
        gc.collect()

        # info statement...
        logging.info("Collapsing spurious paralogs...")

        # debug statement...
        logging.debug(f"Before collapse: {len(merged_graph.nodes())} nodes")

        collapse_spurious_paralogs(
            merged_graph, base_db, options, options.outdir,
            family_threshold=float(options.family_threshold),
            context_threshold=float(options.context_threshold),
            frameshift_identity=float(options.frameshift_identity),
            frameshift_coverage=float(options.frameshift_coverage),
            context_search_iterations=int(options.context_search_iterations),
            allele_lookup=allele_lookup,
            rep_merge_base=(options.rep_merge_base
                            if options.rep_merge_position in ('loop', 'both') else None),
            rep_merge_query=(options.rep_merge_query
                             if options.rep_merge_position in ('loop', 'both') else None),
            rep_merge_seqtype=options.rep_merge_seqtype,
            rep_merge_context=options.rep_merge_context,
            query_suffix=f"_g{graph_count+2}",
        )
        
        # --- final representative pass: once every component graph is in the graph ---
        # Runs on the last merge iteration only, so every COG is present and the context
        # gate is evaluated against complete neighbourhoods.
        if (options.rep_merge_base is not None
                and options.rep_merge_position in ('final', 'both')
                and allele_lookup is not None
                and graph == n_graphs - 1):
            logging.info("Running final representative pass on the complete graph "
                         f"({len(merged_graph.nodes())} nodes)...")
            n_final = final_representative_merge(
                merged_graph, allele_lookup,
                base_strategy=options.rep_merge_base,
                seqtype=options.rep_merge_seqtype,
                family_threshold=float(options.family_threshold),
                context_threshold=float(options.context_threshold),
                outdir=options.outdir,
                threads=options.threads,
                deterministic=(options.mode == 'test'),
                rep_merge_context=options.rep_merge_context,
            )
            logging.info(f"Final representative pass: {n_final} pairs merged; "
                         f"{len(merged_graph.nodes())} nodes remain.")
            for node in merged_graph:
                merged_graph.nodes[node]["degrees"] = int(merged_graph.degree[node])


        # info statement...
        logging.info("Merge complete. Preparing attribute metadata for export...")

        ### clean node names in merged graph
        if graph_count == 0:
            # remove _target suffix, add graph count 
            # (relabel node from graph_1 group_xxx_target to group_xxx_gx)
            mapping = {}
            for node_id, node_data in merged_graph.nodes(data=True):
                name = node_data.get('name', '')
                if '_target' in name:
                    new_name = re.sub(r'_target.*$', f'_g{graph_count+1}', name)
                    mapping[node_id] = new_name
                    #logging.debug(f"Changed: {name} to {new_name}")
                if '_target' not in name:
                    #logging.debug(f"Retained: {name}")
                    continue
            merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping)
            merged_graph = sync_names(merged_graph)
        else:
            # remove _target suffix
            # (relabel target nodes node from group_xxx_x_target to group_xxx_x)
            mapping = {}
            for node_id, node_data in merged_graph.nodes(data=True):
                name = node_data.get('name', '')
                if '_target' in name:
                    new_name = re.sub(r'_target.*$', "", name)
                    mapping[node_id] = new_name
                    #logging.debug(f"Changed: {name} to {new_name}")
                if '_target' not in name:
                    #logging.debug(f"Retained: {name}")
                    continue
            merged_graph = relabel_nodes_preserve_attrs(merged_graph, mapping)
            merged_graph = sync_names(merged_graph)

        # debug statement...
        logging.debug("After updating node names:")
        for node in list(merged_graph.nodes())[:5]:
            logging.debug(f"  node: {node}")

        # ensure metadata written in correct format
        format_metadata_for_gml(merged_graph)

        # debug statement...
        logging.debug("After formatting metadata:")
        for node in list(merged_graph.nodes())[:5]:
            logging.debug(f"  node: {node}")

        # write new pangenome reference to fasta (stream to reduce memory)
        #reference_out = Path(options.outdir) / f"pan_genome_reference_{graph_count+1}.fa"
        #with open(reference_out, "w") as fasta_out:
        #    for node in merged_graph.nodes():
        #        seqs = merged_graph.nodes[node]["dna"].split(";")
        #        node_centroid_seq = max(seqs, key=len)
        #        fasta_out.write(f">{node}\n{node_centroid_seq}\n")

        # after first iter, update base mmseqs database so first graph has _g1 appended node names
        if graph_count == 0:
            updated_node_names = Path(options.outdir) / "mmseqs_tmp" / f"tmp.fa"
            with open(updated_node_names, "w") as fasta_out:
                for node in merged_graph.nodes():
                    name = node
                    seqs = merged_graph.nodes[node]["protein"]
                    if isinstance(seqs, (list, tuple)):
                        seqs = max(seqs, key=len) # if list, pick longest sequence
                    if isinstance(seqs, str):
                        parts = seqs.split(";") # if string split on semicolon and pick longest
                        seqs = max(parts, key=len)
                    seqs = seqs.rstrip('*') # remove trailing stop
                    fasta_out.write(f">{node}\n{seqs}\n")

            outdb = str(Path(options.outdir) / "mmseqs_tmp" / f"pan_genome_db_{graph_count+2}")
            mmseqs_createdb(fasta=updated_node_names, outdb=outdb, threads=options.threads, nt2aa=False)
            base_db = outdb
            
        else:
            # write new nodes to fasta to update mmseqs db
            new_nodes_fasta = Path(options.outdir) / "mmseqs_tmp" / f"new_nodes_{graph_count+2}.fa"
            n_new_nodes = 0
            with open(new_nodes_fasta, "w") as fasta_out:
                for node in merged_graph.nodes():
                    name = node
                    if name.endswith(f'_g{graph_count+2}'):
                        seqs = merged_graph.nodes[node]["protein"]
                        if isinstance(seqs, (list, tuple)):
                            seqs = max(seqs, key=len) # if list, pick longest sequence
                        if isinstance(seqs, str):
                            parts = seqs.split(";") # if string split on semicolon and pick longest
                            seqs = max(parts, key=len)
                        seqs = seqs.rstrip('*') # remove trailing stop
                        fasta_out.write(f">{node}\n{seqs}\n")
                        n_new_nodes += 1

            if n_new_nodes == 0:
                # this graph contributed no unique nodes, so the fasta is empty and
                # there would be nothing to concat. mmseqs createdb exits 1 on an empty
                # fasta, so skip building/concatenating a new-nodes db. The next iteration
                # still recomputes base_db = pan_genome_db_{graph_count+1} and reads it off
                # disk (line ~412), so the incremented db name MUST exist -- copy the current
                # base_db (identical content) to it rather than leaving a gap.
                outdb = str(Path(options.outdir) / "mmseqs_tmp" / f"pan_genome_db_{graph_count+2}")
                logging.info(f"Graph {graph_count+2} added 0 new nodes; carrying base mmseqs db forward.")
                base_prefix = os.path.basename(base_db)
                out_dir = os.path.dirname(outdb)
                out_prefix = os.path.basename(outdb)
                for src in glob_module.glob(base_db + "*"):
                    suffix = os.path.basename(src)[len(base_prefix):]
                    # only the db's own component files (exact prefix + '' / '.' / '_' tail),
                    # never a numerically-adjacent db that shares the leading digits
                    if suffix and suffix[0] not in (".", "_"):
                        continue
                    shutil.copyfile(src, os.path.join(out_dir, out_prefix + suffix))
                base_db = outdb
            else:
                # update mmseqs database
                new_nodes_db = str(Path(options.outdir) / "mmseqs_tmp" / f"tmp_db")
                outdb = str(Path(options.outdir) / "mmseqs_tmp" / f"pan_genome_db_{graph_count+2}")

                mmseqs_createdb(fasta=new_nodes_fasta, outdb=new_nodes_db, threads=options.threads, nt2aa=False)
                mmseqs_concatdbs(db1=base_db, db2=new_nodes_db, outdb=outdb, tmpdir=str(Path(options.outdir) / "mmseqs_tmp"), threads=options.threads)
                base_db = outdb

        # info statement...
        logging.info('Writing new metadata to SQLite database...')

        # add metadata to SQLite database
        add_metadata_to_sqlite(G=merged_graph, iteration=graph_count+1, con=con)

        # ensure WAL doesn't increase dramatically (execute after iteration finishes)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE);")

        # info statement...
        logging.info('Writing merged graph to outdir...')

        # define new graph name
        is_final_iteration = (graph_count == n_graphs - 2)
        if is_final_iteration:
            output_path = Path(options.outdir) / "final_graph.gml"
        else:
            output_path = intermediate_dir / f"merged_graph_{graph_count+1}.gml"

        # always strip metadata from the in-memory graph so add_metadata_to_sqlite
        # stays idempotent across iterations. With --metadata-in-graph the final
        # graph's metadata is rebuilt from SQLite below.
        for n in merged_graph.nodes():
            degrees = merged_graph.nodes[n]["degrees"]
            merged_graph.nodes[n].clear()
            merged_graph.nodes[n]["name"] = n
            merged_graph.nodes[n]["seqIDs"] = []
            merged_graph.nodes[n]["geneIDs"] = ''
            merged_graph.nodes[n]["members"] = []
            merged_graph.nodes[n]["genomeIDs"] = ''
            merged_graph.nodes[n]["size"] = 1
            merged_graph.nodes[n]["lengths"] = []
            merged_graph.nodes[n]['longCentroidID'] = []
            merged_graph.nodes[n]['maxLenId'] = ''
            merged_graph.nodes[n]['centroid'] = []
            merged_graph.nodes[n]['dna'] = [""]
            merged_graph.nodes[n]['protein'] = [""]
            merged_graph.nodes[n]["hasEnd"] = 0
            merged_graph.nodes[n]["annotation"] = ''
            merged_graph.nodes[n]["description"] = ''
            merged_graph.nodes[n]["paralog"] = 0
            merged_graph.nodes[n]["mergedDNA"] = ''
            merged_graph.nodes[n]["degrees"] = degrees

        for u, v in merged_graph.edges():
            merged_graph[u][v].clear()
            merged_graph[u][v]["name"] = n
            merged_graph[u][v]["size"] = 1
            merged_graph[u][v]["members"] = []
            merged_graph[u][v]['genomeIDs'] = ''

        if is_final_iteration and options.keep_metadata_in_graph is True:
            # write the stripped version first, then rebuild metadata from
            # SQLite and write the final graph WITH metadata
            nx.write_gml(
                merged_graph,
                str(Path(options.outdir) / "final_graph_nometadata.gml"),
            )
            load_metadata_from_sqlite(merged_graph, con)
            nx.write_gml(merged_graph, str(output_path))
        else:
            nx.write_gml(merged_graph, str(output_path))

        # Scoring MUST happen after the graph has been written, not before. Metadata --
        # seqIDs included -- is stripped from the in-memory graph at the end of every
        # iteration so add_metadata_to_sqlite stays idempotent, so by the final iteration
        # the in-memory graph holds only the seqIDs of the component merged last. Scoring
        # there silently measured a 1/N subset: a 5-component merge scored 376,373 of
        # 1,880,303 genes. load_metadata_from_sqlite above restores the complete mapping
        # from SQLite, which is authoritative, so the graph is whole by this point.
        # calculate clustering performance (if test mode)
        if options.mode == 'test' and graph_count == (n_graphs-2):

            # info statement...
            logging.info("Calculating adjusted Rand index (ARI) and adjusted mutual information (AMI)...")

            ### gather seqIDs to enable calculation of clustering metrics

            cluster_dict_merged = get_seqIDs_in_nodes(merged_graph)
            cluster_dict_all = get_seqIDs_in_nodes(graph_all)

            # diagnostics: how many seqIDs are shared / excluded between merged and all
            seq_ids_1 = set(cluster_dict_merged)
            seq_ids_2 = set(cluster_dict_all)
            common_seq_ids = seq_ids_1 & seq_ids_2
            only_in_graph_1 = seq_ids_1 - seq_ids_2
            only_in_graph_2 = seq_ids_2 - seq_ids_1
            logging.info(f"shared seqIDs: {len(common_seq_ids)}")
            logging.info(f"seqIDs only in merged (excluded): {len(only_in_graph_1)}")
            logging.info(f"seqIDs only in all (excluded): {len(only_in_graph_2)}")
            logging.debug(f"seqIDs only in merged (excluded): {only_in_graph_1}")
            logging.debug(f"seqIDs only in all (excluded): {only_in_graph_2}")

            # standard scores (per-gene random baseline), post-clustering scores
            # (component graphs as baseline) and attainability-corrected scores (best
            # merge reachable from those graphs as ceiling), on the same aligned seqIDs
            if options.fast:
                logging.info("--fast: skipping AMI (expected mutual information)")
            scores = graph_similarity_scores(
                truth_map=cluster_dict_all,
                merged_map=cluster_dict_merged,
                component_map=component_cluster_of_seqid,
                standard_scores=not options.fast,
            )
            logging.info(f"seqIDs scored: {scores['n_seqIDs']}")
            logging.info(f"Rand Index: {scores['RI']}")
            logging.info(f"Adjusted Rand Index: {scores['ARI']}")
            logging.info(f"Mutual Information: {scores['MI']}")
            if 'AMI' in scores:
                logging.info(f"Adjusted Mutual Information: {scores['AMI']}")

            logging.info("Post-clustering scores (component graphs as baseline):")
            logging.info(f"  RI(merged,all)={scores['RI']:.6f}  RI(component,all)={scores['RI_component']:.6f}")
            logging.info(f"  pcARI: {scores['pcARI']}")
            logging.info(f"  NMI(merged,all)={scores['NMI_merged']:.6f}  NMI(component,all)={scores['NMI_component']:.6f}")
            logging.info(f"  pcNMI: {scores['pcNMI']}")

            if "acARI" in scores:
                logging.info("Attainability-corrected scores (best reachable merge as ceiling):")
                logging.info(f"  acARI: {scores['acARI']}")
                logging.info(f"  acNMI: {scores['acNMI']}")
                logging.info(f"  pairwise errors: component={scores['err_component']:.0f}  "
                             f"merged={scores['err_merged']:.0f}  reachable floor={scores['err_cstar']:.0f}")

            logging.info("Over- vs under-merging (all-data graph as reference labels):")
            logging.info(f"  homogeneity={scores['homogeneity']:.6f}  "
                         f"(component={scores['homogeneity_component']:.6f})  low => over-merging")
            logging.info(f"  completeness={scores['completeness']:.6f}  "
                         f"(component={scores['completeness_component']:.6f})  low => under-merging")

            # machine-readable copy of the same numbers, so a results table does not
            # have to be parsed back out of the log
            metrics_row = dict(scores)
            metrics_row["shared_seqIDs"] = len(common_seq_ids)
            metrics_row["only_in_merged"] = len(only_in_graph_1)
            metrics_row["only_in_all"] = len(only_in_graph_2)
            metrics_path = Path(options.outdir) / "test_metrics.tsv"
            with open(metrics_path, "w") as metrics_handle:
                metrics_handle.write("\t".join(metrics_row.keys()) + "\n")
                metrics_handle.write("\t".join(str(v) for v in metrics_row.values()) + "\n")
            logging.info(f"Wrote test metrics to {metrics_path}")

        # reduce memory by removing intermediate files
        for name in [
            "mapping", "mapping_target", "mapping_groups_new",
            "centroids_fa",
            "reordered_pairs",
        ]:
            if name in locals():
                del locals()[name]
        gc.collect()

        # add 1 to graph count
        graph_count += 1

        # print progress statement...
        logging.info(f"Iteration {graph_count} of {n_graphs-1} complete.")
    
    # populate poppunk_cluster column from external clusters CSV if provided
    if options.include_clusters is not None:
        add_clusters_to_sqlite(con, options.include_clusters)

    # create indexes for SQLite database
    sqlite_create_indexes(con)

    # close connection to sqlite db
    con.close()

    # move final mmseqs AA database to outdir and clean up
    logging.info('Moving final MMSeqs2 database and cleaning up...')
    final_db_prefix = str(mmseqs_dir / f"pan_genome_db_{n_graphs}")
    ref_dir = Path(options.outdir) / "pangenome_reference_aa"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for src in glob_module.glob(f"{final_db_prefix}*"):
        ext = src[len(final_db_prefix):]  # e.g. "", ".dbtype", ".index", "_h", "_h.dbtype", etc.
        dst = ref_dir / f"pangenome_reference_aa{ext}"
        shutil.move(src, dst)
    shutil.rmtree(mmseqs_dir)

    generate_summary_statistics(
        sqlite_path=sqlite_path,
        output_dir=options.outdir,
        sqlite_cache=options.sqlite_cache,
    )

    # info statement...
    logging.info('Finished successfully.')

    #return merge
if __name__ == "__main__":
    main()

