import os
import shutil
import json
import random
import subprocess
import sys
import re
import sqlite3
import threading
import zlib
from datetime import datetime, timezone

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from Bio import SeqIO
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from Bio.Data.CodonTable import generic_by_id
import warnings

from pangenomerge.custom_functions.gene_names import derive_gene_name

unambiguous_degenerate_codons = {"ACN":"T", "TCN":"S", "CTN":"L", "CCN":"P",
                                 "CGN":"R", "GTN":"V", "GCN":"A", "GGN":"G"}

bact_translation_table = np.array([[[b'K', b'N', b'K', b'N', b'X'],
                               [b'T', b'T', b'T', b'T', b'T'],
                               [b'R', b'S', b'R', b'S', b'X'],
                               [b'I', b'I', b'M', b'I', b'X'],
                               [b'X', b'X', b'X', b'X', b'X']],
                              [[b'Q', b'H', b'Q', b'H', b'X'],
                               [b'P', b'P', b'P', b'P', b'P'],
                               [b'R', b'R', b'R', b'R', b'R'],
                               [b'L', b'L', b'L', b'L', b'L'],
                               [b'X', b'X', b'X', b'X', b'X']],
                              [[b'E', b'D', b'E', b'D', b'X'],
                               [b'A', b'A', b'A', b'A', b'A'],
                               [b'G', b'G', b'G', b'G', b'G'],
                               [b'V', b'V', b'V', b'V', b'V'],
                               [b'X', b'X', b'X', b'X', b'X']],
                              [[b'*', b'Y', b'*', b'Y', b'X'],
                               [b'S', b'S', b'S', b'S', b'S'],
                               [b'*', b'C', b'W', b'C', b'X'],
                               [b'L', b'F', b'L', b'F', b'X'],
                               [b'X', b'X', b'X', b'X', b'X']],
                              [[b'X', b'X', b'X', b'X', b'X'],
                               [b'X', b'X', b'X', b'X', b'X'],
                               [b'X', b'X', b'X', b'X', b'X'],
                               [b'X', b'X', b'X', b'X', b'X'],
                               [b'X', b'X', b'X', b'X', b'X']]])

reduce_array = np.full(200, 4)
reduce_array[[65, 97]] = 0
reduce_array[[67, 99]] = 1
reduce_array[[71, 103]] = 2
reduce_array[[84, 116]] = 3

_GN_SUFFIX_RE = re.compile(r"^(.+)_g(\d+)$")
_thread_context = threading.local()


class PangenomeSequenceError(RuntimeError):
    pass


def get_trans_table(table):
    # swap to different codon table
    translation_table = bact_translation_table.copy()
    tb = generic_by_id[table]
    if table!=11:
        if table not in generic_by_id:
            raise RuntimeError("Invalid codon table! Must be available" +
                " as a generic table in BioPython")
        for codon in tb.forward_table:
            if 'U' in codon: continue
            ind = reduce_array[np.array(bytearray(codon.encode()), dtype=np.int8)]
            translation_table[ind[0], ind[1], ind[2]] = tb.forward_table[codon].encode('utf-8')
        for codon in tb.stop_codons:
            if 'U' in codon: continue
            ind = reduce_array[np.array(bytearray(codon.encode()), dtype=np.int8)]
            translation_table[ind[0], ind[1], ind[2]] = b'*'

    return([translation_table, set(tb.start_codons)])


def translate(seq, translation_table):
    indices = reduce_array[np.array(bytearray(seq.encode()), dtype=np.int8)]
    pseq = translation_table[0][
        indices[np.arange(0, len(seq), 3)], indices[np.arange(1, len(seq), 3)],
        indices[np.arange(2, len(seq), 3)]].tobytes().decode('ascii')
    # Check for a different start codon.
    if seq[0:3] in translation_table[1]:
        return ('M' + pseq[1:])
    return(pseq)


def shared_dir_is_distinct(output_dir, shared_dir):
    """True when the caller supplied a shared dir separate from the output dir."""
    if shared_dir is None:
        return False
    return _normalise_output_dir(shared_dir) != _normalise_output_dir(output_dir)


def _resolve_shared_dir(output_dir, shared_dir):
    """Directory holding intermediates that do not depend on --strict-codons.

    aligned_protein_sequences/ and unaligned_dna_sequences/ are byte-identical
    between --codons and --strict-codons, so both runs can point at one shared
    directory and the (expensive) protein alignment stage happens once. Defaults
    to output_dir, which reproduces the single-run layout exactly.
    """
    if shared_dir is None:
        return output_dir
    return _normalise_output_dir(shared_dir)


def _normalise_output_dir(output_dir):
    return os.path.join(str(output_dir), "")


def _value_to_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        if value == "":
            return []
        if ";" in value:
            return [item for item in value.split(";") if item]
        return [value]
    return list(value)


def get_node_sequence_ids(node):
    if "_geneids" in node:
        return _value_to_list(node["_geneids"])
    if "seqIDs" in node:
        return _value_to_list(node["seqIDs"])
    if "geneIDs" in node:
        return _value_to_list(node["geneIDs"])
    return []


def get_node_sequence_count(node):
    """Number of gene sequences in a node.

    Graphs from load_pangenomerge_alignment_graph carry the count directly; the
    fallback is for graphs built some other way, which still carry the ID lists.
    """
    if "n_geneids" in node:
        return int(node["n_geneids"])
    return len(get_node_sequence_ids(node))


def _parse_member_key_from_geneid(geneid, member=None, graph_id=None):
    if member:
        return str(member)

    match = _GN_SUFFIX_RE.match(str(geneid))
    if match is None:
        return None

    prefix, suffix_graph_id = match.groups()
    member_index = prefix.split("_", 1)[0]
    graph_id = graph_id if graph_id is not None else suffix_graph_id
    return f"{member_index}_g{graph_id}"


def _connect_alignment_sqlite(sqlite_path, sequences_sqlite_path):
    # PANGENOMERGE_SQLITE_IMMUTABLE=1 opens both databases with SQLite's
    # immutable=1: no locks are taken and -wal/-shm are never touched, which is
    # what lets many extraction processes read the databases at once without
    # contending on GPFS (the msa never writes them). The runner sets it only
    # for --immutable-db after verifying the WAL is empty, and it is an
    # environment variable so loky worker processes inherit it.
    if os.environ.get("PANGENOMERGE_SQLITE_IMMUTABLE") == "1":
        con = sqlite3.connect(f"file:{sqlite_path}?immutable=1", uri=True)
        con.execute("ATTACH DATABASE ? AS seq",
                    (f"file:{sequences_sqlite_path}?immutable=1",))
    else:
        con = sqlite3.connect(str(sqlite_path))
        con.execute("ATTACH DATABASE ? AS seq", (str(sequences_sqlite_path),))
    con.execute("PRAGMA query_only=ON;")
    return con


def _load_member_to_sample(con):
    return {
        f"{member_index}_g{graph_id}": str(sample_name)
        for graph_id, member_index, sample_name in con.execute(
            "SELECT graph_id, member_index, sample_name "
            "FROM isolate_names ORDER BY graph_id, member_index"
        )
    }


def _get_thread_context(sqlite_path, sequences_sqlite_path):
    key = (os.path.abspath(str(sqlite_path)),
           os.path.abspath(str(sequences_sqlite_path)))
    context = getattr(_thread_context, "alignment_context", None)
    if context is not None and context["key"] == key:
        return context["con"], context["member_to_sample"]

    if context is not None:
        try:
            context["con"].close()
        except sqlite3.Error:
            pass

    con = _connect_alignment_sqlite(sqlite_path, sequences_sqlite_path)
    member_to_sample = _load_member_to_sample(con)
    context = {"key": key, "con": con, "member_to_sample": member_to_sample}
    _thread_context.alignment_context = context
    return con, member_to_sample


def _decompress_sequence(blob):
    return zlib.decompress(blob).decode()


def _sequence_type_label(seq_type):
    if seq_type == "nt":
        return "nucleotide"
    if seq_type == "aa":
        return "amino-acid"
    raise ValueError("seq_type must be 'nt' or 'aa'")


def get_node_sequence_records(sqlite_path, sequences_sqlite_path, node_id,
                              seq_type="nt"):
    """Load FASTA records for one pangenomerge node from SQLite."""
    seq_label = _sequence_type_label(seq_type)
    hash_col = "gs.nt_hash" if seq_type == "nt" else "gs.aa_hash"
    con, member_to_sample = _get_thread_context(sqlite_path, sequences_sqlite_path)

    try:
        rows = con.execute(f"""
            SELECT ng.geneid, ng.member, ng.graph_id, us.seq_data
            FROM node_geneids ng
            LEFT JOIN seq.gene_sequences gs ON gs.geneid = ng.geneid
            LEFT JOIN seq.unique_sequences us
              ON us.seq_hash = {hash_col}
             AND us.seq_type = ?
            WHERE ng.node_id = ?
            ORDER BY ng.geneid
        """, (seq_type, str(node_id))).fetchall()
    except sqlite3.Error as exc:
        raise PangenomeSequenceError(
            "Could not read pangenome sequences from SQLite. Ensure "
            "pangenome_metadata.sqlite and pangenome_sequences.sqlite were "
            "generated by pangenomerge-postprocess --output sequences."
        ) from exc

    if not rows:
        raise PangenomeSequenceError(
            f"No gene IDs were found for node {node_id!r} in "
            "pangenome_metadata.sqlite."
        )

    records = []
    missing_geneids = []
    for geneid, member, graph_id, seq_blob in rows:
        if seq_blob is None:
            missing_geneids.append(geneid)
            continue

        member_key = _parse_member_key_from_geneid(geneid, member, graph_id)
        sample_name = member_to_sample.get(member_key, member_key or "unknown")
        sample_name = str(sample_name).replace(";", "")
        records.append(
            SeqRecord(
                Seq(_decompress_sequence(seq_blob)),
                id=f"{sample_name};{geneid}",
                description="",
            )
        )

    if missing_geneids:
        preview = ", ".join(missing_geneids[:5])
        if len(missing_geneids) > 5:
            preview += f", ... (+{len(missing_geneids) - 5} more)"
        raise PangenomeSequenceError(
            f"Missing {seq_label} sequence(s) for {len(missing_geneids)} "
            f"gene ID(s) in pangenome_sequences.sqlite: {preview}. Run "
            "pangenomerge-postprocess --output sequences before "
            "pangenomerge-msa."
        )

    return records


def _require_metadata_tables(con, sqlite_path):
    required_tables = {
        "nodes", "node_members", "node_geneids", "isolate_names",
    }
    present_tables = {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing_tables = sorted(required_tables - present_tables)
    if missing_tables:
        raise RuntimeError(
            f"{sqlite_path} is missing required table(s): "
            + ", ".join(missing_tables)
        )


def load_isolate_names(sqlite_path):
    con = sqlite3.connect(str(sqlite_path))
    con.execute("PRAGMA query_only=ON;")
    try:
        _require_metadata_tables(con, sqlite_path)
        isolate_names = [
            row[0]
            for row in con.execute(
                "SELECT sample_name FROM isolate_names "
                "ORDER BY graph_id, member_index"
            )
        ]
    finally:
        con.close()

    if not isolate_names:
        raise RuntimeError("No isolate names found in pangenome_metadata.sqlite.")
    return isolate_names


def load_pangenomerge_alignment_graph(sqlite_path, gml_path):
    """Load a graph shell from GML and attach alignment metadata from SQLite."""
    try:
        raw_graph = nx.read_gml(str(gml_path))
    except Exception as exc:
        raise RuntimeError(f"Could not read final graph GML: {gml_path}") from exc

    graph = nx.relabel_nodes(
        raw_graph,
        {node_id: str(node_id) for node_id in raw_graph.nodes()},
        copy=True,
    )

    con = sqlite3.connect(str(sqlite_path))
    con.execute("PRAGMA query_only=ON;")
    try:
        _require_metadata_tables(con, sqlite_path)

        # Only the COUNTS are ever needed downstream -- every live caller of
        # get_node_sequence_ids takes len() of it. Aggregating in SQL keeps two
        # 14k-row results instead of ~80M interned strings held three ways per
        # node, which is what made this function OOM on large pangenomes. Both
        # queries are index scans on the existing primary keys.
        member_counts = {
            str(node_id): count
            for node_id, count in con.execute(
                "SELECT node_id, COUNT(DISTINCT member) "
                "FROM node_members GROUP BY node_id"
            )
        }

        geneid_counts = {
            str(node_id): count
            for node_id, count in con.execute(
                "SELECT node_id, COUNT(*) FROM node_geneids GROUP BY node_id"
            )
        }

        node_rows = con.execute(
            "SELECT node_id, size, annotation FROM nodes ORDER BY node_id"
        ).fetchall()
    finally:
        con.close()

    if not node_rows:
        raise RuntimeError("No node metadata found in pangenome_metadata.sqlite.")

    used_gene_names = set([""])
    group_counter = 0
    alignment_node_count = 0
    metadata_node_ids = set()
    for node_id, size, annotation in node_rows:
        node_id = str(node_id)
        metadata_node_ids.add(node_id)
        geneid_count = geneid_counts.get(node_id, 0)
        member_count = member_counts.get(node_id, 0)
        # nodes.name is deliberately ignored: generate_output derives gene names
        # from the annotation alone, and the two must agree.
        name, group_counter = derive_gene_name(
            annotation,
            used_gene_names,
            group_counter,
        )

        if member_count:
            size_value = member_count
        elif size is not None:
            size_value = int(size)
        else:
            size_value = geneid_count

        if node_id not in graph:
            graph.add_node(node_id)
        graph.nodes[node_id].update({
            "node_id": node_id,
            "name": name,
            "size": size_value,
            "n_geneids": geneid_count,
        })
        alignment_node_count += 1

    for node_id in list(graph.nodes()):
        if str(node_id) not in metadata_node_ids:
            graph.remove_node(node_id)

    if alignment_node_count < 1:
        raise RuntimeError("No alignable nodes found in pangenome_metadata.sqlite.")

    return graph


def get_alignment_basename(node):
    gene_name = node["name"]
    if len(gene_name) >= 237:
        return gene_name[:236]
    return gene_name


def get_temp_dna_input_path(node, temp_directory):
    outname = os.path.join(temp_directory, node["name"] + ".fasta")
    if len(outname) >= 248:
        outname = outname[:248] + ".fasta"
    return outname


def get_expected_gene_alignment_path(node, output_dir, codons, aligner=None):
    if codons:
        return os.path.join(output_dir, "aligned_gene_sequences",
                            get_alignment_basename(node) + ".aln.fas")

    n_sequences = get_node_sequence_count(node)

    if aligner == "none" and n_sequences > 1:
        return os.path.join(output_dir, "unaligned_gene_sequences",
                            get_alignment_basename(node) + ".fasta")

    if n_sequences > 1:
        return os.path.join(output_dir, "aligned_gene_sequences",
                            get_alignment_basename(node) + ".aln.fas")

    return os.path.join(output_dir, "aligned_gene_sequences",
                        node["name"] + ".fasta")


def get_expected_protein_input_path(node, temp_directory):
    return os.path.join(temp_directory, get_alignment_basename(node) + ".fasta")


def get_expected_protein_alignment_path(node, shared_dir):
    return os.path.join(shared_dir, "aligned_protein_sequences",
                        get_alignment_basename(node) + ".aln.fas")


def get_expected_unaligned_dna_path(node, shared_dir):
    return os.path.join(shared_dir, "unaligned_dna_sequences",
                        get_alignment_basename(node) + ".fasta")


def get_expected_extracted_protein_path(node, shared_dir):
    """Persistent home of a protein input written by extract_sequences_sharded."""
    return os.path.join(shared_dir, "unaligned_protein_sequences",
                        get_alignment_basename(node) + ".fasta")


def _output_dna_and_protein_or_reuse(node, temp_dir, output_dir, sqlite_path,
                                     sequences_sqlite_path, shared_dir):
    """output_dna_and_protein, unless a sharded extraction already ran.

    extract_sequences_sharded leaves each gene's protein input under the
    shared directory and its DNA where output_dna_and_protein would put it.
    When both exist the protein file is hard-linked into the temp dir -
    align_sequences deletes its input after aligning, and a link keeps the
    shared copy - and SQLite is not touched at all. Anything missing falls
    through to the normal extraction, so an absent or partial sharded run is
    always safe, just slower.
    """
    pre = get_expected_extracted_protein_path(node, shared_dir)
    dna = get_expected_unaligned_dna_path(node, shared_dir)
    if os.path.isfile(pre) and os.path.isfile(dna):
        prot = get_expected_protein_input_path(node, temp_dir)
        os.makedirs(os.path.dirname(prot), exist_ok=True)
        if not os.path.exists(prot):
            try:
                os.link(pre, prot)
            except OSError:
                shutil.copyfile(pre, prot)
        return (prot, dna)
    return output_dna_and_protein(node, temp_dir, output_dir,
                                  sqlite_path=sqlite_path,
                                  sequences_sqlite_path=sequences_sqlite_path,
                                  shared_dir=shared_dir)


def get_shared_manifest_path(shared_dir):
    return os.path.join(shared_dir, "shared_alignment_state.json")


def check_shared_manifest(shared_dir, aligner):
    """Record the aligner used for the shared intermediates, or verify it matches.

    Protein alignments in the shared directory are reused without --resume, so
    they must not be silently reused under a different aligner.
    """
    manifest_path = get_shared_manifest_path(shared_dir)
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as handle:
            existing = json.load(handle)
        if existing.get("aligner") != aligner:
            raise RuntimeError(
                "--shared-alignment-dir " + shared_dir + " was built with aligner "
                + str(existing.get("aligner")) + ", but this run uses " + aligner
                + ". Use a different --shared-alignment-dir, or delete "
                + manifest_path + " and the directories beside it."
            )
        return existing

    os.makedirs(shared_dir, exist_ok=True)
    manifest = {"aligner": aligner}
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def get_resume_manifest_path(output_dir):
    return os.path.join(output_dir, "alignment_resume_state.json")


def load_resume_manifest(output_dir):
    manifest_path = get_resume_manifest_path(output_dir)
    if not os.path.isfile(manifest_path):
        return None

    with open(manifest_path, "r") as handle:
        return json.load(handle)


def check_resume_manifest_collision(output_dir, resume, shared_dir=None):
    if resume:
        return

    manifest_path = get_resume_manifest_path(output_dir)
    if not os.path.isfile(manifest_path):
        return

    shared_dir = _resolve_shared_dir(output_dir, shared_dir)
    raise RuntimeError(
        "Found an existing gene-alignment resume manifest in "
        + output_dir
        + ". Re-run with --resume to continue the previous alignment. "
        + "To start the sequence alignment again from scratch, delete: "
        + manifest_path
        + ", "
        + os.path.join(output_dir, "aligned_gene_sequences/")
        + ", "
        + os.path.join(shared_dir, "aligned_protein_sequences/")
        + ", and "
        + os.path.join(shared_dir, "unaligned_dna_sequences/")
        + "."
    )


def write_resume_manifest(output_dir, alignment, aligner, codons, strict_codons,
                          core_threshold, subset=None, resume=False):
    manifest_path = get_resume_manifest_path(output_dir)
    existing_manifest = load_resume_manifest(output_dir)

    if resume:
        if existing_manifest is None:
            raise RuntimeError(
                "Cannot resume pangenomerge-msa: no alignment resume manifest was found."
            )

        for field, expected_value in [
            ("alignment", alignment),
            ("aligner", aligner),
            ("codons", codons),
            ("strict_codons", strict_codons),
            ("core_threshold", core_threshold),
            ("subset", subset),
        ]:
            if existing_manifest.get(field) != expected_value:
                raise RuntimeError(
                    "Cannot resume pangenomerge-msa: current run does not match the "
                    + "existing manifest for '" + field + "'."
                )

        return existing_manifest

    manifest = {
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alignment": alignment,
        "aligner": aligner,
        "codons": codons,
        "strict_codons": strict_codons,
        "core_threshold": core_threshold,
        "subset": subset,
    }

    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest


def is_valid_fasta(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as handle:
            records = list(SeqIO.parse(handle, "fasta"))
        return len(records) > 0
    except Exception:
        return False


def is_valid_alignment(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as handle:
            alignment = AlignIO.read(handle, "fasta")
        return len(alignment) > 0
    except Exception:
        return False


def gene_has_valid_final_output(node, output_dir, codons, aligner=None):
    output_path = get_expected_gene_alignment_path(node, output_dir, codons, aligner)
    if output_path.endswith(".aln.fas"):
        return is_valid_alignment(output_path)
    return is_valid_fasta(output_path)


def gene_has_valid_protein_output(node, shared_dir):
    return is_valid_alignment(get_expected_protein_alignment_path(node, shared_dir))


def node_requires_msa(node):
    return get_node_sequence_count(node) > 1


def get_pending_gene_ids(nodes, output_dir, codons, resume, aligner=None):
    pending_gene_ids = []
    for node_id, node in nodes:
        if resume and gene_has_valid_final_output(node, output_dir, codons, aligner):
            continue
        pending_gene_ids.append(node_id)
    return pending_gene_ids


def get_pending_codon_gene_ids(nodes, output_dir, resume, shared_dir=None):
    # A distinct shared dir holds intermediates that do not depend on the codon
    # mode, so a valid protein alignment there is reused even without --resume:
    # that reuse is the whole point of --shared-alignment-dir.
    reuse_shared = shared_dir_is_distinct(output_dir, shared_dir)
    shared_dir = _resolve_shared_dir(output_dir, shared_dir)
    protein_pending_gene_ids = []
    reverse_translate_pending_gene_ids = []

    for node_id, node in nodes:
        if resume and gene_has_valid_final_output(node, output_dir, codons=True):
            continue

        if not node_requires_msa(node):
            protein_pending_gene_ids.append(node_id)
            continue

        reverse_translate_pending_gene_ids.append(node_id)
        if (resume or reuse_shared) and gene_has_valid_protein_output(node, shared_dir):
            continue

        protein_pending_gene_ids.append(node_id)

    return protein_pending_gene_ids, reverse_translate_pending_gene_ids


def get_codon_pending_files(nodes, shared_dir, gene_ids):
    gene_id_set = set(gene_ids)
    selected_nodes = [node for node_id, node in nodes if node_id in gene_id_set]

    protein_alignment_files = [
        get_expected_protein_alignment_path(node, shared_dir)
        for node in selected_nodes
    ]
    dna_sequence_files = [
        get_expected_unaligned_dna_path(node, shared_dir)
        for node in selected_nodes
    ]

    return protein_alignment_files, dna_sequence_files


def print_stage_progress(stage_name, completed, remaining, total=None):
    if total is None:
        total = completed + remaining
    print(
        f"{stage_name}: {completed} completed alignments found, "
        f"{remaining} to be aligned out of {total}."
    )


def check_aligner_install(aligner):
    """Checks for the presence of the specified aligned in $PATH

    Args:
        check_aligner_install(str)
            str = specified aligner

    Returns:
        presence (bool)
            True/False aligner present
    """
    if aligner == "clustal":
        command = "clustalo --help"
    elif aligner == "prank":
        command = "prank -help"
    elif aligner == "mafft":
        command = "mafft --help"
    elif aligner in {"muscle", "muscle-super5"}:
        command = "muscle -h"
    elif aligner == "famsa":
        command = "famsa -h"
    elif aligner == "none":
        return True
    else:
        sys.stderr.write("Incorrect aligner specification\n")
        sys.exit()
    p = str(
        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
    )
    present = False

    if aligner == "clustal":
        find_ver = re.search(r"Clustal Omega - \d+\.\d+\.\d+", p)
    elif aligner == "prank":
        find_ver = re.search(r"prank v\.\d+\.", p)
    elif aligner == "mafft":
        find_ver = re.search(r"MAFFT v\d+\.\d+", p)
    elif aligner in {"muscle", "muscle-super5"}:
        find_ver = re.search(r"muscle\s+\d+\.\d+\.\S+", p, re.IGNORECASE)
    elif aligner == "famsa":
        find_ver = re.search(r"FAMSA.*?version\s+\d+\.\d+\.\d+(?:-[A-Za-z0-9]+)?", p, re.IGNORECASE | re.DOTALL)
    
    if find_ver != None:
        present = True

    if present == False:
        sys.stderr.write("Need specified aligner to be installed " + "\n")
        sys.exit(1)

    return present

def check_aligner_sanity(aligner, codons, isolate_count):
    if aligner == "famsa" and codons == False:
        raise RuntimeError(
                "FAMSA2 only supports amino-acid alignment. "
                "Use --codons or --strict-codons to align."
            )
    if aligner == "none" and codons:
        raise RuntimeError(
            "Codon alignment requires a protein aligner; --aligner none is "
            "only valid for nucleotide FASTA output."
        )
    elif aligner == "muscle" and (isolate_count > 300):
        warnings.warn("MUSCLE is not optimised to run on more than a few"
                      "hundred isolates. Aligning the core genome may be very"
                      "slow or fail to complete. Use muscle-super5 for faster"
                      "alignment on larger datasets",
                      UserWarning)
    return True

def output_sequence(node, temp_directory, outdir,
                    sqlite_path=None, sequences_sqlite_path=None):
    outdir = _normalise_output_dir(outdir)
    if sqlite_path is None or sequences_sqlite_path is None:
        raise PangenomeSequenceError(
            "output_sequence requires pangenome_metadata.sqlite and "
            "pangenome_sequences.sqlite paths.")
    output_sequences = get_node_sequence_records(
        sqlite_path,
        sequences_sqlite_path,
        node.get("node_id", node.get("id", node["name"])),
        seq_type="nt",
    )

    if len(output_sequences) > 1:
        outname = get_temp_dna_input_path(node, temp_directory)
    else:
        outname = get_expected_gene_alignment_path(node, outdir, codons=False)
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        SeqIO.write(output_sequences, outname, "fasta")
        return None

    os.makedirs(os.path.dirname(outname), exist_ok=True)
    SeqIO.write(output_sequences, outname, "fasta")
    return outname

def output_dna_and_protein(node, temp_directory, outdir,
                           sqlite_path=None, sequences_sqlite_path=None,
                           shared_dir=None):
    outdir = _normalise_output_dir(outdir)
    shared_dir = _resolve_shared_dir(outdir, shared_dir)
    if sqlite_path is None or sequences_sqlite_path is None:
        raise PangenomeSequenceError(
            "output_dna_and_protein requires pangenome_metadata.sqlite and "
            "pangenome_sequences.sqlite paths.")
    node_id = node.get("node_id", node.get("id", node["name"]))
    output_dna = get_node_sequence_records(
        sqlite_path, sequences_sqlite_path, node_id, seq_type="nt"
    )
    output_protein = get_node_sequence_records(
        sqlite_path, sequences_sqlite_path, node_id, seq_type="aa"
    )

    if len(output_dna) != len(output_protein):
        raise PangenomeSequenceError(
            "DNA and protein sequence counts do not match for node "
            + str(node.get("node_id", node.get("name")))
        )

    if len(output_dna) > 1:
        prot_outname = get_expected_protein_input_path(node, temp_directory)
        dna_outname = get_expected_unaligned_dna_path(node, shared_dir)
        os.makedirs(os.path.dirname(prot_outname), exist_ok=True)
        os.makedirs(os.path.dirname(dna_outname), exist_ok=True)
        SeqIO.write(output_protein, prot_outname, 'fasta')
        SeqIO.write(output_dna, dna_outname, 'fasta')
        output_files = (prot_outname, dna_outname)
    else:
        singleton_outname = get_expected_gene_alignment_path(
            node, outdir, codons=True
        )
        os.makedirs(os.path.dirname(singleton_outname), exist_ok=True)
        SeqIO.write(output_dna, singleton_outname, 'fasta')
        output_files = (None, None)

    return output_files

def get_alignment_commands(fastafile_name, outdir, aligner, threads):
    geneName = fastafile_name.split("/")[-1].split(".")[0]

    if  aligner == "mafft":
        command = "mafft "
        command += "--auto --adjustdirection --thread 1 --nuc "
        command += fastafile_name

    elif aligner == "muscle":
        command = "muscle "
        command += " -align " + fastafile_name 
        command += " -nt "
        command += " -threads 1" 
        command += " -output " + outdir + "aligned_gene_sequences/" + geneName + ".aln.fas"

    elif aligner == "muscle-super5":
        command = "muscle "
        command += " -super5 " + fastafile_name 
        command += " -nt "
        command += " -threads 1" 
        command += " -output " + outdir + "aligned_gene_sequences/" + geneName + ".aln.fas"
    #FAMSA only supports Amino acids, this should never trigger!
    elif aligner == "famsa":
        raise RuntimeError(
                "FAMSA2 only supports amino-acid alignment."
                "Use --codons or --strict-codons to align."
            )
    return (command, fastafile_name)

def get_protein_commands(fastafile_name, outdir, aligner, threads):
    if fastafile_name != None:
        geneName = fastafile_name.split('/')[-1].split('.')[0]
    else:
        return (None, None)
    if aligner == "mafft":
        command = "mafft "
        command += "--auto --amino "
        command += fastafile_name

    elif aligner == "muscle":
        command = "muscle "
        command += " -align " + fastafile_name 
        command += " -amino "
        command += " -threads 1" 
        command += " -output " + outdir + "aligned_protein_sequences/" + geneName + ".aln.fas"

    elif aligner == "muscle-super5":
        command = "muscle "
        command += " -super5 " + fastafile_name 
        command += " -amino "
        command += " -threads 1" 
        command += " -output " + outdir + "aligned_protein_sequences/" + geneName + ".aln.fas"

    elif aligner == "famsa":
        command = "famsa "
        command += " -t 1 " 
        command += fastafile_name 
        command += " " + outdir + "aligned_protein_sequences/" + geneName + ".aln.fas"

    return (command, fastafile_name)

def get_align_dna_to_alignment_commands(bad_dna_seqs_file, codonalignment_file, 
                                        outdir, aligner):
    geneName = codonalignment_file.split('/')[-1].split('.')[0]
    if aligner == "prank":
        raise Exception("This is a bug! Panaroo does not supports codon "
                        "alignment with PRANK")
    #default to MAFFT for profile alignment (other aligners do not support it)    
    elif aligner in {"mafft", "muscle", "muscle-super5", "famsa"}:
        command = ["mafft",
                   "--add",
                   bad_dna_seqs_file,
                   codonalignment_file,
                   outdir + "aligned_gene_sequences/" + geneName + ".aln.fas"]
    #Note that the MAFFT command must be run with command[:-1] as it writes 
    # to STDOUT by default. Use capture STDOUT when running with subprocess    

    return (command, bad_dna_seqs_file)

def _check_aligner_output(returncode, stdout, stderr, outpath, what):
    """Fail loudly when an aligner writes nothing to stdout.

    mafft reports failures on stderr and returns a non-zero code; writing its
    empty stdout regardless leaves a 0-byte .aln.fas that only blows up later,
    in concatenate_core_genome_alignments, as "No records found in handle".
    """
    if returncode == 0 and stdout.strip():
        return
    detail = stderr.decode(errors="replace").strip() if stderr else ""
    raise RuntimeError(
        f"{what} failed for {outpath} (exit {returncode}): "
        + (detail or "the aligner produced no output"))


def align_sequences(command, outdir, aligner):
    #Avoid running alignments on single-isolate genes
    if command[0] == None:
        return None
    if aligner == "mafft":
        name = command[0].split()[-1].split("/")[-1].split(".")[0]
        
        proc = subprocess.Popen(
            command[0], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate()
        outpath = outdir + name + ".aln.fas"
        _check_aligner_output(proc.returncode, stdout, stderr, outpath,
                              "Alignment")

        with open(outpath, "wb+") as handle:
            handle.write(stdout)

    else:
        result = subprocess.run(
            command[0], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
    try:
        os.remove(command[1])
    except FileNotFoundError:
        None
    return True

def realign_dna_sequences(command, outdir, aligner):
    if aligner == "prank":
        raise Exception("This is a bug! Please report it. Panaroo does not " + 
                        "support codon alignment with PRANK")    
    elif aligner in {"mafft", "muscle", "muscle-super5", "famsa"}:
        result = subprocess.Popen(command[0][:-1], stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
        mafft_out, mafft_err = result.communicate()
        _check_aligner_output(result.returncode, mafft_out, mafft_err,
                              command[0][-1],
                              "Profile alignment of untranslatable DNA")
        with open(command[0][-1], 'wb') as outhandle:
            outhandle.write(mafft_out)
    elif aligner == "clustal":
        result = subprocess.run(command[0])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
    #Delete the bad DNA seqs file
    try:
        os.remove(command[1])
    except FileNotFoundError:
        None
    return True
    
def multi_align_sequences(commands, outdir, threads, aligner):
    for command in commands:
        if command == None:
            print(command)
    alignment_results = Parallel(n_jobs=threads, prefer="threads")(
        delayed(align_sequences)(x, outdir, aligner) for x in tqdm(commands)
    )

    return True

def multi_realign_sequences(commands, outdir, threads, aligner):
    
    alignment_results = Parallel(n_jobs=threads, prefer="threads")(
        delayed(realign_dna_sequences)(x, outdir, aligner) for x in tqdm(commands))
    
    return True

def replace_last(string, find, replace):
    reversed = string[::-1]
    replaced = reversed.replace(find[::-1], replace[::-1], 1)
    return replaced[::-1]

def read_sequences(handle):
    with open(handle, 'r') as inhandle:
        sequences = list(SeqIO.parse(inhandle, 'fasta'))
    return sequences

def read_alignment(handle):
    with open(handle, 'r') as inhandle:
        alignment = AlignIO.read(inhandle, 'fasta')
    return alignment


def reorder_protein_alignment_to_match_dna(dna_records, protein_alignment, gene_name):
    dna_ids = [record.id for record in dna_records]
    protein_ids = [record.id for record in protein_alignment]

    if len(set(dna_ids)) != len(dna_ids):
        raise ValueError(f"Duplicate DNA sequence IDs found for gene: {gene_name}")
    if len(set(protein_ids)) != len(protein_ids):
        raise ValueError(f"Duplicate protein sequence IDs found for gene: {gene_name}")
    if set(dna_ids) != set(protein_ids):
        raise ValueError(f"DNA and protein sequence IDs do not match for gene: {gene_name}")

    protein_by_id = {record.id: record for record in protein_alignment}
    return MultipleSeqAlignment([protein_by_id[record.id] for record in dna_records])

def thread_codons(protein_alignment, dna_records, gene_name):
    """Thread unaligned DNA onto a gapped protein alignment.

    Replaces Bio.codonalign.build, which re-derives a protein/DNA
    correspondence this pipeline has already proven and cost ~99% of the
    runtime doing it (6.1 ms per sequence, working codon by codon in Python).

    Preconditions, guaranteed by _qc_gene_sequences and
    reorder_protein_alignment_to_match_dna: the records are 1:1 and in the same
    order, and every DNA sequence is a multiple of 3 that translates to its
    protein. Each residue therefore consumes exactly three bases and each gap
    becomes '---'; any trailing stop codon is simply never consumed, matching
    codonalign's behaviour of dropping it.
    """
    records = []
    for protein_record, dna_record in zip(protein_alignment, dna_records):
        aligned_protein = str(protein_record.seq)
        dna = str(dna_record.seq)
        n_residues = len(aligned_protein) - aligned_protein.count("-")

        if len(dna) < 3 * n_residues:
            raise PangenomeSequenceError(
                f"Cannot thread codons for {gene_name}: {dna_record.id} has "
                f"{len(dna)} bases but its protein needs {3 * n_residues}. "
                "Sequences reaching this point should already have passed "
                "translation QC.")

        protein_chars = np.frombuffer(aligned_protein.encode(), dtype="S1")
        codons = np.full((protein_chars.size, 3), b"-", dtype="S1")
        codons[protein_chars != b"-"] = np.frombuffer(
            dna[:3 * n_residues].encode(), dtype="S1").reshape(n_residues, 3)

        records.append(SeqRecord(Seq(codons.tobytes().decode()),
                                 id=dna_record.id, description=""))

    return MultipleSeqAlignment(records)


def _qc_gene_sequences(dna, protein, gene_name, trans_table):
    """Split one gene's records into codon-alignable and rejected sets.

    A DNA sequence is rejected when it cannot be reverse-translated onto the
    protein alignment: length not a multiple of 3, translation disagreeing with
    the stored protein, a run of unknown nucleotides, or a degenerate codon that
    codonalign cannot handle.
    """
    protein = reorder_protein_alignment_to_match_dna(dna, protein, gene_name)
    seqids_to_remove = []

    for seq_index in range(len(dna)):
        #set up sequentially checked QC failure variables for each sequence
        fail_condition_1 = False
        fail_condition_2 = False
        fail_condition_3 = False
        #Need to take protein without proceeding or trailing gaps
        nogapped_protein_seq = str(protein[seq_index].seq).replace("-", "")

        dna_seq = str(dna[seq_index].seq)

        #fail if the sequence is not divisible by 3
        fail_condition_0 = (len(dna_seq) % 3) != 0

        if fail_condition_0 == False:
            translated_dna = translate(dna_seq, trans_table)
            #fail if the translated sequence isn't the same as the protein
            fail_condition_1 = translated_dna.strip("*") != str(nogapped_protein_seq)

        #fail if there is a run of > 1 unknown nucleotides
        if fail_condition_1 == False:
            #only test if it hasn't already failed
            fail_condition_2 = "NN" in dna_seq

        #Fail if the DNA contains degenerate codon, codonalign cannot cope
        if not fail_condition_1 and not fail_condition_2:
            #Most expensive test, only test things passing both
            #Such an expensive test, do a cheaper filtering first
            if "N" in dna_seq:
                for codon in unambiguous_degenerate_codons.keys():
                    if codon in dna_seq:
                        fail_condition_3 = True

        if fail_condition_0 or fail_condition_1 or fail_condition_2 or fail_condition_3:
            seqids_to_remove.append(dna[seq_index].id)
            seqids_to_remove.append(protein[seq_index].id)

    if not seqids_to_remove:
        return dna, protein, []

    remove = set(seqids_to_remove)
    clean_nucs = []
    reject_dna = []
    for sequence in dna:
        if sequence.id in remove:
            reject_dna.append(sequence)
        else:
            clean_nucs.append(sequence)
    clean_prots = [s for s in protein if s.id not in remove]

    return clean_nucs, MultipleSeqAlignment(clean_prots), reject_dna


def _codon_align_one_gene(protein_file, dna_file, strict, outdir,
                          temp_directory, aligner, trans_table):
    """Reverse-translate a single gene, start to finish, then free it.

    Everything this touches goes out of scope on return, so peak memory is
    bounded by (threads x largest gene) rather than by the whole pangenome.
    """
    gene_name = dna_file.split('/')[-1].split(".")[0]

    dna = read_sequences(dna_file)
    protein = read_alignment(protein_file)

    clean_dna, clean_protein, reject_dna = _qc_gene_sequences(
        dna, protein, gene_name, trans_table)
    del dna, protein

    codon_alignment = thread_codons(clean_protein, clean_dna, gene_name)
    del clean_dna, clean_protein

    #Remove <unknown description> from codon alignments
    for sequence in codon_alignment:
        sequence.description = ""

    final_outname = outdir + "aligned_gene_sequences/" + gene_name + ".aln.fas"

    #With nothing rejected the two codon modes agree; --strict-codons also stops
    #here, having simply dropped the rejected sequences.
    if not reject_dna or strict:
        AlignIO.write(codon_alignment, final_outname, 'fasta')
        return False

    #output the alignment missing some DNA sequences to tmpdir, then realign
    #the untranslatable DNA onto it
    partial_outname = temp_directory + gene_name + ".aln.fas"
    AlignIO.write(codon_alignment, partial_outname, 'fasta')
    del codon_alignment

    reject_outname = temp_directory + gene_name + "_untrans_dna.fasta"
    SeqIO.write(reject_dna, reject_outname, "fasta")
    del reject_dna

    command = get_align_dna_to_alignment_commands(
        reject_outname, partial_outname, outdir, aligner)
    #called directly rather than via multi_realign_sequences: that is itself a
    #Parallel, and nesting it inside this one would oversubscribe the pool
    realign_dna_sequences(command, outdir, aligner)
    return True


def reverse_translate_sequences(protein_sequence_files, dna_sequence_files, 
                                strict, outdir, temp_directory, aligner, 
                                threads, completed_alignments_found=0):
    outdir = _normalise_output_dir(outdir)

    #Check that the dna and protein files match up
    for index in range(len(protein_sequence_files)):
        gene_id = protein_sequence_files[index].split('/')[-1].split(".")[0]
        if gene_id == dna_sequence_files[index].split('/')[-1].split(".")[0]:
            continue
        else:
            print(protein_sequence_files[index])
            print(dna_sequence_files[index])
            raise ValueError("DNA and protien sequence IDs do not match!")

    print_stage_progress(
        "Reverse translating DNA",
        completed_alignments_found,
        len(dna_sequence_files),
    )

    trans_table = get_trans_table(11)

    # _codon_align_one_gene is CPU-bound pure Python (Biopython/numpy): with
    # prefer="threads" the GIL serialised 48 workers onto ~1 core. All arguments
    # (paths, bools, [ndarray, set]) pickle, so a process pool is safe.
    realigned = Parallel(n_jobs=threads, backend="loky")(
        delayed(_codon_align_one_gene)(
            protein_sequence_files[index],
            dna_sequence_files[index],
            strict,
            outdir,
            temp_directory,
            aligner,
            trans_table,
        )
        for index in tqdm(range(len(dna_sequence_files))))

    if not strict:
        print(str(sum(1 for x in realigned if x))
              + " genes required DNA realignment.")

    all_alignments = os.listdir(outdir + "aligned_gene_sequences/")

    return all_alignments

def write_alignment_header(alignment_list, outdir, filename):
    out_entries = []
    # Set the tracking variables for gene positions
    gene_start = 1
    gene_end = 0
    for gene in alignment_list:
        # Get length and name from one sequence in the alignment
        # Set variables that need to be set pre-output
        gene_end += gene[2]
        gene_name = gene[0]
        # Create the 3 line feature entry
        gene_entry1 = (
            "FT   feature         " + str(gene_start) + ".." + str(gene_end) + "\n"
        )
        gene_entry2 = "FT                   /label=" + gene_name + "\n"
        gene_entry3 = "FT                   /locus_tag=" + gene_name + "\n"
        gene_entry = gene_entry1 + gene_entry2 + gene_entry3
        # Add it to the output list
        out_entries.append(gene_entry)
        # Alter the post-output variables
        gene_start += gene[2]
    # Create the header and footer
    header = (
        "ID   Genome standard; DNA; PRO; 1234 BP.\nXX\nFH   Key"
        + "             Location/Qualifiers\nFH\n"
    )
    footer = (
        "XX\nSQ   Sequence 1234 BP; 789 A; 1717 C; 1693 G; 691 T;" + " 0 other;\n//\n"
    )
    # open file and output
    with open(outdir + filename, "w+") as outhandle:
        outhandle.write(header)
        for entry in out_entries:
            outhandle.write(entry)
        outhandle.write(footer)

    return True


def extract_sequences_sharded(G, output_dir, shared_dir, shard_index, shard_count,
                              procs, sqlite_path, sequences_sqlite_path):
    """Run only the codon-path sequence extraction, for one shard of the genes.

    Writes each gene's protein input to shared_dir/unaligned_protein_sequences/
    and its DNA to shared_dir/unaligned_dna_sequences/ (singletons go straight
    to output_dir/aligned_gene_sequences/, exactly as output_dna_and_protein
    does), so a subsequent full run reuses them through
    _output_dna_and_protein_or_reuse and never queries SQLite for them.

    Genes are sorted(G.nodes())[shard_index::shard_count], so shard_count
    shards partition the pangenome exactly and can run as independent jobs.
    Within a shard a process pool is used deliberately: the per-gene work is
    Python-bound, so threads only add GIL contention (measured 2-4x SLOWER
    than serial even with locking removed), whereas processes measured ~15x
    faster on one node. Pair with PANGENOMERGE_SQLITE_IMMUTABLE=1 so the
    processes do not contend on SQLite locks.
    """
    output_dir = _normalise_output_dir(output_dir)
    shared_dir = _resolve_shared_dir(output_dir, shared_dir)
    protein_dir = os.path.join(shared_dir, "unaligned_protein_sequences", "")
    for d in (protein_dir,
              os.path.join(shared_dir, "unaligned_dna_sequences"),
              os.path.join(output_dir, "aligned_gene_sequences")):
        os.makedirs(d, exist_ok=True)

    genes = sorted(G.nodes())[shard_index::shard_count]
    print(f"Extracting sequences: shard {shard_index}/{shard_count}, "
          f"{len(genes)} genes, {procs} processes", flush=True)
    results = Parallel(n_jobs=procs, backend="loky")(
        delayed(output_dna_and_protein)(
            G.nodes[gene], protein_dir, output_dir,
            sqlite_path=sqlite_path,
            sequences_sqlite_path=sequences_sqlite_path,
            shared_dir=shared_dir,
        )
        for gene in tqdm(genes))
    n_multi = sum(1 for r in results if r[0])
    print(f"Extracted {len(genes)} genes: {n_multi} multi-sequence, "
          f"{len(genes) - n_multi} singletons", flush=True)
    return len(genes)


def generate_pan_genome_alignment(G, temp_dir, output_dir, threads, aligner,
                                  codons, strict, resume=False,
                                  sqlite_path=None, sequences_sqlite_path=None,
                                  shared_dir=None):
    output_dir = _normalise_output_dir(output_dir)
    shared_dir = _resolve_shared_dir(output_dir, shared_dir)
    os.makedirs(os.path.join(output_dir, "aligned_gene_sequences"), exist_ok=True)

    gene_ids = list(G.nodes())
    node_pairs = [(gene_id, G.nodes[gene_id]) for gene_id in gene_ids]
    codon_mode = codons or strict

    pending_gene_ids = get_pending_gene_ids(
        node_pairs,
        output_dir=output_dir,
        codons=codon_mode,
        resume=resume,
        aligner=aligner,
    )
    total_gene_count = len(gene_ids)

    if codon_mode:
        protein_pending_gene_ids, reverse_translate_pending_gene_ids = (
            get_pending_codon_gene_ids(node_pairs, output_dir=output_dir,
                                       resume=resume, shared_dir=shared_dir)
        )
        print("Codon alignment is experimental in Panaroo...")
        os.makedirs(os.path.join(shared_dir, "aligned_protein_sequences"),
                    exist_ok=True)
        os.makedirs(os.path.join(shared_dir, "unaligned_dna_sequences"),
                    exist_ok=True)

        output_files = []
        for gene in protein_pending_gene_ids:
            output = _output_dna_and_protein_or_reuse(
                G.nodes[gene], temp_dir, output_dir,
                sqlite_path, sequences_sqlite_path, shared_dir,
            )
            output_files.append(output)

        filtered_output_files = [x for x in output_files if x[0]]
        unaligned_protein_files = [x[0] for x in filtered_output_files]

        commands = [
            get_protein_commands(fastafile, shared_dir, aligner, threads)
            for fastafile in unaligned_protein_files
        ]
        print_stage_progress("Protein alignments",
                             total_gene_count - len(protein_pending_gene_ids),
                             len(commands),
                             total_gene_count)
        if commands:
            multi_align_sequences(commands,
                                  shared_dir + "aligned_protein_sequences/",
                                  threads, aligner)

        protein_sequences, unaligned_dna_files = get_codon_pending_files(
            node_pairs,
            shared_dir,
            reverse_translate_pending_gene_ids)

        for file in protein_sequences:
            if os.path.isfile(file) == False:
                print(file)
                raise RuntimeError("Some alignments failed to complete!")

        if len(reverse_translate_pending_gene_ids) > 0:
            completed_final_alignments = (
                total_gene_count - len(reverse_translate_pending_gene_ids)
            )
            reverse_translate_sequences(protein_sequences,
                                        unaligned_dna_files,
                                        strict,
                                        output_dir,
                                        temp_dir,
                                        aligner,
                                        threads,
                                        completed_alignments_found=completed_final_alignments)
    else:
        if aligner == 'none':
            temp_dir = output_dir + "unaligned_gene_sequences/"
            os.makedirs(temp_dir, exist_ok=True)

        print_stage_progress("Gene alignments",
                             total_gene_count - len(pending_gene_ids),
                             len(pending_gene_ids),
                             total_gene_count)
        unaligned_sequence_files = Parallel(n_jobs=threads, prefer="threads")(
            delayed(output_sequence)(
                G.nodes[x],
                temp_dir,
                output_dir,
                sqlite_path=sqlite_path,
                sequences_sqlite_path=sequences_sqlite_path,
            )
            for x in tqdm(pending_gene_ids))

        unaligned_sequence_files = [x for x in unaligned_sequence_files if x]

        if aligner == 'none':
            print("No aligner specified. Returning unaligned gene fasta files.")
            return

        commands = [
            get_alignment_commands(fastafile, output_dir, aligner, threads)
            for fastafile in unaligned_sequence_files
        ]
        if commands:
            multi_align_sequences(commands, output_dir + "aligned_gene_sequences/",
                                  threads, aligner)
    return


def get_core_gene_nodes(G, threshold, num_isolates, subset=None):
    core_nodes = []
    for node in G.nodes():
        size = G.nodes[node].get("size", 0)
        try:
            size = float(size)
        except (TypeError, ValueError):
            size = float(get_node_sequence_count(G.nodes[node]))
        if size / float(num_isolates) >= threshold:
            core_nodes.append(node)
    if subset is not None:
        if subset > len(core_nodes):
            raise RuntimeError(f"Cannot subset core genes to {subset}, "
                               f"only {len(core_nodes)} are available!")
        random.shuffle(core_nodes)
        core_nodes = core_nodes[:subset]
    return core_nodes


def update_col_counts(col_counts, s):
    s = np.array(bytearray(s.lower().encode()), dtype=np.int8)
    s[(s != 97) & (s != 99) & (s != 103) & (s != 116)] = 110
    col_counts[0, s == 97] += 1
    col_counts[1, s == 99] += 1
    col_counts[2, s == 103] += 1
    col_counts[3, s == 116] += 1
    col_counts[4, s == 110] += 1
    return col_counts


def calc_hc(col_counts):
    with np.errstate(divide='ignore', invalid='ignore'):
        col_counts = col_counts / np.sum(col_counts, 0)
        hc = -np.nansum(col_counts[0:4, :] * np.log(col_counts[0:4, :]), 0)
    informative = np.sum(1 - col_counts[4, :])
    if informative == 0:
        return 0.0
    return np.sum((1 - col_counts[4, :]) * hc) / informative


def concatenate_core_genome_alignments(core_names, output_dir, hc_threshold):
    output_dir = _normalise_output_dir(output_dir)
    alignments_dir = os.path.join(output_dir, "aligned_gene_sequences")
    core_name_set = set(core_names)

    alignment_filenames = sorted(os.listdir(alignments_dir))
    core_filenames = [
        x for x in alignment_filenames if x.split('.')[0] in core_name_set
    ]
    if not core_filenames:
        raise RuntimeError("No core gene alignment files were found to concatenate.")

    gene_alignments = []
    isolates = set()
    for filename in core_filenames:
        gene_name = os.path.splitext(os.path.basename(filename))[0]
        alignment = AlignIO.read(os.path.join(alignments_dir, filename), "fasta")
        gene_dict = {}
        for record in alignment:
            if len(gene_dict) < 1:
                gene_length = len(record.seq)
                col_counts = np.zeros((5, gene_length), dtype=float)
            col_counts = update_col_counts(col_counts, str(record.seq))

            if record.id[:3] == "_R_":
                record.id = record.id[3:]
            genome_id = record.id.split(";")[0]
            record_seq = str(record.seq)

            if genome_id in gene_dict:
                if record_seq.count("-") < gene_dict[genome_id][1].count("-"):
                    gene_dict[genome_id] = (record.id, record_seq)
            else:
                gene_dict[genome_id] = (record.id, record_seq)

            isolates.add(genome_id)
        gene_alignments.append((gene_name, gene_dict, gene_length,
                                calc_hc(col_counts)))

    isolate_aln = []
    for iso in sorted(isolates):
        seq = ""
        for gene in gene_alignments:
            if iso in gene[1]:
                seq += gene[1][iso][1]
            else:
                seq += "-" * gene[2]
        isolate_aln.append(SeqRecord(Seq(seq), id=iso, description=""))

    SeqIO.write(isolate_aln, output_dir + "core_gene_alignment.aln", "fasta")
    write_alignment_header(gene_alignments, output_dir, "core_alignment_header.embl")

    if hc_threshold is None:
        allh = np.array([gene[3] for gene in gene_alignments])
        q = np.quantile(allh, [0.25, 0.75])
        hc_threshold = max(0.01, q[1] + 1.5 * (q[1] - q[0]))
        print(f"Entropy threshold automatically set to {hc_threshold}.")

    filtered_genes = [gene for gene in gene_alignments if gene[3] <= hc_threshold]
    isolate_aln = []
    for iso in sorted(isolates):
        seq = ""
        for gene in filtered_genes:
            if iso in gene[1]:
                seq += gene[1][iso][1]
            else:
                seq += "-" * gene[2]
        isolate_aln.append(SeqRecord(Seq(seq), id=iso, description=""))

    with open(output_dir + 'alignment_entropy.csv', 'w') as outfile:
        for g in gene_alignments:
            outfile.write(str(g[0]) + ',' + str(g[3]) + '\n')

    SeqIO.write(isolate_aln, output_dir + "core_gene_alignment_filtered.aln",
                "fasta")
    write_alignment_header(
        filtered_genes,
        output_dir,
        "core_alignment_filtered_header.embl",
    )

    print(f"{len(filtered_genes)} out of {len(gene_alignments)} genes kept in filtered core genome")

    return core_filenames


def generate_core_genome_alignment(
    G, temp_dir, output_dir, threads, aligner, threshold, codons, strict,
    num_isolates, hc_threshold, subset=None, resume=False, sqlite_path=None,
    sequences_sqlite_path=None, shared_dir=None, concatenate=False
):
    output_dir = _normalise_output_dir(output_dir)
    shared_dir = _resolve_shared_dir(output_dir, shared_dir)
    os.makedirs(os.path.join(output_dir, "aligned_gene_sequences"), exist_ok=True)

    core_genes = get_core_gene_nodes(G, threshold, num_isolates, subset)
    core_gene_names = [G.nodes[x]["name"] for x in core_genes]

    if len(core_genes) < 1:
        print("No gene clusters were present above the core frequency"
              " threshold! Try adjusting the '--core_threshold' parameter")
        return

    node_pairs = [(gene_id, G.nodes[gene_id]) for gene_id in core_genes]
    codon_mode = codons or strict
    pending_gene_ids = get_pending_gene_ids(
        node_pairs,
        output_dir=output_dir,
        codons=codon_mode,
        resume=resume,
        aligner=aligner,
    )
    total_gene_count = len(core_genes)

    if codon_mode:
        protein_pending_gene_ids, reverse_translate_pending_gene_ids = (
            get_pending_codon_gene_ids(node_pairs, output_dir=output_dir,
                                       resume=resume, shared_dir=shared_dir)
        )
        print("Codon alignment is experimental in Panaroo...")
        os.makedirs(os.path.join(shared_dir, "aligned_protein_sequences"),
                    exist_ok=True)
        os.makedirs(os.path.join(shared_dir, "unaligned_dna_sequences"),
                    exist_ok=True)

        output_files = []
        for gene in protein_pending_gene_ids:
            output = output_dna_and_protein(
                G.nodes[gene],
                temp_dir,
                output_dir,
                sqlite_path=sqlite_path,
                sequences_sqlite_path=sequences_sqlite_path,
                shared_dir=shared_dir,
            )
            output_files.append(output)

        filtered_output_files = [x for x in output_files if x[0]]
        unaligned_protein_files = [x[0] for x in filtered_output_files]

        commands = [
            get_protein_commands(fastafile, shared_dir, aligner, threads)
            for fastafile in unaligned_protein_files
        ]
        print_stage_progress("Protein alignments",
                             total_gene_count - len(protein_pending_gene_ids),
                             len(commands),
                             total_gene_count)
        if commands:
            multi_align_sequences(commands,
                                  shared_dir + "aligned_protein_sequences/",
                                  threads, aligner)

        protein_sequences, unaligned_dna_files = get_codon_pending_files(
            node_pairs,
            shared_dir,
            reverse_translate_pending_gene_ids)

        for file in protein_sequences:
            if os.path.isfile(file) == False:
                print(file)
                raise RuntimeError("Some alignments failed to complete!")

        if len(reverse_translate_pending_gene_ids) > 0:
            completed_final_alignments = (
                total_gene_count - len(reverse_translate_pending_gene_ids)
            )
            reverse_translate_sequences(protein_sequences,
                                        unaligned_dna_files,
                                        strict,
                                        output_dir,
                                        temp_dir,
                                        aligner,
                                        threads,
                                        completed_alignments_found=completed_final_alignments)
    else:
        if aligner == 'none':
            temp_dir = output_dir + "unaligned_gene_sequences/"
            os.makedirs(temp_dir, exist_ok=True)

        print_stage_progress("Gene alignments",
                             total_gene_count - len(pending_gene_ids),
                             len(pending_gene_ids),
                             total_gene_count)
        unaligned_sequence_files = Parallel(n_jobs=threads, prefer="threads")(
            delayed(output_sequence)(
                G.nodes[x],
                temp_dir,
                output_dir,
                sqlite_path=sqlite_path,
                sequences_sqlite_path=sequences_sqlite_path,
            )
            for x in tqdm(pending_gene_ids))

        if aligner == 'none':
            print("No aligner specified. Returning unaligned gene fasta files.")
            return

        unaligned_sequence_files = [x for x in unaligned_sequence_files if x]
        commands = [
            get_alignment_commands(fastafile, output_dir, aligner, threads)
            for fastafile in unaligned_sequence_files
        ]
        if commands:
            multi_align_sequences(commands, output_dir + "aligned_gene_sequences/",
                                  threads, aligner)

    if concatenate:
        concatenate_core_genome_alignments(core_gene_names, output_dir,
                                           hc_threshold)

