import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from pangenomerge import __version__
from .generate_alignments import (
    PangenomeSequenceError,
    check_aligner_install,
    check_aligner_sanity,
    check_resume_manifest_collision,
    check_shared_manifest,
    concatenate_core_genome_alignments,
    extract_sequences_sharded,
    generate_core_genome_alignment,
    generate_pan_genome_alignment,
    get_core_gene_nodes,
    load_isolate_names,
    load_pangenomerge_alignment_graph,
    write_resume_manifest,
)


def _path_from_results(results_dir, filename):
    if results_dir is None:
        return None
    return str(Path(results_dir) / filename)


def get_options(argv=None):
    description = "Generate Panaroo-style MSAs from Pangenomerge output"
    parser = argparse.ArgumentParser(
        description=description,
        prog="pangenomerge-msa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_opts = parser.add_argument_group("Input/output")
    io_opts.add_argument(
        "-o",
        "--outdir",
        dest="pangenomerge_results",
        default=None,
        help=(
            "Pangenomerge results directory. Defaults --gml, --sqlite, "
            "--sequences-sqlite to files in this directory. Optionally,"
            "supply them separately."
        ),
    )
    io_opts.add_argument(
        "--pangenomerge-results",
        dest="pangenomerge_results",
        default=None,
        help=argparse.SUPPRESS,
    )
    io_opts.add_argument("--sqlite", default=None,
                         help=argparse.SUPPRESS)
    io_opts.add_argument("--gml", default=None,
                         help=argparse.SUPPRESS)
    io_opts.add_argument(
        "--sequences-sqlite",
        default=None,
        dest="sequences_sqlite",
        help=argparse.SUPPRESS,
    )
    io_opts.add_argument(
        "--alignment-outdir",
        default=None,
        help=(
            "Output directory. Defaults to --outdir directory containing --sqlite"
            "or when that option is provided."
        ),
    )

    io_opts.add_argument(
        "--shared-alignment-dir",
        default=None,
        dest="shared_alignment_dir",
        help=(
            "Directory for intermediates that do not depend on --strict-codons "
            "(aligned_protein_sequences/, unaligned_dna_sequences/). Point a "
            "--codons and a --strict-codons run at the same directory to align "
            "the proteins once instead of twice. Defaults to the alignment "
            "output directory."
        ),
    )

    aln_opts = parser.add_argument_group("Gene alignment")
    aln_opts.add_argument(
        "--alignment",
        choices=["core", "pan"],
        default="core",
        help="Generate core or pan genome per-gene alignments.",
    )
    aln_opts.add_argument("-a",
        "--aligner",
        choices=["mafft", "muscle", "muscle-super5", "famsa", "none"],
        default=None,
        help=(
            "External aligner to use, or 'none' to write unaligned FASTA files. "
            "Defaults to famsa for codon alignments, which is far faster than "
            "mafft on the large gene families found in a species pangenome, and "
            "to mafft otherwise -- famsa has no nucleotide mode, so it cannot "
            "align DNA. Note mafft is still required for codon runs without "
            "--strict-codons, which re-align untranslatable DNA with 'mafft "
            "--add'."
        ),
    )
    aln_opts.add_argument(
        "--codons",
        action="store_true",
        default=False,
        help="Generate codon alignments by aligning amino-acid sequences first.",
    )
    aln_opts.add_argument(
        "--strict-codons",
        action="store_true",
        default=False,
        dest="strict_codons",
        help="Only generate codon alignments with well-formed protein sequences,"
        "no DNA re-alignment of pseudogenes or frameshifts.",
    )
    aln_opts.add_argument(
        "--core_threshold",
        type=float,
        default=0.95,
        help="Core-genome frequency threshold.",
    )
    aln_opts.add_argument(
        "--core_subset",
        type=int,
        default=None,
        help="Randomly subset the core genome to this many genes.",
    )
    aln_opts.add_argument(
        "--core_entropy_filter",
        type=float,
        default=None,
        help=(
            "Manual Block Mapping and Gathering with Entropy filter. "
            "If omitted, the Tukey outlier rule is used. Only applies with "
            "--core-alignment."
        ),
    )
    aln_opts.add_argument(
        "--core-alignment",
        action="store_true",
        default=False,
        dest="core_alignment",
        help=(
            "Also write the concatenated core genome alignment "
            "(core_gene_alignment.aln and friends). Off by default: the "
            "concatenation holds every core gene alignment in memory at once, "
            "which on a large pangenome means tens of GB of RAM and an "
            "output file to match. Per-gene alignments are always written."
        ),
    )
    aln_opts.add_argument(
        "--extract-only", action="store_true", default=False,
        help="Only extract per-gene protein/DNA inputs for the pan codon "
             "alignment, for one shard of the genes (see --shard), into the "
             "shared alignment directory; a later full run reuses them. "
             "Lets the extraction run as many independent jobs.")
    aln_opts.add_argument(
        "--shard", default=None, metavar="I/K",
        help="With --extract-only: process genes I, I+K, I+2K, ... of the "
             "sorted node list, 0 <= I < K (default 0/1 = all genes).")
    aln_opts.add_argument(
        "--extract-procs", type=int, default=None,
        help="With --extract-only: worker processes (default: --threads).")
    aln_opts.add_argument(
        "--immutable-db", action="store_true", default=False,
        help="Open the SQLite databases with immutable=1 (no locking). Safe "
             "because the msa never writes them; required for many concurrent "
             "readers to scale. Ignored, with a warning, if a WAL file is "
             "non-empty, since immutable mode cannot see uncheckpointed data.")
    aln_opts.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume a previously incomplete gene alignment run.",
    )

    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Number of worker threads to use.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print additional progress information.")
    parser.add_argument("--version", action="version",
                        version="%(prog)s " + __version__)

    args = parser.parse_args(argv)
    _resolve_and_validate_paths(parser, args)

    if args.shard is None:
        args.shard = (0, 1)
    else:
        try:
            shard_index, shard_count = (int(x) for x in args.shard.split("/"))
        except ValueError:
            parser.error("--shard must look like I/K, e.g. 3/32")
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            parser.error("--shard I/K needs K >= 1 and 0 <= I < K")
        args.shard = (shard_index, shard_count)
    if args.extract_only:
        if args.alignment != "pan" or not (args.codons or args.strict_codons):
            parser.error("--extract-only applies to --alignment pan with "
                         "--codons or --strict-codons")
        if args.extract_procs is None:
            args.extract_procs = args.threads
        if args.extract_procs < 1:
            parser.error("--extract-procs must be at least 1")
    elif args.shard != (0, 1) or args.extract_procs is not None:
        parser.error("--shard and --extract-procs require --extract-only")
    return args


def _resolve_and_validate_paths(parser, args):
    results_dir = Path(args.pangenomerge_results) if args.pangenomerge_results else None
    if results_dir is not None and not results_dir.is_dir():
        parser.error(f"--outdir is not a directory: {results_dir}")

    if args.sqlite is None:
        args.sqlite = _path_from_results(results_dir, "pangenome_metadata.sqlite")
    if args.gml is None:
        args.gml = _path_from_results(results_dir, "final_graph.gml")
    if args.sequences_sqlite is None:
        args.sequences_sqlite = _path_from_results(results_dir, "pangenome_sequences.sqlite")
        if args.sequences_sqlite is None and args.sqlite is not None:
            args.sequences_sqlite = str(Path(args.sqlite).parent / "pangenome_sequences.sqlite")

    if args.sqlite is None:
        parser.error("must specify either --outdir or --sqlite")
    if args.gml is None:
        parser.error("must specify either --outdir or --gml")

    for option, value in [("--sqlite", args.sqlite), ("--gml", args.gml)]:
        if not Path(value).is_file():
            parser.error(f"{option} file was not found: {value}")

    if args.sequences_sqlite is None or not Path(args.sequences_sqlite).is_file():
        parser.error(
            "pangenome_sequences.sqlite was not found. Run "
            "pangenomerge-postprocess --output sequences first, or pass "
            "--sequences-sqlite. Expected: " + str(args.sequences_sqlite)
        )

    if args.aligner is None:
        # famsa only aligns amino acids, so it can only be the default on the
        # codon path; the DNA path stays on mafft.
        args.aligner = "famsa" if (args.codons or args.strict_codons) else "mafft"

    if args.core_threshold < 0 or args.core_threshold > 1:
        parser.error("--core_threshold must be between 0 and 1")
    if args.core_entropy_filter is not None and (
        args.core_entropy_filter < 0 or args.core_entropy_filter > 1
    ):
        parser.error("--core_entropy_filter must be between 0 and 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    if args.alignment_outdir is not None:
        output_dir = args.alignment_outdir
    elif results_dir is not None:
        output_dir = str(results_dir)
    else:
        output_dir = str(Path(args.sqlite).parent)

    args.output_dir = os.path.join(str(output_dir), "")

    if args.shared_alignment_dir is not None:
        args.shared_alignment_dir = os.path.join(
            str(args.shared_alignment_dir), "")


def _enable_immutable_sqlite(db_paths):
    """Turn on immutable SQLite access unless a WAL still holds unflushed data.

    immutable=1 reads the main database file only, so any content still in a
    -wal file would be invisible. The databases are normally fully checkpointed
    (the writer closed cleanly and the -wal is empty or absent); if not, fall
    back to the default locking mode rather than risk missing sequences.
    """
    for db_path in db_paths:
        wal = f"{db_path}-wal"
        if os.path.exists(wal) and os.path.getsize(wal) > 0:
            sys.stderr.write(
                f"pangenomerge-msa: warning: {wal} is non-empty; not using "
                "immutable SQLite mode\n")
            return False
    os.environ["PANGENOMERGE_SQLITE_IMMUTABLE"] = "1"
    return True


def main(argv=None):
    args = get_options(argv)
    try:
        isolate_names = load_isolate_names(args.sqlite)
        check_aligner_sanity(
            args.aligner,
            args.codons or args.strict_codons,
            len(isolate_names),
        )
        if args.aligner != "none":
            check_aligner_install(args.aligner)

        if args.immutable_db:
            _enable_immutable_sqlite([args.sqlite, args.sequences_sqlite])

        graph = load_pangenomerge_alignment_graph(args.sqlite, args.gml)
        os.makedirs(args.output_dir, exist_ok=True)
        if args.shared_alignment_dir is not None:
            os.makedirs(args.shared_alignment_dir, exist_ok=True)
            check_shared_manifest(args.shared_alignment_dir, args.aligner)

        if args.extract_only:
            # No resume manifest and no temp dir: this writes only persistent
            # per-gene inputs, which the full run picks up on its own.
            shard_index, shard_count = args.shard
            extract_sequences_sharded(
                graph,
                args.output_dir,
                args.shared_alignment_dir,
                shard_index,
                shard_count,
                args.extract_procs,
                args.sqlite,
                args.sequences_sqlite,
            )
            return 0

        temp_dir = os.path.join(tempfile.mkdtemp(dir=args.output_dir), "")

        try:
            check_resume_manifest_collision(args.output_dir, args.resume,
                                            args.shared_alignment_dir)
            write_resume_manifest(
                output_dir=args.output_dir,
                alignment=args.alignment,
                aligner=args.aligner,
                codons=args.codons,
                strict_codons=args.strict_codons,
                core_threshold=args.core_threshold,
                subset=args.core_subset if args.alignment == "core" else None,
                resume=args.resume,
            )

            if args.alignment == "pan":
                if args.verbose:
                    print("generating pan genome MSAs...")
                generate_pan_genome_alignment(
                    graph,
                    temp_dir,
                    args.output_dir,
                    args.threads,
                    args.aligner,
                    args.codons,
                    args.strict_codons,
                    resume=args.resume,
                    sqlite_path=args.sqlite,
                    sequences_sqlite_path=args.sequences_sqlite,
                    shared_dir=args.shared_alignment_dir,
                )
                if args.aligner != "none" and args.core_alignment:
                    core_nodes = get_core_gene_nodes(
                        graph,
                        args.core_threshold,
                        len(isolate_names),
                    )
                    core_names = [graph.nodes[x]["name"] for x in core_nodes]
                    concatenate_core_genome_alignments(
                        core_names,
                        args.output_dir,
                        args.core_entropy_filter,
                    )
            else:
                if args.verbose:
                    print("generating core genome MSAs...")
                generate_core_genome_alignment(
                    graph,
                    temp_dir,
                    args.output_dir,
                    args.threads,
                    args.aligner,
                    args.core_threshold,
                    args.codons,
                    args.strict_codons,
                    len(isolate_names),
                    args.core_entropy_filter,
                    args.core_subset,
                    resume=args.resume,
                    concatenate=args.core_alignment,
                    sqlite_path=args.sqlite,
                    sequences_sqlite_path=args.sequences_sqlite,
                    shared_dir=args.shared_alignment_dir,
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    except (OSError, RuntimeError, PangenomeSequenceError) as exc:
        sys.stderr.write(f"pangenomerge-msa: error: {exc}\n")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
