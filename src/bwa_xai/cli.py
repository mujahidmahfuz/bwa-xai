"""``bwa-bench``: run the comparison or stability benchmark from the shell."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from .aggregators import AGGREGATORS, DEFAULT_METHODS
from .benchmark import SYNTHETIC, make_task, paired_wilcoxon, run_comparison, run_stability, summarize
from .datasets import LOADERS
from .reporting import summary_table, write_csv

LOGGER = logging.getLogger("bwa_xai")
DEFAULT_DATASETS = (SYNTHETIC, "breast_cancer", "wine")
DATASET_CHOICES = (SYNTHETIC, *LOADERS)
REFERENCE_METHOD = "vanilla_mean"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=DATASET_CHOICES)
    parser.add_argument("--n-samples", nargs="+", type=int, default=[20, 50], help="training-set sizes N")
    parser.add_argument("--n-boot", type=int, default=50, help="bootstrap replicates B")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=sorted(AGGREGATORS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bwa-bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="attribution quality against a reference")
    _add_common(compare)
    compare.add_argument("--n-trials", type=int, default=100)
    stability = commands.add_parser("stability", help="run-to-run stability of each method")
    _add_common(stability)
    stability.add_argument("--n-datasets", type=int, default=20, help="dataset draws per cell")
    stability.add_argument("--n-runs", type=int, default=8, help="independent runs per draw")
    return parser


def _run_cell(args: argparse.Namespace, dataset: str, n_samples: int) -> list[dict[str, object]]:
    task = make_task(dataset, n_samples)
    common = {
        "n_boot": args.n_boot,
        "methods": args.methods,
        "seed": args.seed,
        "labels": {"dataset": dataset, "n_samples": n_samples},
    }
    if args.command == "compare":
        return run_comparison(task, n_trials=args.n_trials, **common)
    return run_stability(task, n_datasets=args.n_datasets, n_runs=args.n_runs, **common)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    records: list[dict[str, object]] = []
    for dataset in args.datasets:
        for n_samples in args.n_samples:
            LOGGER.info("running %s on %s (N=%d)", args.command, dataset, n_samples)
            try:
                records.extend(_run_cell(args, dataset, n_samples))
            except (ValueError, OSError, RuntimeError) as error:
                LOGGER.error("skipping %s (N=%d): %s", dataset, n_samples, error)
    if not records:
        LOGGER.error("no results were produced")
        return 1
    summary = summarize(records)
    table = summary_table(summary)
    write_csv(records, args.out / f"{args.command}_records.csv")
    write_csv(summary, args.out / f"{args.command}_summary.csv")
    tests = paired_wilcoxon(records) if REFERENCE_METHOD in args.methods else []
    if tests:
        write_csv(tests, args.out / f"{args.command}_wilcoxon.csv")
    (args.out / f"{args.command}_summary.md").write_text(table, encoding="utf-8")
    sys.stdout.write(table)
    LOGGER.info("wrote results to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
