"""Write benchmark records to CSV and render summaries as Markdown tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

SUMMARY_VALUE_KEYS = ("metric", "mean", "sem", "n")
DECIMALS = 3

Record = dict[str, object]


def write_csv(rows: Sequence[Record], path: str | Path) -> Path:
    """Write ``rows`` to ``path`` with one column per key; returns the path."""
    if not rows:
        raise ValueError("no rows to write")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _format_cell(mean: float, sem: float) -> str:
    if mean != mean:  # NaN
        return "n/a"
    if sem != sem:
        return f"{mean:.{DECIMALS}f}"
    return f"{mean:.{DECIMALS}f} ± {sem:.{DECIMALS}f}"


def summary_table(summary: Sequence[Record]) -> str:
    """Markdown table of ``mean ± SEM``: one row per group, one column per metric.

    ``summary`` is the output of :func:`bwa_xai.benchmark.summarize`.
    """
    if not summary:
        raise ValueError("summary is empty")
    group_columns = [key for key in summary[0] if key not in SUMMARY_VALUE_KEYS]
    metrics = list(dict.fromkeys(str(row["metric"]) for row in summary))
    cells: dict[tuple, dict[str, str]] = {}
    for row in summary:
        group = tuple(row[column] for column in group_columns)
        cells.setdefault(group, {})[str(row["metric"])] = _format_cell(
            float(row["mean"]), float(row["sem"])
        )
    header = group_columns + metrics
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for group, values in cells.items():
        row_cells = [str(item) for item in group] + [values.get(metric, "n/a") for metric in metrics]
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines) + "\n"
