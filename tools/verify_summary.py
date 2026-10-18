#!/usr/bin/env python3
"""Check published aggregate arithmetic and snapshot integrity without inference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys


def verify(root: Path) -> None:
    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    total = evaluation["spans"] * evaluation["reference_positions_per_span"]
    if total != evaluation["reference_positions_total"]:
        raise ValueError("reference denominator mismatch")

    rows = summary["methods"]
    if [row["method"] for row in rows] != ["FC-L", "FC-F", "CP-L", "CP-F"]:
        raise ValueError("unexpected method order")
    integer_fields = (
        "views_per_span", "spans", "reference_positions", "matched_positions",
        "substitutions", "insertions", "deletions", "total_edits",
        "prediction_positions", "exact_spans",
    )
    for row in rows:
        if any(type(row[key]) is not int or row[key] < 0 for key in integer_fields):
            raise ValueError("counts must be nonnegative integers")
        if row["reference_positions"] != total or row["spans"] != evaluation["spans"]:
            raise ValueError("method denominator mismatch")
        if row["views_per_span"] != evaluation["overlap_views_per_span"]:
            raise ValueError("view count mismatch")
        if row["matched_positions"] + row["substitutions"] + row["deletions"] != total:
            raise ValueError("M + S + D != N")
        if row["substitutions"] + row["insertions"] + row["deletions"] != row["total_edits"]:
            raise ValueError("S + I + D != total edits")
        if row["matched_positions"] + row["substitutions"] + row["insertions"] != row["prediction_positions"]:
            raise ValueError("prediction length mismatch")
        if row["exact_spans"] > row["spans"]:
            raise ValueError("exact span count exceeds span count")

    with (root / "results/summary.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_rows = []
    for row in rows:
        expected_rows.append({
            **{key: str(value) for key, value in row.items()},
            "recovery_percent": f"{100 * row['matched_positions'] / total:.6f}",
            "exact_span_percent": f"{100 * row['exact_spans'] / row['spans']:.6f}",
        })
    if csv_rows != expected_rows:
        raise ValueError("CSV and JSON disagree")

    provenance = summary["provenance"]
    paper = root / provenance["paper_path"]
    if paper.name != "ICISC_2026_paper__52.pdf":
        raise ValueError("unexpected paper filename")
    if hashlib.sha256(paper.read_bytes()).hexdigest() != provenance["paper_sha256"]:
        raise ValueError("paper checksum mismatch")

    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        relative = Path(record["published_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe manifest path")
        content = (root / relative).read_bytes()
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"snapshot mismatch: {relative}")

    lower, fidelity = rows[:2]
    delta = fidelity["matched_positions"] - lower["matched_positions"]
    print(f"Reference positions: {evaluation['spans']} x {evaluation['reference_positions_per_span']} = {total}")
    print(f"FC-F minus FC-L: +{delta} matches, +{100 * delta / total:.6f} percentage points")
    print(f"Total edits: {lower['total_edits']} -> {fidelity['total_edits']}")
    print(f"PASS: aggregate arithmetic, CSV, paper checksum, and {len(manifest['files'])} copied files")
    print("Scope: publication integrity only; no research code, tests, or inference executed.")


if __name__ == "__main__":
    try:
        verify(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
