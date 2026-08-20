#!/usr/bin/env python3
"""Validate and rank ProteinMPNN score-only outputs for mature PLA variants."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCORE_DIR = SCRIPT_DIR / "output" / "score_only"
DEFAULT_MANIFEST = SCRIPT_DIR / "input" / "PLA_all_candidates_manifest.csv"
DEFAULT_FASTA = SCRIPT_DIR / "input" / "PLA_all_candidates_mature.fasta"
DEFAULT_RESULT_DIR = SCRIPT_DIR / "result"
NPZ_PREFIX = "PLA_WIL_mature"
EXPECTED_GROUPS = ("E20", "E30", "M30")

ALL_SCORE_FIELDS = (
    "Fasta_Index",
    "NPZ_File",
    "Sequence_ID",
    "Group",
    "Source_File",
    "Source_Row",
    "PLL_Rank",
    "GA_Rank",
    "Mutation_Count",
    "Sequence_Length",
    "Mature_Sequence",
    "Sample_Size",
    "MPNN_Score_Mean",
    "MPNN_Score_SD",
    "MPNN_Global_Score_Mean",
    "MPNN_Global_Score_SD",
    "WT_MPNN_Score",
    "Delta_MPNN_vs_WT",
    "Better_Than_WT",
    "MPNN_Rank_All",
    "MPNN_Rank_Within_Group",
)

SUMMARY_FIELDS = (
    "Group",
    "Candidate_Count",
    "MPNN_Score_Mean",
    "MPNN_Score_SD_Across_Candidates",
    "MPNN_Score_Median",
    "MPNN_Score_Min",
    "Best_Sequence_ID",
    "MPNN_Score_Max",
    "Worst_Sequence_ID",
    "Mean_Delta_vs_WT",
    "Median_Delta_vs_WT",
    "Better_Than_WT_Count",
    "Better_Than_WT_Fraction",
    "WT_MPNN_Score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join ProteinMPNN NPZ score-only outputs to the mature PLA manifest, "
            "validate sequence/order integrity, and produce independent rankings."
        )
    )
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=8,
        help="Required number of repeated scores per sequence.",
    )
    return parser.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"FASTA does not exist: {path}")
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    sequence_parts: list[str] = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, "".join(sequence_parts)))
            current_id = line[1:].split()[0]
            if not current_id:
                raise ValueError(f"Empty FASTA identifier at {path}:{line_number}")
            sequence_parts = []
        else:
            if current_id is None:
                raise ValueError(f"Sequence appears before a header at {path}:{line_number}")
            sequence_parts.append(line.upper())
    if current_id is not None:
        records.append((current_id, "".join(sequence_parts)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    if len({identifier for identifier, _ in records}) != len(records):
        raise ValueError(f"Duplicate identifiers found in {path}")
    return records


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Fasta_Index",
            "Sequence_ID",
            "Group",
            "Source_File",
            "Source_Row",
            "PLL_Rank",
            "GA_Rank",
            "Mutation_Count",
            "Sequence_Length",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    for expected_index, row in enumerate(rows):
        try:
            actual_index = int(row["Fasta_Index"])
        except ValueError as exc:
            raise ValueError(f"Invalid Fasta_Index in manifest row {expected_index + 2}") from exc
        if actual_index != expected_index:
            raise ValueError(
                f"Manifest Fasta_Index is {actual_index}, expected {expected_index}"
            )
    return rows


def scalar_string(value: np.ndarray) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def load_npz_score(
    path: Path,
    expected_sequence: str,
    expected_samples: int,
) -> dict[str, float | int]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"score", "global_score", "seq_str"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path.name} is missing arrays: {sorted(missing)}")
        score = np.asarray(payload["score"], dtype=float).reshape(-1)
        global_score = np.asarray(payload["global_score"], dtype=float).reshape(-1)
        sequence = scalar_string(payload["seq_str"])

    if score.size != expected_samples:
        raise ValueError(
            f"{path.name} has {score.size} score samples; expected {expected_samples}"
        )
    if global_score.size != expected_samples:
        raise ValueError(
            f"{path.name} has {global_score.size} global-score samples; "
            f"expected {expected_samples}"
        )
    if not np.isfinite(score).all() or not np.isfinite(global_score).all():
        raise ValueError(f"{path.name} contains non-finite score values")
    if sequence != expected_sequence:
        raise ValueError(
            f"Sequence mismatch in {path.name}: NPZ has {sequence!r}, "
            f"FASTA has {expected_sequence!r}"
        )
    return {
        "Sample_Size": int(score.size),
        "MPNN_Score_Mean": float(score.mean()),
        "MPNN_Score_SD": float(score.std(ddof=0)),
        "MPNN_Global_Score_Mean": float(global_score.mean()),
        "MPNN_Global_Score_SD": float(global_score.std(ddof=0)),
    }


def validate_inputs(
    score_dir: Path,
    manifest_rows: list[dict[str, str]],
    fasta_records: list[tuple[str, str]],
) -> tuple[list[Path], Path]:
    if not score_dir.is_dir():
        raise FileNotFoundError(f"ProteinMPNN score directory does not exist: {score_dir}")
    if len(manifest_rows) != len(fasta_records):
        raise ValueError(
            f"Manifest has {len(manifest_rows)} rows, FASTA has {len(fasta_records)} records"
        )

    fasta_paths: list[Path] = []
    for index, (manifest, (fasta_id, sequence)) in enumerate(
        zip(manifest_rows, fasta_records)
    ):
        if manifest["Sequence_ID"] != fasta_id:
            raise ValueError(
                f"Manifest/FASTA identifier mismatch at index {index}: "
                f"{manifest['Sequence_ID']!r} versus {fasta_id!r}"
            )
        if int(manifest["Sequence_Length"]) != len(sequence):
            raise ValueError(
                f"Sequence length mismatch for {fasta_id}: manifest says "
                f"{manifest['Sequence_Length']}, FASTA has {len(sequence)}"
            )
        fasta_paths.append(score_dir / f"{NPZ_PREFIX}_fasta_{index + 1}.npz")

    pdb_path = score_dir / f"{NPZ_PREFIX}_pdb.npz"
    expected_paths = {pdb_path, *fasta_paths}
    actual_paths = set(score_dir.glob("*.npz"))
    missing = sorted(path.name for path in expected_paths.difference(actual_paths))
    extra = sorted(path.name for path in actual_paths.difference(expected_paths))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing[:10]}")
        if extra:
            details.append(f"extra={extra[:10]}")
        raise ValueError("Unexpected NPZ file set: " + "; ".join(details))
    return fasta_paths, pdb_path


def assign_ranks(candidate_rows: list[dict[str, object]]) -> None:
    sorted_all = sorted(
        candidate_rows,
        key=lambda row: (
            float(row["MPNN_Score_Mean"]),
            float(row["MPNN_Score_SD"]),
            str(row["Sequence_ID"]),
        ),
    )
    for rank, row in enumerate(sorted_all, start=1):
        row["MPNN_Rank_All"] = rank

    for group in EXPECTED_GROUPS:
        group_rows = [row for row in candidate_rows if row["Group"] == group]
        if not group_rows:
            raise ValueError(f"No candidate rows found for group {group}")
        group_rows.sort(
            key=lambda row: (
                float(row["MPNN_Score_Mean"]),
                float(row["MPNN_Score_SD"]),
                str(row["Sequence_ID"]),
            )
        )
        for rank, row in enumerate(group_rows, start=1):
            row["MPNN_Rank_Within_Group"] = rank


def float_text(value: object) -> str:
    return f"{float(value):.8f}"


def output_row(row: dict[str, object]) -> dict[str, object]:
    formatted = dict(row)
    for field in (
        "MPNN_Score_Mean",
        "MPNN_Score_SD",
        "MPNN_Global_Score_Mean",
        "MPNN_Global_Score_SD",
        "WT_MPNN_Score",
        "Delta_MPNN_vs_WT",
    ):
        formatted[field] = float_text(formatted[field])
    return {field: formatted.get(field, "") for field in ALL_SCORE_FIELDS}


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    candidate_rows: list[dict[str, object]], wt_score: float
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for group in (*EXPECTED_GROUPS, "ALL"):
        rows = (
            candidate_rows
            if group == "ALL"
            else [row for row in candidate_rows if row["Group"] == group]
        )
        scores = [float(row["MPNN_Score_Mean"]) for row in rows]
        deltas = [float(row["Delta_MPNN_vs_WT"]) for row in rows]
        best = min(rows, key=lambda row: float(row["MPNN_Score_Mean"]))
        worst = max(rows, key=lambda row: float(row["MPNN_Score_Mean"]))
        better_count = sum(bool(row["Better_Than_WT"]) for row in rows)
        summary_rows.append(
            {
                "Group": group,
                "Candidate_Count": len(rows),
                "MPNN_Score_Mean": float_text(mean(scores)),
                "MPNN_Score_SD_Across_Candidates": float_text(pstdev(scores)),
                "MPNN_Score_Median": float_text(median(scores)),
                "MPNN_Score_Min": float_text(min(scores)),
                "Best_Sequence_ID": best["Sequence_ID"],
                "MPNN_Score_Max": float_text(max(scores)),
                "Worst_Sequence_ID": worst["Sequence_ID"],
                "Mean_Delta_vs_WT": float_text(mean(deltas)),
                "Median_Delta_vs_WT": float_text(median(deltas)),
                "Better_Than_WT_Count": better_count,
                "Better_Than_WT_Fraction": float_text(better_count / len(rows)),
                "WT_MPNN_Score": float_text(wt_score),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    if args.expected_samples < 1:
        raise ValueError("--expected-samples must be positive")

    manifest_rows = read_manifest(args.manifest.resolve())
    fasta_records = read_fasta(args.fasta.resolve())
    fasta_paths, pdb_path = validate_inputs(
        args.score_dir.resolve(), manifest_rows, fasta_records
    )

    if manifest_rows[0]["Sequence_ID"] != "PLA_WT" or manifest_rows[0]["Group"] != "WT":
        raise ValueError("Manifest index 0 must be the PLA_WT reference")
    observed_groups = Counter(row["Group"] for row in manifest_rows)
    expected_groups = Counter({"WT": 1, "E20": 300, "E30": 300, "M30": 300})
    if observed_groups != expected_groups:
        raise ValueError(
            f"Unexpected manifest group counts: {dict(observed_groups)}; "
            f"expected {dict(expected_groups)}"
        )
    wt_sequence = fasta_records[0][1]
    pdb_score = load_npz_score(pdb_path, wt_sequence, args.expected_samples)

    score_rows: list[dict[str, object]] = []
    for manifest, (_, sequence), npz_path in zip(
        manifest_rows, fasta_records, fasta_paths
    ):
        score = load_npz_score(npz_path, sequence, args.expected_samples)
        row: dict[str, object] = {
            **manifest,
            "NPZ_File": npz_path.name,
            "Mature_Sequence": sequence,
            **score,
            "MPNN_Rank_All": "",
            "MPNN_Rank_Within_Group": "",
        }
        score_rows.append(row)

    wt_score = float(score_rows[0]["MPNN_Score_Mean"])
    for row in score_rows:
        delta = float(row["MPNN_Score_Mean"]) - wt_score
        row["WT_MPNN_Score"] = wt_score
        row["Delta_MPNN_vs_WT"] = delta
        row["Better_Than_WT"] = delta < 0.0

    candidate_rows = score_rows[1:]
    assign_ranks(candidate_rows)
    expected_candidate_count = len(score_rows) - 1
    if len(candidate_rows) != expected_candidate_count:
        raise RuntimeError("Candidate row count changed unexpectedly")

    result_dir = args.result_dir.resolve()
    all_scores_path = result_dir / "ProteinMPNN_all_scores.csv"
    ranked_path = result_dir / "ProteinMPNN_ranked_candidates.csv"
    summary_path = result_dir / "ProteinMPNN_group_summary.csv"

    write_csv(all_scores_path, ALL_SCORE_FIELDS, [output_row(row) for row in score_rows])
    ranked_rows = sorted(candidate_rows, key=lambda row: int(row["MPNN_Rank_All"]))
    write_csv(ranked_path, ALL_SCORE_FIELDS, [output_row(row) for row in ranked_rows])
    summary_rows = build_summary(candidate_rows, wt_score)
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)

    print(f"PDB WT mean score:   {float(pdb_score['MPNN_Score_Mean']):.8f}")
    print(f"FASTA WT mean score: {wt_score:.8f}")
    print(f"Validated NPZ files: {len(fasta_paths) + 1}")
    print(f"Ranked candidates:   {len(candidate_rows)}")
    for summary in summary_rows:
        print(
            f"{summary['Group']:>3}: n={summary['Candidate_Count']}, "
            f"mean={summary['MPNN_Score_Mean']}, "
            f"median={summary['MPNN_Score_Median']}, "
            f"better_than_WT={summary['Better_Than_WT_Count']}"
        )
    print(f"All scores: {all_scores_path}")
    print(f"Ranked:     {ranked_path}")
    print(f"Summary:    {summary_path}")


if __name__ == "__main__":
    main()
