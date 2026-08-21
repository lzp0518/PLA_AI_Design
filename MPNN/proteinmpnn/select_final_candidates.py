#!/usr/bin/env python3
"""Build a final PLA shortlist that preserves two independent selection signals.

The script keeps the existing PLL/GA-selected FASTA candidates as
``PLL_GA_CHAMPION`` records, adds distinct ``THREE_METRIC_BALANCED`` records
selected from the Pareto front of PLL rank, GA rank, and within-group
ProteinMPNN rank, and retains selected PLL rank-1 sequences as explicit
``PLL_TOP1_CONTROL`` records.  No weighted composite score is used.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_RANKED_CSV = SCRIPT_DIR / "result" / "ProteinMPNN_ranked_candidates.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "final_selection"
EXPECTED_GROUPS = ("E20", "E30", "M30")
DEFAULT_PLL_TOP1_GROUPS = ("E20", "E30")
CHAMPION_DIRS = {
    "E20": PROJECT_DIR / "ESM_GA" / "01_ESM_GA_20" / "result",
    "E30": PROJECT_DIR / "ESM_GA" / "02_ESM_GA_30" / "result",
    "M30": PROJECT_DIR / "ESM_GA" / "02_ESM_MPNN_GA" / "result",
}
CHAMPION_PATTERN = re.compile(
    r"^PLA_(E20|E30|M30)_([A-Z]+)_PLL(\d+)_GA(\d+)\.fasta$"
)
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

OUTPUT_FIELDS = (
    "Final_Order",
    "Group",
    "Selection_Class",
    "Class_Order",
    "Export_ID",
    "Sequence_ID",
    "PLL_Rank",
    "GA_Rank",
    "MPNN_Rank_Within_Group",
    "MPNN_Score_Mean",
    "MPNN_Score_SD",
    "WT_MPNN_Score",
    "Delta_MPNN_vs_WT",
    "Better_Than_WT",
    "Pareto_Optimal_Within_Cutoffs",
    "Worst_Of_Three_Ranks",
    "Sum_Of_Three_Ranks",
    "Min_Sequence_Distance_Within_Class",
    "Min_Sequence_Distance_In_Final_Group",
    "Mutation_Count",
    "Selection_Reason",
    "Mature_Sequence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retain existing PLL/GA champions and add unweighted Pareto-balanced "
            "PLL/GA/ProteinMPNN candidates for each PLA experiment group."
        )
    )
    parser.add_argument("--ranked-csv", type=Path, default=DEFAULT_RANKED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--champions-per-group", type=int, default=3)
    parser.add_argument("--balanced-per-group", type=int, default=3)
    parser.add_argument("--max-pll-rank", type=int, default=100)
    parser.add_argument("--max-ga-rank", type=int, default=100)
    parser.add_argument(
        "--pll-top1-groups",
        nargs="*",
        choices=EXPECTED_GROUPS,
        default=DEFAULT_PLL_TOP1_GROUPS,
        help=(
            "Groups whose PLL rank-1 sequence is retained as a model-specific "
            "control (default: E20 E30). Pass the flag with no groups to disable."
        ),
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=4,
        help=(
            "Minimum mature-sequence Hamming distance between candidates in "
            "the same selection class and group (default: 4)."
        ),
    )
    args = parser.parse_args()
    for name in (
        "champions_per_group",
        "balanced_per_group",
        "max_pll_rank",
        "max_ga_rank",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.min_distance < 0:
        parser.error("--min-distance cannot be negative")
    return args


def read_fasta(path: Path) -> tuple[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Invalid FASTA file: {path}")
    identifier = lines[0][1:].split()[0]
    sequence = "".join(lines[1:]).upper()
    if not sequence or set(sequence).difference(VALID_AMINO_ACIDS):
        raise ValueError(f"Invalid amino-acid sequence in {path}")
    return identifier, sequence


def int_field(row: dict[str, str], field: str) -> int:
    try:
        value = int(float(row[field]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for {row.get('Sequence_ID', '<unknown>')}") from exc
    if value < 1:
        raise ValueError(f"{field} must be positive for {row['Sequence_ID']}")
    return value


def float_field(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for {row.get('Sequence_ID', '<unknown>')}") from exc


def read_ranked_rows(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ranked ProteinMPNN CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Sequence_ID",
            "Group",
            "PLL_Rank",
            "GA_Rank",
            "Mutation_Count",
            "Mature_Sequence",
            "MPNN_Rank_Within_Group",
            "MPNN_Score_Mean",
            "MPNN_Score_SD",
            "WT_MPNN_Score",
            "Delta_MPNN_vs_WT",
            "Better_Than_WT",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        raw_rows = list(reader)

    rows: dict[str, dict[str, object]] = {}
    wt_scores: set[float] = set()
    for raw in raw_rows:
        sequence_id = raw["Sequence_ID"].strip()
        group = raw["Group"].strip()
        if group not in EXPECTED_GROUPS:
            raise ValueError(f"Unexpected group {group!r} for {sequence_id}")
        sequence = raw["Mature_Sequence"].strip().upper()
        if len(sequence) != 269 or set(sequence).difference(VALID_AMINO_ACIDS):
            raise ValueError(f"Invalid 269-aa mature sequence for {sequence_id}")
        if sequence_id in rows:
            raise ValueError(f"Duplicate Sequence_ID: {sequence_id}")
        wt_score = float_field(raw, "WT_MPNN_Score")
        wt_scores.add(wt_score)
        rows[sequence_id] = {
            **raw,
            "Sequence_ID": sequence_id,
            "Group": group,
            "PLL_Rank": int_field(raw, "PLL_Rank"),
            "GA_Rank": int_field(raw, "GA_Rank"),
            "Mutation_Count": int_field(raw, "Mutation_Count"),
            "MPNN_Rank_Within_Group": int_field(raw, "MPNN_Rank_Within_Group"),
            "MPNN_Score_Mean": float_field(raw, "MPNN_Score_Mean"),
            "MPNN_Score_SD": float_field(raw, "MPNN_Score_SD"),
            "WT_MPNN_Score": wt_score,
            "Delta_MPNN_vs_WT": float_field(raw, "Delta_MPNN_vs_WT"),
            "Better_Than_WT": raw["Better_Than_WT"].strip().lower() == "true",
            "Mature_Sequence": sequence,
        }
    if Counter(row["Group"] for row in rows.values()) != Counter(
        {"E20": 300, "E30": 300, "M30": 300}
    ):
        raise ValueError("Ranked table must contain exactly 300 candidates per group")
    if len(wt_scores) != 1:
        raise ValueError(f"Expected one WT score, found {sorted(wt_scores)}")
    return rows, {"WT_MPNN_Score": wt_scores.pop()}


def label_key(label: str) -> tuple[int, ...]:
    return tuple(ord(character) - ord("A") + 1 for character in label)


def load_champions(
    group: str,
    ranked_rows: dict[str, dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    directory = CHAMPION_DIRS[group]
    matches: list[tuple[str, Path, re.Match[str]]] = []
    for path in directory.glob(f"PLA_{group}_*_PLL*_GA*.fasta"):
        match = CHAMPION_PATTERN.fullmatch(path.name)
        if match:
            matches.append((match.group(2), path, match))
    matches.sort(key=lambda item: label_key(item[0]))
    if len(matches) < limit:
        raise ValueError(f"Found {len(matches)} champion FASTA files for {group}; need {limit}")

    champions: list[dict[str, object]] = []
    for class_order, (_, path, match) in enumerate(matches[:limit], start=1):
        fasta_id, sequence = read_fasta(path)
        expected_id = f"PLA_{group}_PLL{int(match.group(3))}_GA{int(match.group(4))}"
        if expected_id not in ranked_rows:
            raise ValueError(f"Champion {path.name} is absent from the ranked CSV")
        row = dict(ranked_rows[expected_id])
        if sequence != row["Mature_Sequence"]:
            raise ValueError(f"Champion FASTA sequence does not match CSV: {path}")
        row.update(
            {
                "Selection_Class": "PLL_GA_CHAMPION",
                "Class_Order": class_order,
                "Export_ID": fasta_id,
                "Pareto_Optimal_Within_Cutoffs": "",
                "Selection_Reason": (
                    "Retained from the existing PLL/GA rank-and-diversity selection; "
                    "ProteinMPNN is reported independently and did not determine retention."
                ),
            }
        )
        champions.append(row)
    return champions


def dominates(first: dict[str, object], second: dict[str, object]) -> bool:
    fields = ("PLL_Rank", "GA_Rank", "MPNN_Rank_Within_Group")
    no_worse = all(int(first[field]) <= int(second[field]) for field in fields)
    strictly_better = any(int(first[field]) < int(second[field]) for field in fields)
    return no_worse and strictly_better


def pareto_front(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]


def sequence_distance(first: str, second: str) -> int:
    if len(first) != len(second):
        raise ValueError("Cannot compare sequences of different lengths")
    return sum(left != right for left, right in zip(first, second))


def balance_key(row: dict[str, object]) -> tuple[int, int, int, int, int, str]:
    ranks = (
        int(row["PLL_Rank"]),
        int(row["GA_Rank"]),
        int(row["MPNN_Rank_Within_Group"]),
    )
    return (max(ranks), sum(ranks), ranks[2], ranks[0], ranks[1], str(row["Sequence_ID"]))


def load_balanced(
    group: str,
    ranked_rows: dict[str, dict[str, object]],
    champions: list[dict[str, object]],
    limit: int,
    max_pll_rank: int,
    max_ga_rank: int,
    min_distance: int,
) -> list[dict[str, object]]:
    pool = [
        row
        for row in ranked_rows.values()
        if row["Group"] == group
        and int(row["PLL_Rank"]) <= max_pll_rank
        and int(row["GA_Rank"]) <= max_ga_rank
    ]
    if not pool:
        raise ValueError(f"No {group} candidates pass the PLL/GA cutoffs")
    front = pareto_front(pool)
    front_ids = {str(row["Sequence_ID"]) for row in front}
    champion_ids = {str(row["Sequence_ID"]) for row in champions}
    ordered = sorted(front, key=balance_key) + sorted(
        [row for row in pool if row["Sequence_ID"] not in front_ids], key=balance_key
    )

    selected: list[dict[str, object]] = []
    for candidate in ordered:
        if candidate["Sequence_ID"] in champion_ids:
            continue
        # The champion and balanced panels represent independent model signals.
        # Enforcing distance from champions can force a genuinely balanced
        # candidate far down the ProteinMPNN ranking, so diversity is enforced
        # within the new balanced panel while all cross-panel distances remain
        # visible in the output table.
        existing = selected
        if all(
            sequence_distance(
                str(candidate["Mature_Sequence"]), str(other["Mature_Sequence"])
            )
            >= min_distance
            for other in existing
        ):
            row = dict(candidate)
            is_pareto = candidate["Sequence_ID"] in front_ids
            row.update(
                {
                    "Selection_Class": "THREE_METRIC_BALANCED",
                    "Class_Order": len(selected) + 1,
                    "Export_ID": str(candidate["Sequence_ID"]),
                    "Pareto_Optimal_Within_Cutoffs": is_pareto,
                    "Selection_Reason": (
                        "Unweighted Pareto-front balance of PLL rank, GA rank, and "
                        "within-group ProteinMPNN rank, with sequence diversity."
                        if is_pareto
                        else
                        "Best remaining unweighted three-rank balance after Pareto "
                        "candidates, needed to satisfy count and sequence diversity."
                    ),
                }
            )
            selected.append(row)
            if len(selected) == limit:
                return selected
    raise RuntimeError(
        f"Only {len(selected)} distinct balanced candidates found for {group}; "
        f"requested {limit} with min_distance={min_distance}."
    )


def load_pll_top1_control(
    group: str,
    ranked_rows: dict[str, dict[str, object]],
    existing: list[dict[str, object]],
) -> dict[str, object]:
    matches = [
        row
        for row in ranked_rows.values()
        if row["Group"] == group and int(row["PLL_Rank"]) == 1
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one PLL rank-1 candidate for {group}; found {len(matches)}"
        )
    candidate = matches[0]
    if candidate["Sequence_ID"] in {row["Sequence_ID"] for row in existing}:
        raise ValueError(
            f"PLL rank-1 candidate for {group} is already in another selection class: "
            f"{candidate['Sequence_ID']}"
        )
    row = dict(candidate)
    row.update(
        {
            "Selection_Class": "PLL_TOP1_CONTROL",
            "Class_Order": 1,
            "Export_ID": str(candidate["Sequence_ID"]),
            "Pareto_Optimal_Within_Cutoffs": "",
            "Selection_Reason": (
                "Retained as the group PLL rank-1 model-specific control; GA and "
                "ProteinMPNN metrics are reported but did not determine retention."
            ),
        }
    )
    return row


def add_group_metrics(rows: list[dict[str, object]]) -> None:
    for row in rows:
        ranks = (
            int(row["PLL_Rank"]),
            int(row["GA_Rank"]),
            int(row["MPNN_Rank_Within_Group"]),
        )
        row["Worst_Of_Three_Ranks"] = max(ranks)
        row["Sum_Of_Three_Ranks"] = sum(ranks)
        class_distances = [
            sequence_distance(str(row["Mature_Sequence"]), str(other["Mature_Sequence"]))
            for other in rows
            if other is not row and other["Selection_Class"] == row["Selection_Class"]
        ]
        group_distances = [
            sequence_distance(str(row["Mature_Sequence"]), str(other["Mature_Sequence"]))
            for other in rows
            if other is not row
        ]
        row["Min_Sequence_Distance_Within_Class"] = (
            min(class_distances) if class_distances else ""
        )
        row["Min_Sequence_Distance_In_Final_Group"] = (
            min(group_distances) if group_distances else ""
        )


def format_output_row(row: dict[str, object]) -> dict[str, object]:
    result = {field: row.get(field, "") for field in OUTPUT_FIELDS}
    for field in (
        "MPNN_Score_Mean",
        "MPNN_Score_SD",
        "WT_MPNN_Score",
        "Delta_MPNN_vs_WT",
    ):
        result[field] = f"{float(result[field]):.8f}"
    return result


def wrap_sequence(sequence: str, width: int = 80) -> list[str]:
    return [sequence[index : index + width] for index in range(0, len(sequence), width)]


def write_outputs(
    output_dir: Path,
    final_rows: list[dict[str, object]],
    wt_score: float,
    wt_sequence: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "PLA_final_experimental_candidates.csv"
    fasta_path = output_dir / "PLA_final_experimental_candidates_with_WT.fasta"
    summary_path = output_dir / "PLA_final_experimental_summary.txt"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(format_output_row(row) for row in final_rows)

    fasta_lines = [f">PLA_WT|ProteinMPNN={wt_score:.8f}", *wrap_sequence(wt_sequence)]
    for row in final_rows:
        fasta_lines.extend(
            [
                (
                    f">{row['Export_ID']}|class={row['Selection_Class']}"
                    f"|PLL={row['PLL_Rank']}|GA={row['GA_Rank']}"
                    f"|MPNN_group={row['MPNN_Rank_Within_Group']}"
                    f"|delta_WT={float(row['Delta_MPNN_vs_WT']):+.8f}"
                ),
                *wrap_sequence(str(row["Mature_Sequence"])),
            ]
        )
    fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8", newline="\n")

    summary_lines = [
        "PLA final experimental candidate selection",
        "==========================================",
        f"WT ProteinMPNN score: {wt_score:.8f}",
        "ProteinMPNN direction: lower is better.",
        "No weighted composite score was used.",
        "M30 ProteinMPNN values are compared within group because its generation used ProteinMPNN information.",
        "E20 and E30 PLL rank-1 sequences are retained as model-specific controls.",
        "Sequence diversity is enforced within multi-candidate selection classes; cross-class distance is reported, not optimized.",
        "",
    ]
    for group in EXPECTED_GROUPS:
        summary_lines.append(f"[{group}]")
        for row in (item for item in final_rows if item["Group"] == group):
            summary_lines.append(
                f"{row['Selection_Class']:<23} {row['Export_ID']:<28} "
                f"PLL={int(row['PLL_Rank']):3d} GA={int(row['GA_Rank']):3d} "
                f"MPNN_group={int(row['MPNN_Rank_Within_Group']):3d} "
                f"score={float(row['MPNN_Score_Mean']):.8f} "
                f"delta_WT={float(row['Delta_MPNN_vs_WT']):+.8f} "
                f"within_class_distance={row['Min_Sequence_Distance_Within_Class']} "
                f"whole_group_distance={row['Min_Sequence_Distance_In_Final_Group']}"
            )
        summary_lines.append("")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8", newline="\n")
    return csv_path, fasta_path, summary_path


def main() -> None:
    args = parse_args()
    ranked_rows, wt = read_ranked_rows(args.ranked_csv.resolve())
    final_rows: list[dict[str, object]] = []

    for group in EXPECTED_GROUPS:
        champions = load_champions(group, ranked_rows, args.champions_per_group)
        balanced = load_balanced(
            group,
            ranked_rows,
            champions,
            args.balanced_per_group,
            args.max_pll_rank,
            args.max_ga_rank,
            args.min_distance,
        )
        group_rows = [*champions, *balanced]
        if group in args.pll_top1_groups:
            group_rows.append(load_pll_top1_control(group, ranked_rows, group_rows))
        add_group_metrics(group_rows)
        final_rows.extend(group_rows)

    all_scores_path = args.ranked_csv.resolve().parent / "ProteinMPNN_all_scores.csv"
    if not all_scores_path.is_file():
        raise FileNotFoundError(f"WT source table does not exist: {all_scores_path}")
    with all_scores_path.open("r", encoding="utf-8-sig", newline="") as handle:
        wt_rows = [row for row in csv.DictReader(handle) if row["Group"] == "WT"]
    if len(wt_rows) != 1:
        raise ValueError(f"Expected exactly one WT row in {all_scores_path}")
    wt_sequence = wt_rows[0]["Mature_Sequence"].strip().upper()
    if len(wt_sequence) != 269:
        raise ValueError("WT mature sequence must be 269 aa")

    for final_order, row in enumerate(final_rows, start=1):
        row["Final_Order"] = final_order
    expected = len(EXPECTED_GROUPS) * (
        args.champions_per_group + args.balanced_per_group
    ) + len(args.pll_top1_groups)
    if len(final_rows) != expected:
        raise RuntimeError(f"Selected {len(final_rows)} candidates; expected {expected}")
    if len({row["Sequence_ID"] for row in final_rows}) != len(final_rows):
        raise RuntimeError("Final selection contains duplicate candidates")

    csv_path, fasta_path, summary_path = write_outputs(
        args.output_dir.resolve(),
        final_rows,
        float(wt["WT_MPNN_Score"]),
        wt_sequence,
    )
    print(
        "Selection rule: keep existing PLL/GA champions; add distinct candidates "
        "using unweighted PLL/GA/within-group-MPNN Pareto balance and diversity; "
        "retain requested PLL rank-1 controls."
    )
    print(f"WT ProteinMPNN score: {float(wt['WT_MPNN_Score']):.8f}")
    for group in EXPECTED_GROUPS:
        rows = [row for row in final_rows if row["Group"] == group]
        print(f"\n[{group}] {len(rows)} candidates")
        for row in rows:
            print(
                f"  {row['Selection_Class']:<23} {row['Export_ID']:<28} "
                f"PLL={row['PLL_Rank']:>3} GA={row['GA_Rank']:>3} "
                f"MPNN={row['MPNN_Rank_Within_Group']:>3} "
                f"delta={row['Delta_MPNN_vs_WT']:+.8f}"
            )
    print(f"\nCSV:     {csv_path}")
    print(f"FASTA:   {fasta_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
