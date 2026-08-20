#!/usr/bin/env python3
"""Select balanced, diverse GA candidates and export mature-sequence FASTA files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

LEADER_SEQUENCE = "MFLRREFGAVAALSVLAHAAPAPAPMQRR"
FULL_SEQUENCE_LENGTH = 298
MATURE_SEQUENCE_LENGTH = FULL_SEQUENCE_LENGTH - len(LEADER_SEQUENCE)

DEFAULT_TOP_N = 3
DEFAULT_MAX_PLL_RANK = 100
DEFAULT_MAX_GA_RANK = 100

GROUPS = {
    "E20": {
        "input": SCRIPT_DIR / "01_ESM_GA_20" / "Top300_ESM_PLL_ranked.csv",
        "output": SCRIPT_DIR / "01_ESM_GA_20" / "result",
        "expected_mutations": 20,
        "min_distance": 4,
    },
    "E30": {
        "input": SCRIPT_DIR / "02_ESM_GA_30" / "Top300_ESM_PLL_ranked.csv",
        "output": SCRIPT_DIR / "02_ESM_GA_30" / "result",
        "expected_mutations": 30,
        "min_distance": 6,
    },
    "M30": {
        "input": SCRIPT_DIR / "02_ESM_MPNN_GA" / "Top300_ESM_PLL_ranked.csv",
        "output": SCRIPT_DIR / "02_ESM_MPNN_GA" / "result",
        "expected_mutations": 30,
        "min_distance": 4,
    },
}

MUTATION_PATTERN = re.compile(r"^(\d+)([A-Z])>([A-Z])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select candidates that rank well in both GA and PLL, apply a "
            "mutation-distance filter, and export mature-sequence FASTA files."
        )
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Candidates to export per group (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--max-pll-rank",
        type=int,
        default=DEFAULT_MAX_PLL_RANK,
        help=f"Maximum accepted PLL rank (default: {DEFAULT_MAX_PLL_RANK}).",
    )
    parser.add_argument(
        "--max-ga-rank",
        type=int,
        default=DEFAULT_MAX_GA_RANK,
        help=f"Maximum accepted GA rank (default: {DEFAULT_MAX_GA_RANK}).",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=None,
        help="Override the group-specific minimum mutation distance.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove previously generated FASTA files for each group first.",
    )
    args = parser.parse_args()

    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if args.max_pll_rank < 1 or args.max_ga_rank < 1:
        parser.error("rank cutoffs must be at least 1")
    if args.min_distance is not None and args.min_distance < 0:
        parser.error("--min-distance cannot be negative")

    return args


def index_to_label(index: int) -> str:
    """Return A, B, ..., Z, AA, AB, ... for a zero-based index."""
    label = ""
    while True:
        index, remainder = divmod(index, 26)
        label = chr(ord("A") + remainder) + label
        if index == 0:
            return label
        index -= 1


def parse_mutations(mutation_string: str) -> dict[int, tuple[str, str]]:
    mutations: dict[int, tuple[str, str]] = {}
    for token in mutation_string.split(";"):
        match = MUTATION_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid mutation token: {token!r}")

        position = int(match.group(1))
        if position in mutations:
            raise ValueError(f"Duplicate mutation position: {position}")
        mutations[position] = (match.group(2), match.group(3))

    return mutations


def mutation_distance(
    first: dict[int, tuple[str, str]],
    second: dict[int, tuple[str, str]],
) -> int:
    """Count positions whose selected mutant residue differs between variants."""
    all_positions = set(first) | set(second)
    return sum(
        first.get(position, (None, None))[1]
        != second.get(position, (None, None))[1]
        for position in all_positions
    )


def read_candidates(
    input_path: Path,
    expected_mutations: int,
    max_pll_rank: int,
    max_ga_rank: int,
) -> list[dict[str, object]]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"PLL_Rank", "GA_Rank", "Mutations", "Sequence"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{input_path} is missing columns: {', '.join(sorted(missing))}"
            )

        candidates: list[dict[str, object]] = []
        seen_rank_pairs: set[tuple[int, int]] = set()

        for line_number, raw_row in enumerate(reader, start=2):
            pll_rank = int(raw_row["PLL_Rank"])
            ga_rank = int(raw_row["GA_Rank"])
            rank_pair = (pll_rank, ga_rank)
            if rank_pair in seen_rank_pairs:
                raise ValueError(
                    f"Duplicate PLL/GA rank pair {rank_pair} in {input_path}"
                )
            seen_rank_pairs.add(rank_pair)

            sequence = raw_row["Sequence"].strip()
            if len(sequence) != FULL_SEQUENCE_LENGTH:
                raise ValueError(
                    f"{input_path}:{line_number} sequence length is "
                    f"{len(sequence)}, expected {FULL_SEQUENCE_LENGTH}"
                )
            if not sequence.startswith(LEADER_SEQUENCE):
                raise ValueError(
                    f"{input_path}:{line_number} has an unexpected leader sequence"
                )

            mutations = parse_mutations(raw_row["Mutations"].strip())
            if len(mutations) != expected_mutations:
                raise ValueError(
                    f"{input_path}:{line_number} has {len(mutations)} mutations, "
                    f"expected {expected_mutations}"
                )
            if min(mutations) <= len(LEADER_SEQUENCE):
                raise ValueError(
                    f"{input_path}:{line_number} contains a leader-region mutation"
                )

            if pll_rank <= max_pll_rank and ga_rank <= max_ga_rank:
                candidates.append(
                    {
                        "pll_rank": pll_rank,
                        "ga_rank": ga_rank,
                        "worst_rank": max(pll_rank, ga_rank),
                        "rank_sum": pll_rank + ga_rank,
                        "mutations": mutations,
                        "mutation_string": raw_row["Mutations"].strip(),
                        "sequence": sequence,
                    }
                )

    candidates.sort(
        key=lambda row: (
            row["worst_rank"],
            row["rank_sum"],
            row["pll_rank"],
            row["ga_rank"],
        )
    )
    return candidates


def select_candidates(
    candidates: list[dict[str, object]],
    top_n: int,
    min_distance: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []

    for candidate in candidates:
        if all(
            mutation_distance(candidate["mutations"], existing["mutations"])
            >= min_distance
            for existing in selected
        ):
            selected.append(candidate)
            if len(selected) == top_n:
                return selected

    raise RuntimeError(
        f"Only {len(selected)} candidates satisfy top_n={top_n} and "
        f"min_distance={min_distance}. Increase rank cutoffs or lower distance."
    )


def write_fasta(output_path: Path, header: str, full_sequence: str) -> None:
    mature_sequence = full_sequence[len(LEADER_SEQUENCE) :]
    if len(mature_sequence) != MATURE_SEQUENCE_LENGTH:
        raise ValueError(
            f"Mature sequence length is {len(mature_sequence)}, "
            f"expected {MATURE_SEQUENCE_LENGTH}"
        )

    lines = [
        mature_sequence[index : index + 80]
        for index in range(0, len(mature_sequence), 80)
    ]
    output_path.write_text(
        f">{header}\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def clean_group_output(output_dir: Path, group_name: str) -> None:
    for path in output_dir.glob(f"PLA_{group_name}_*_PLL*_GA*.fasta"):
        path.unlink()


def verify_fasta(path: Path, expected_header: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != f">{expected_header}":
        raise ValueError(f"Unexpected FASTA header in {path}")
    sequence = "".join(lines[1:])
    if len(sequence) != MATURE_SEQUENCE_LENGTH:
        raise ValueError(f"Unexpected FASTA sequence length in {path}")


def main() -> None:
    args = parse_args()
    print(
        "Selection rule: minimize worst(PLL_Rank, GA_Rank), then rank sum; "
        "apply mutation-distance filter."
    )

    total_written = 0
    for group_name, config in GROUPS.items():
        output_dir = config["output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.clean_output:
            clean_group_output(output_dir, group_name)

        min_distance = (
            args.min_distance
            if args.min_distance is not None
            else config["min_distance"]
        )
        candidates = read_candidates(
            config["input"],
            config["expected_mutations"],
            args.max_pll_rank,
            args.max_ga_rank,
        )
        selected = select_candidates(candidates, args.top_n, min_distance)

        print(f"\n[{group_name}] min_distance={min_distance}")
        for selection_index, candidate in enumerate(selected):
            label = index_to_label(selection_index)
            name = (
                f"PLA_{group_name}_{label}"
                f"_PLL{candidate['pll_rank']}"
                f"_GA{candidate['ga_rank']}"
            )
            output_path = output_dir / f"{name}.fasta"
            write_fasta(output_path, name, candidate["sequence"])
            verify_fasta(output_path, name)
            total_written += 1
            print(
                f"  {name}: worst_rank={candidate['worst_rank']}, "
                f"rank_sum={candidate['rank_sum']}"
            )

    expected_total = len(GROUPS) * args.top_n
    if total_written != expected_total:
        raise RuntimeError(
            f"Wrote {total_written} FASTA files, expected {expected_total}"
        )
    print(f"\nFinished: wrote {total_written} mature-sequence FASTA files.")


if __name__ == "__main__":
    main()
