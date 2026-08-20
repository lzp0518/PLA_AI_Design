#!/usr/bin/env python3
"""Build validated mature-PLA inputs for ProteinMPNN score-only inference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
INPUT_DIR = SCRIPT_DIR / "input"

LEADER_SEQUENCE = "MFLRREFGAVAALSVLAHAAPAPAPMQRR"
FULL_WT_SEQUENCE = (
    "MFLRREFGAVAALSVLAHAAPAPAPMQRRDISSTVLDNIDLFAQYSAAAYCSSNIESTGTTLTCDVGN"
    "CPLVEAAGATTIDEFDDTSSYGDPTGFIAVDPTNELIVLSFRGSSDLSNWIADLNFGLTSVSSICDGC"
    "EMHKGFYEAWEVIADTITSKVEAAVSSYPDYTLVFTGHSYGAALAAVAATVLRNAGYTLDLYNFGQPR"
    "IGNLALADYITGQNMGSNYRVTHTDDIVPKLPPELLGYHHFSPEYWITSGNDVTVTTSDVTEVVGVD"
    "STAGNDGTLLDSTTAHRWYTIYISECS"
)
MATURE_WT_SEQUENCE = FULL_WT_SEQUENCE[len(LEADER_SEQUENCE) :]
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

GROUPS = (
    (
        "E20",
        PROJECT_DIR / "ESM_GA" / "01_ESM_GA_20" / "Top300_ESM_PLL_ranked.csv",
        20,
    ),
    (
        "E30",
        PROJECT_DIR / "ESM_GA" / "02_ESM_GA_30" / "Top300_ESM_PLL_ranked.csv",
        30,
    ),
    (
        "M30",
        PROJECT_DIR / "ESM_GA" / "02_ESM_MPNN_GA" / "Top300_ESM_PLL_ranked.csv",
        30,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a 269-residue mature WT PDB and combine WT plus the three "
            "Top300 tables into one mature-sequence FASTA."
        )
    )
    parser.add_argument(
        "--pdb-source",
        type=Path,
        required=True,
        help="Source mature WT PDB. Residue numbering may be 30-298 or 1-269.",
    )
    return parser.parse_args()


def normalize_pdb(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Mature WT PDB does not exist: {source}")

    atom_lines: list[str] = []
    residue_order: list[tuple[str, int, str]] = []
    residue_names: dict[tuple[str, int, str], str] = {}
    residue_atoms: dict[tuple[str, int, str], set[str]] = {}

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            raise ValueError(f"Malformed ATOM record in {source}: {line!r}")
        chain = line[21]
        residue_number = int(line[22:26].strip())
        insertion_code = line[26]
        key = (chain, residue_number, insertion_code)
        residue_name = line[17:20].strip()
        atom_name = line[12:16].strip()
        alt_loc = line[16]

        if chain != "A":
            raise ValueError(f"Expected only chain A, found chain {chain!r}")
        if residue_name not in THREE_TO_ONE:
            raise ValueError(f"Unsupported residue {residue_name!r} at {key}")
        if alt_loc not in {" ", "A"}:
            continue
        if key not in residue_names:
            residue_order.append(key)
            residue_names[key] = residue_name
            residue_atoms[key] = set()
        elif residue_names[key] != residue_name:
            raise ValueError(f"Conflicting residue names at {key}")
        residue_atoms[key].add(atom_name)
        atom_lines.append(line)

    if len(residue_order) != len(MATURE_WT_SEQUENCE):
        raise ValueError(
            f"PDB has {len(residue_order)} residues; expected {len(MATURE_WT_SEQUENCE)}"
        )
    pdb_sequence = "".join(THREE_TO_ONE[residue_names[key]] for key in residue_order)
    if pdb_sequence != MATURE_WT_SEQUENCE:
        raise ValueError("Mature PDB sequence does not match the project WT mature sequence")

    missing_backbone = [
        f"{key}:{atom}"
        for key in residue_order
        for atom in ("N", "CA", "C", "O")
        if atom not in residue_atoms[key]
    ]
    if missing_backbone:
        raise ValueError(
            "PDB is missing required backbone atoms: " + ", ".join(missing_backbone[:10])
        )

    renumber = {key: index for index, key in enumerate(residue_order, start=1)}
    normalized_lines: list[str] = []
    for line in atom_lines:
        key = (line[21], int(line[22:26].strip()), line[26])
        normalized_lines.append(line[:22] + f"{renumber[key]:4d}" + " " + line[27:])
    normalized_lines.extend(("TER", "END"))

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")


def integer_rank(value: str, field_name: str, source: Path, row_number: int) -> int:
    try:
        rank = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field_name} in {source}, row {row_number}: {value!r}"
        ) from exc
    if rank < 1:
        raise ValueError(f"{field_name} must be positive in {source}, row {row_number}")
    return rank


def wrap_sequence(sequence: str, width: int = 80) -> list[str]:
    return [sequence[start : start + width] for start in range(0, len(sequence), width)]


def build_fasta(fasta_path: Path, manifest_path: Path) -> None:
    fasta_lines = [">PLA_WT", *wrap_sequence(MATURE_WT_SEQUENCE)]
    manifest_rows: list[dict[str, object]] = [
        {
            "Fasta_Index": 0,
            "Sequence_ID": "PLA_WT",
            "Group": "WT",
            "Source_File": "",
            "Source_Row": "",
            "PLL_Rank": "",
            "GA_Rank": "",
            "Mutation_Count": 0,
            "Sequence_Length": len(MATURE_WT_SEQUENCE),
        }
    ]
    seen_sequences = {MATURE_WT_SEQUENCE: "PLA_WT"}
    seen_ids = {"PLA_WT"}
    fasta_index = 1

    for group, source, expected_mutations in GROUPS:
        if not source.is_file():
            raise FileNotFoundError(f"Top300 table does not exist: {source}")
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"PLL_Rank", "GA_Rank", "Sequence"}
            missing = required_columns.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{source} is missing columns: {sorted(missing)}")
            rows = list(reader)
        if len(rows) != 300:
            raise ValueError(f"{source} has {len(rows)} rows; expected 300")

        for source_row, row in enumerate(rows, start=2):
            full_sequence = row["Sequence"].strip().upper()
            if len(full_sequence) != len(FULL_WT_SEQUENCE):
                raise ValueError(
                    f"{source}, row {source_row}: sequence length is "
                    f"{len(full_sequence)}, expected {len(FULL_WT_SEQUENCE)}"
                )
            if not full_sequence.startswith(LEADER_SEQUENCE):
                raise ValueError(f"{source}, row {source_row}: unexpected leader sequence")
            mature_sequence = full_sequence[len(LEADER_SEQUENCE) :]
            invalid = set(mature_sequence).difference(VALID_AMINO_ACIDS)
            if invalid:
                raise ValueError(
                    f"{source}, row {source_row}: invalid amino acids {sorted(invalid)}"
                )
            mutation_count = sum(
                candidate != wt
                for candidate, wt in zip(mature_sequence, MATURE_WT_SEQUENCE)
            )
            if mutation_count != expected_mutations:
                raise ValueError(
                    f"{source}, row {source_row}: found {mutation_count} mutations, "
                    f"expected {expected_mutations}"
                )

            pll_rank = integer_rank(row["PLL_Rank"], "PLL_Rank", source, source_row)
            ga_rank = integer_rank(row["GA_Rank"], "GA_Rank", source, source_row)
            sequence_id = f"PLA_{group}_PLL{pll_rank}_GA{ga_rank}"
            if sequence_id in seen_ids:
                raise ValueError(f"Duplicate FASTA identifier: {sequence_id}")
            if mature_sequence in seen_sequences:
                raise ValueError(
                    f"Duplicate mature sequence: {sequence_id} and "
                    f"{seen_sequences[mature_sequence]}"
                )
            seen_ids.add(sequence_id)
            seen_sequences[mature_sequence] = sequence_id

            fasta_lines.extend((f">{sequence_id}", *wrap_sequence(mature_sequence)))
            manifest_rows.append(
                {
                    "Fasta_Index": fasta_index,
                    "Sequence_ID": sequence_id,
                    "Group": group,
                    "Source_File": source.relative_to(PROJECT_DIR).as_posix(),
                    "Source_Row": source_row,
                    "PLL_Rank": pll_rank,
                    "GA_Rank": ga_rank,
                    "Mutation_Count": mutation_count,
                    "Sequence_Length": len(mature_sequence),
                }
            )
            fasta_index += 1

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    if fasta_index != 901:
        raise RuntimeError(f"Generated {fasta_index} FASTA records; expected 901")


def main() -> None:
    args = parse_args()
    pdb_output = INPUT_DIR / "PLA_WIL_mature.pdb"
    fasta_output = INPUT_DIR / "PLA_all_candidates_mature.fasta"
    manifest_output = INPUT_DIR / "PLA_all_candidates_manifest.csv"

    normalize_pdb(args.pdb_source.resolve(), pdb_output)
    build_fasta(fasta_output, manifest_output)

    print(f"PDB:      {pdb_output}")
    print(f"FASTA:    {fasta_output}")
    print(f"Manifest: {manifest_output}")
    print("Validated: 269-aa WT PDB + WT and 900 unique mature candidates")


if __name__ == "__main__":
    main()
