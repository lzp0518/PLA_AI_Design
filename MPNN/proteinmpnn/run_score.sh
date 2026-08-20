#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROTEINMPNN_DIR="${PROTEINMPNN_DIR:-$HOME/software/ProteinMPNN}"
PLA_ENV_ACTIVATE="${PLA_ENV_ACTIVATE:-$HOME/envs/pla/bin/activate}"

PDB_PATH="$PROJECT_DIR/MPNN/proteinmpnn/input/PLA_WIL_mature.pdb"
FASTA_PATH="$PROJECT_DIR/MPNN/proteinmpnn/input/PLA_all_candidates_mature.fasta"
OUTPUT_DIR="$PROJECT_DIR/MPNN/proteinmpnn/output"

MODEL_NAME="${MODEL_NAME:-v_48_020}"
NUM_REPEATS="${NUM_REPEATS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-20260819}"

if [[ ! -f "$PLA_ENV_ACTIVATE" ]]; then
    echo "Missing PLA environment activation script: $PLA_ENV_ACTIVATE" >&2
    exit 1
fi
source "$PLA_ENV_ACTIVATE"

if [[ ! -f "$PROTEINMPNN_DIR/protein_mpnn_run.py" ]]; then
    echo "ProteinMPNN is not installed at: $PROTEINMPNN_DIR" >&2
    exit 1
fi
if [[ ! -f "$PROTEINMPNN_DIR/vanilla_model_weights/$MODEL_NAME.pt" ]]; then
    echo "Missing ProteinMPNN model: $MODEL_NAME" >&2
    exit 1
fi
if [[ ! -f "$PDB_PATH" || ! -f "$FASTA_PATH" ]]; then
    echo "Missing prepared ProteinMPNN input under: $SCRIPT_DIR/input" >&2
    exit 1
fi
if (( NUM_REPEATS < 1 || BATCH_SIZE < 1 || NUM_REPEATS % BATCH_SIZE != 0 )); then
    echo "NUM_REPEATS must be positive and divisible by BATCH_SIZE." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Project:       $PROJECT_DIR"
echo "ProteinMPNN:   $PROTEINMPNN_DIR"
echo "PDB:           $PDB_PATH"
echo "FASTA:         $FASTA_PATH"
echo "Output:        $OUTPUT_DIR"
echo "Model:         $MODEL_NAME"
echo "Repeats/batch: $NUM_REPEATS/$BATCH_SIZE"
echo "Seed:          $SEED"

python "$PROTEINMPNN_DIR/protein_mpnn_run.py" \
    --path_to_model_weights "$PROTEINMPNN_DIR/vanilla_model_weights" \
    --model_name "$MODEL_NAME" \
    --pdb_path "$PDB_PATH" \
    --pdb_path_chains "A" \
    --path_to_fasta "$FASTA_PATH" \
    --score_only 1 \
    --num_seq_per_target "$NUM_REPEATS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --out_folder "$OUTPUT_DIR"

echo "ProteinMPNN scoring finished: $OUTPUT_DIR"
