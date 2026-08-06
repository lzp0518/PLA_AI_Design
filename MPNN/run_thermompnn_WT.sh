#!/bin/bash

source ~/envs/pla/bin/activate

THERMOMPNN_DIR=~/software/ThermoMPNN/ThermoMPNN-main
PDB=~/projects/PLA/MPNN/input/WIL_PLA.pdb
OUT=~/projects/PLA/MPNN/thermompnn_result

mkdir -p $OUT

cd $THERMOMPNN_DIR

python analysis/custom_inference.py \
--pdb $PDB \
--chain A \
--model_path models/thermoMPNN_default.pt \
--out_dir $OUT
