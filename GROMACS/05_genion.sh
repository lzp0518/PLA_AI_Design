#!/bin/bash
set -e

GMX=/home/rduser/software/gromacs/gromacs-2025.1/build/bin/gmx

echo SOL | $GMX genion \
-s ions.tpr \
-o PLA_ions.gro \
-p topol.top \
-neutral \
-conc 0.15
