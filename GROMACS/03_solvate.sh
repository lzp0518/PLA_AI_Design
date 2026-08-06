#!/bin/bash
set -e

GMX=/home/rduser/software/gromacs/gromacs-2025.1/build/bin/gmx

$GMX solvate \
-cp PLA_box.gro \
-cs spc216.gro \
-o PLA_solv.gro \
-p topol.top

