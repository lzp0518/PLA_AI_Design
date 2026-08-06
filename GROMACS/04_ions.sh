#!/bin/bash
set -e

GMX=/home/rduser/software/gromacs/gromacs-2025.1/build/bin/gmx

$GMX grompp \
-f mdp/ions.mdp \
-c PLA_solv.gro \
-p topol.top \
-o ions.tpr
