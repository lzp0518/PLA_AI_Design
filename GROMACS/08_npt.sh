#!/bin/bash
set -e

GMX=/home/rduser/software/gromacs/gromacs-2025.1/build/bin/gmx

$GMX grompp \
-f mdp/npt.mdp \
-c nvt.gro \
-r nvt.gro \
-t nvt.cpt \
-p topol.top \
-o npt.tpr

$GMX mdrun \
-deffnm npt \
-nb gpu \
-pme gpu \
-update gpu
