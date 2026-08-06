#!/bin/bash
set -e

GMX=/home/rduser/software/gromacs/gromacs-2025.1/build/bin/gmx

$GMX grompp \
-f mdp/nvt.mdp \
-c em.gro \
-r em.gro \
-p topol.top \
-o nvt.tpr

$GMX mdrun \
-deffnm nvt \
-nb gpu \
-pme gpu \
-update gpu
