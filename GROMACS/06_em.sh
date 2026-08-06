#!/bin/bash
set -e

GMX=/home/rduser/software/gromacs/gromacs-2025.1/build/bin/gmx

$GMX grompp \
-f mdp/em.mdp \
-c PLA_ions.gro \
-p topol.top \
-o em.tpr

$GMX mdrun \
-deffnm em \
-nb gpu