#!/bin/bash

GMX=/home/rduser/software/gromacs/gromacs-2025.1/build/bin/gmx

mkdir -p results/PLA_top1

$GMX grompp \
-f md.mdp \
-c npt.gro \
-t npt.cpt \
-p topol.top \
-o results/PLA_top1/md.tpr

$GMX mdrun \
-s results/PLA_top1/md.tpr \
-deffnm results/PLA_top1/md \
-nb gpu \
-pme gpu \
-update gpu