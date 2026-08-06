#!/bin/bash
set -e

echo "Starting pdb2gmx..."

echo 8 | gmx pdb2gmx \
-f input/PLA_top1.pdb \
-o PLA_processed.gro \
-p topol.top \
-water tip3p

echo "Done"