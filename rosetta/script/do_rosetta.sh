#!/bin/bash

BIN=/home/rduser/software/rosetta/rosetta.binary.ubuntu.release-408/main/source/bin
DB=/home/rduser/software/rosetta/rosetta.binary.ubuntu.release-408/main/database

cd ~/projects/PLA/rosetta/relax

$BIN/relax.static.linuxgccrelease \
-database $DB \
-s input/PLA_top1.pdb \
-relax:cartesian \
-score:weights ref2015_cart \
-use_input_sc \
-fa_max_dis 9.0 \
-nstruct 5 \
-out:path:all output