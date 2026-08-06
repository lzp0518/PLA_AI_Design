#!/bin/bash
gmx editconf \
-f PLA_processed.gro \
-o PLA_box.gro \
-c \
-d 1.0 \
-bt dodecahedron
