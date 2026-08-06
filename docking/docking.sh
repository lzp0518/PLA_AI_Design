# 受体
mk_prepare_receptor.py -i PLA.pdb -o PLA.pdbqt

# 配体
mk_prepare_ligand.py -i DPPC.sdf -o DPPC.pdbqt

# 对接
vina --config config.txt --out docking_out.pdbqt
