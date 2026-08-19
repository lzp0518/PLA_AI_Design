# GROMACS batch simulation and analysis

`run_all.sh` processes protein PDB files sequentially. Each input gets an
independent directory under `results/`, so topology, coordinates and trajectory
files cannot overwrite files from another protein.

The existing `01_pdb2gmx.sh` through `09_md.sh` scripts are retained for
reference. The batch script performs the equivalent workflow itself and then
runs the standard post-MD analyses.

## Scope

The automated topology setup supports ordinary proteins made from residues
known to the selected force field. A PDB containing a ligand, cofactor,
non-standard amino acid, membrane or covalently attached small molecule needs
additional topology parameters and should not be run blindly through this
workflow.

Defaults match the current PLA setup:

- CHARMM27 force field and TIP3P water;
- dodecahedral box with 1.0 nm padding;
- neutralization plus 0.15 mol/L NaCl;
- energy minimization, 100 ps NVT, 500 ps NPT and 100 ns production MD;
- post-MD equilibrium analysis over 50-100 ns.

## Server usage

```bash
cd ~/projects/PLA/GROMACS
chmod +x run_all.sh

# Process every input/*.pdb sequentially
nohup ./run_all.sh > run_all.out 2>&1 &

# Process one or more named inputs
./run_all.sh PLA_top1.pdb another_model.pdb

# Analyze an already completed results/PLA_top1 simulation without rerunning MD
ANALYSIS_ONLY=1 ./run_all.sh PLA_top1.pdb
```

The last command is compatible with the existing server directory that already
contains `results/PLA_top1/md.tpr`, `md.xtc`, `md.edr` and `md.gro`.

Follow progress with:

```bash
tail -f run_all.out
tail -f results/PLA_top1/pipeline.log
```

Do not launch several PDBs as separate background jobs on the same GPU. The
script intentionally runs them one after another.

## Configuration

Settings can be changed with environment variables:

```bash
GMX=/path/to/gmx \
ANALYSIS_BEGIN_PS=50000 \
FORCE_FIELD=charmm27 \
WATER_MODEL=tip3p \
SALT_CONC=0.15 \
./run_all.sh
```

Useful controls:

- `ANALYSIS_ONLY=1`: do not run simulation; analyze existing results.
- `SIMULATION_ONLY=1`: run simulation but skip post-processing.
- `FORCE_ANALYSIS=1`: regenerate analysis outputs. GROMACS may create backup
  files for outputs that already exist.
- `CORE_RMSD=1`, `CORE_RESID_START=29`, `CORE_RESID_END=298`: control the
  PLA core-backbone RMSD. Set `CORE_RMSD=0` for unrelated proteins.
- `EM_MDRUN_ARGS` and `MD_MDRUN_ARGS`: replace the default GPU arguments.
- `SYSTEM_GROUP`, `PROTEIN_GROUP`, `CALPHA_GROUP`, `BACKBONE_GROUP`: override
  default index group numbers for non-standard topologies.

For example, CPU-only execution can be requested with explicit CPU arguments:

```bash
EM_MDRUN_ARGS='-ntmpi 1' MD_MDRUN_ARGS='-ntmpi 1' ./run_all.sh
```

## Resume behavior

Completed stage outputs are reused. If production MD has a checkpoint but no
normal completion record, the script continues with `-cpi md.cpt -append`.
It does not delete existing trajectories or topology files.

If `md.log` contains `Finished mdrun` and the expected production outputs are
present, simulation is skipped and only missing analyses are generated.

## Analysis outputs

Each result directory contains an `analysis/` folder with:

- `trajectory_check.txt`: original trajectory integrity and final time;
- `protein.tpr`: protein-only topology matching the corrected analysis trajectory;
- `md_protein_noPBC.xtc`: centered, molecule-whole, protein-only trajectory;
- `rmsd_backbone.xvg`: full-trajectory backbone RMSD;
- `rmsd_core_backbone.xvg`: residues 29-298 backbone RMSD for the PLA models;
- `rmsf_calpha_equilibrium.xvg`: equilibrium-window C-alpha RMSF;
- `gyrate_equilibrium.xvg`: radius of gyration;
- `sasa_equilibrium.xvg` and `sasa_per_residue_equilibrium.xvg`: SASA;
- `hbond_protein_equilibrium.xvg`: intraprotein hydrogen-bond count;
- `dssp_equilibrium.dat` and `dssp_count_equilibrium.xvg`: secondary structure;
- `min_periodic_distance_equilibrium.xvg`: protein distance to its periodic image;
- equilibrium temperature, pressure, density and potential-energy XVG files;
- energy-minimization, NVT and NPT quality-control XVG files when those stage
  files are available in the target directory;
- `run_qc.txt`: convergence/completion and serious-warning excerpts;
- `summary.tsv`: count, mean, standard deviation, minimum and maximum of the
  main numerical metrics;
- `rmsf_top20.tsv`: the twenty highest-fluctuation residues.

The protein-only corrected trajectory is used to avoid creating another full
solvated trajectory of roughly the same size as `md.xtc`.

After all targets finish, `results/summary_all.tsv` combines the numerical
summaries from every PDB for direct comparison.

These are baseline stability analyses. Ligand binding energies, mutation
comparisons, PCA, clustering and free-energy calculations require a specific
scientific design and are not treated as universally valid automatic outputs.
