# BioEmu — Conformational Ensemble Generation & Evaluation

Generate new protein conformations from a reference structure using [BioEmu](https://github.com/microsoft/bioemu) and evaluate them against the first frame in the MD simulation data.

## Installation

```bash
pip install biopython tmtools numpy mdanalysis matplotlib torch
pip install bioemu  # follow BioEmu's installation instructions
```

## Data

| Protein | Class | PDB | Source |
|---------|-------|-----|--------|
| Dopamine D2 Receptor (D2R) | A | 6CM4 | [Zenodo 10.5281/zenodo.15479781](https://zenodo.org/records/15479781) |
| Rhodopsin | A | 6AKY | [GPCRmd](https://www.gpcrmd.org/) |
| Secretin | B | 4K5Y | [GPCRmd](https://www.gpcrmd.org/) |

## Usage

### 1. Generate conformations — `run_bioemu.py`

Replace `--pdb`, `--psf`, and `--xtc` with the actual file paths from your data directory.

```bash
# D2R (from Zenodo) — protein_initial.pdb is in run1/ or run6/
python run_bioemu.py --mode d2r --pdb /path/protein.pdb --chain A --num_samples 1000 --output_dir ./ensemble_d2r

# Rhodopsin / Secretin (GPCRmd) — replace with your downloaded PSF and XTC paths
python run_bioemu.py --mode gpcr --psf /path/to/gpcr.psf --xtc /path/to/gpcr.xtc --num_samples 1000 --output_dir ./ensemble_gpcr
```

Output: `./ensemble/topology.pdb` + `./ensemble/samples.xtc`

### 2. Evaluate ensemble — `evaluate_ensemble.py`

```bash
# PDB reference (D2R)
python evaluate_ensemble.py --reference protein_initial.pdb --ensemble_dir ./ensemble_d2r
 
# PSF reference (GPCRmd — no PDB available)
# The script will automatically extract the first frame as a PDB
python evaluate_ensemble.py --reference gpcr.psf --reference_xtc gpcr.xtc --ensemble_dir ./ensemble_gpcr
```
 
Replace `--reference`, `--reference_xtc`, and `--ensemble_dir` with your actual file paths.
 
Aligns each generated conformation to the reference and computes:
 
| Metric | Range | Better |
|--------|-------|--------|
| MSE (Å²) | ≥ 0 | lower |
| lDDT | [0, 1] | higher |
| TM-score | (0, 1] — > 0.5 = same fold | higher |
 
Metrics are evaluated at sample sizes [500, 1000, 2000] to check convergence. One CSV is saved per run. Uncomment `ramachandran_bioemu()` in `main()` for a φ/ψ density plot over the full ensemble.
