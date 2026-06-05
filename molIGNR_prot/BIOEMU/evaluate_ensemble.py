"""
Evaluate a BioEmu conformational ensemble against a reference structure.
Computes MSE (after alignment with the first frame), lDDT, and TM-score.

Accepts either:
  - a BioEmu output directory containing topology.pdb + samples.xtc  (auto-detected)
  - a directory of individual .pdb files

Install dependencies:
    pip install biopython tmtools numpy mdanalysis

Usage:
    python evaluate_ensemble.py --reference heavy_chain.pdb --ensemble_dir ./ensemble
    python evaluate_ensemble.py --reference heavy_chain.pdb --ensemble_dir ./ensemble --output results.csv
"""

import argparse
import csv
import math
import os
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from Bio.PDB import PPBuilder

try:
    from Bio.PDB import PDBParser
    from Bio.SVDSuperimposer import SVDSuperimposer
except ImportError:
    raise ImportError("Run: pip install biopython")

try:
    import tmtools
except ImportError:
    raise ImportError("Run: pip install tmtools")

import torch

from MDAnalysis.analysis.dihedrals import Ramachandran
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


import glob

# ── structure parsing ──────────────────────────────────────────────────────────


def xtc_to_pdbs(topology_pdb: str, xtc_file: str, out_dir: str) -> list[Path]:
    """Convert topology.pdb + samples.xtc into individual PDB files."""
    try:
        import MDAnalysis as mda
    except ImportError:
        raise ImportError("Run: pip install mdanalysis")
    
    existing = sorted(Path(out_dir).glob("conformation_*.pdb"))
    if existing:
        print(f"Found {len(existing)} existing PDBs, skipping extraction.")
        return existing

    os.makedirs(out_dir, exist_ok=True)
    u = mda.Universe(topology_pdb, xtc_file)
    paths = []
    for i, _ in enumerate(u.trajectory):
        path = os.path.join(out_dir, f"conformation_{i:04d}.pdb")
        u.select_atoms("all").write(path)
        paths.append(Path(path))
    print(f"Extracted {len(paths)} conformations from XTC → {out_dir}")
    return sorted(paths)


def get_pdb_files(ensemble_dir: str) -> list[Path]:
    """
    Return PDB files to evaluate.
    Auto-detects BioEmu XTC output (topology.pdb + samples.xtc) and converts
    to individual PDBs if needed.
    """
    ensemble_path = Path(ensemble_dir)
    topology = ensemble_path / "topology.pdb"
    xtc      = ensemble_path / "samples.xtc"

    if xtc.exists() and topology.exists():
        print("Detected BioEmu XTC output — converting to individual PDBs...")
        pdbs_dir = str(ensemble_path / "pdbs")
        return xtc_to_pdbs(str(topology), str(xtc), pdbs_dir)

    pdb_files = sorted(ensemble_path.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(
            f"No .pdb files or samples.xtc found in {ensemble_dir}"
        )
    return pdb_files

################### METRICS ##########################

def compute_mse(coords1, coords2_aligned):
    return torch.mean(torch.sum((coords1 - coords2_aligned) ** 2, dim=-1))

def compute_lddt(
    ref_coords: np.ndarray,
    sample_coords: np.ndarray,
    inclusion_radius: float = 15.0,
    thresholds: tuple = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """lDDT — superposition-free local distance preservation. Range [0, 1]."""
    ref_dists    = np.sqrt(((ref_coords[:, None] - ref_coords[None, :]) ** 2).sum(-1))
    sample_dists = np.sqrt(((sample_coords[:, None] - sample_coords[None, :]) ** 2).sum(-1))
    diff         = np.abs(ref_dists - sample_dists)
    mask         = (ref_dists < inclusion_radius) & (ref_dists > 0)
    n_pairs      = mask.sum()
    if n_pairs == 0:
        return 0.0
    per_threshold = [(diff[mask] < t).sum() / n_pairs for t in thresholds]
    return float(np.mean(per_threshold))

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "MSE":"M","HSD":"H","HSE":"H","HSP":"H",
}

def compute_tm_score(
    ref_coords: np.ndarray,
    ref_seq: list[str],
    sample_coords: np.ndarray,
    sample_seq: list[str],
) -> float:
    """TM-score normalised by reference length. Range (0, 1], >0.5 = same fold."""
    ref_str    = "".join(THREE_TO_ONE.get(r, "X") for r in ref_seq)
    sample_str = "".join(THREE_TO_ONE.get(r, "X") for r in sample_seq)
    result = tmtools.tm_align(sample_coords, ref_coords, sample_str, ref_str)
    return float(result.tm_norm_chain2)

################ RAMACHANDRAN #######################

def ramachandran_bioemu(
    bioemu_dir: str,
    save_path: str = "ramachandran_bioemu.png"
):
    phi_all = []
    psi_all = []

    pdb_files = list(Path(bioemu_dir).glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")

    for pdb_path in pdb_files:
        try:
            u       = mda.Universe(str(pdb_path))
            protein = u.select_atoms("protein")
            rama    = Ramachandran(protein).run()
            phi_all.append(rama.results.angles[:, :, 0].flatten())
            psi_all.append(rama.results.angles[:, :, 1].flatten())
        except Exception as e:
            print(f"Skipping {pdb_path.name}: {e}")
            continue

    phi_all = np.concatenate(phi_all)
    psi_all = np.concatenate(psi_all)

    print(f"Total angle pairs: {len(phi_all)}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist2d(phi_all, psi_all, bins=180, cmap="Blues", density=True)
    ax.set_xlabel("φ (degrees)")
    ax.set_ylabel("ψ (degrees)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title("Ramachandran — BioEmu ensemble")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    return phi_all, psi_all

############################ EVALUATION ################################ 

def evaluate_backbone(reference_pdb: str, pdb_files, output_csv: str, num_samples = None) -> None:
    """ Evaluation is done on all backbone atoms (N, CA, C). """

    u = mda.Universe(reference_pdb)
    backbone = u.select_atoms("protein and backbone")
    ca = u.select_atoms("protein and name CA")

    ref_ca_coords = ca.positions
    ref_coords = backbone.positions
    ref_seq = [r.resname for r in backbone.residues]

    print(f"ref_coords.shape = {ref_coords.shape}")
    print(f"ref_ca_coords.shape = {ref_ca_coords.shape}")
    print(f"\nReference : {reference_pdb}  ({len(ref_coords)} residues)")
    print(f"Ensemble  : {len(pdb_files)} conformations")
    print()
    print(f"{'File':<35} {'MSE (Å²)':>10} {'lDDT':>8} {'TM-score':>10}")
    print("─" * 67)

    rows = []
    cntr = 0
    for pdb in pdb_files:
        u_ = mda.Universe(str(pdb))
        backbone_ = u_.select_atoms("protein and backbone")
        ca_ = u_.select_atoms("protein and name CA")
        
        sample_coords = backbone_.positions
        sample_ca_coords = ca_.positions

        sample_seq = [r.resname for r in backbone_.residues]

        # Alignment with the first frame 
        sup = SVDSuperimposer()
        sup.set(ref_coords, sample_coords)
        sup.run()
        sample_coords = sup.get_transformed() 

        mse  = compute_mse(torch.tensor(ref_coords), torch.tensor(sample_coords))
        lddt = compute_lddt(ref_coords, sample_coords) 
        tm_score = compute_tm_score(ref_ca_coords, ref_seq, sample_ca_coords, sample_seq=sample_seq)

        rows.append({"file": pdb.name, "mse": mse, "lddt": lddt, "tm_score": tm_score})

        cntr += 1
        # Controls the number of samples evaluated, useful for large ensembles or testing. If num_samples is set, the loop will break after processing that many samples.
        if num_samples is not None and cntr >= num_samples:
            print(f"\nReached sample limit: {num_samples}. Stopping evaluation.")
            print(f"len(rows) = {len(rows)}")
            break

    mses  = [r["mse"]      for r in rows]
    lddts = [r["lddt"]     for r in rows]
    tms   = [r["tm_score"] for r in rows]

    print("─" * 67)
    print(f"{'Mean':<35} {np.mean(mses):10.3f} {np.mean(lddts):8.4f} {np.mean(tms):10.4f}")
    print(f"{'Std':<35} {np.std(mses):10.3f} {np.std(lddts):8.4f} {np.std(tms):10.4f}")
    print(f"{'Min':<35} {np.min(mses):10.3f} {np.min(lddts):8.4f} {np.min(tms):10.4f}")
    print(f"{'Max':<35} {np.max(mses):10.3f} {np.max(lddts):8.4f} {np.max(tms):10.4f}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "mse", "lddt", "tm_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute MSE, lDDT, and TM-score for a BioEmu ensemble vs a reference PDB."
    )

    parser.add_argument("--reference", default = "data/heavy_chain.pdb",  help="Reference PDB file")
    parser.add_argument("--ensemble_dir", default = "./ensemble_d2r", help="BioEmu output dir or dir of PDB files")
    parser.add_argument("--output",       default="ensemble_metrics.csv", help="Output CSV path")
    parser.add_argument("--num_samples",  type=int, help="Maximum number of samples to evaluate")
    args = parser.parse_args()


    # Turns the generated XTC file into individual PDBs for evaluation. If the PDBs already exist, this will be skipped.
    _ = xtc_to_pdbs(
       topology_pdb = f"{args.ensemble_dir}/topology.pdb",
        xtc_file     = f"{args.ensemble_dir}/samples.xtc",
        out_dir      = f"{args.ensemble_dir}/pdbs")
    
    # Collect the pdb file names to evaluate. This will either be the newly generated PDBs from the XTC or existing PDBs in the ensemble_dir.
    ensemble_path = Path(args.ensemble_dir) / "pdbs"
    pdb_files = sorted(ensemble_path.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files in {args.ensemble_dir}")
    
    # Evaluate the ensemble with different sample sizes to see how metrics converge. If num_samples is not set, it will evaluate all samples.
    num_samples_list = [500, 1000, 2000]
    for num_samples in num_samples_list:
        args.num_samples = num_samples
        args.output = f"ensemble_metrics_{num_samples}_{args.reference.split('/')[-1].split('.')[0]}.csv"
        print(f"\nEvaluating with num_samples = {args.num_samples}...\n")
        evaluate_backbone(args.reference, pdb_files, args.output, num_samples=args.num_samples)
        print()

    # Uncomment to generate Ramachandran plot for the ensemble. This can be time-consuming for large ensembles, so it's placed after the main evaluation loop. 
    # Considers all the PDB files in the ensemble directory.

    # ramachandran_bioemu(
    #    bioemu_dir = ensemble_path,
    #    save_path  = f"ramachandran_bioemu.png")
    # print(f"ramachandran_bioemu() completed. Plot saved to ramachandran_bioemu_{num_samples}.png")
    
    print("Evaluation completed for all specified sample sizes.")


if __name__ == "__main__":
    main()