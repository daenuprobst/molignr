"""
Full pipeline: extract sequence from PDB → run BioEmu → evaluate ensemble.

Usage:
    python run_bioemu.py --pdb heavy_chain.pdb --num_samples 100 --output_dir ./results
"""

import argparse
from pathlib import Path

import MDAnalysis as mda

from bioemu.sample import main as bioemu_sample


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H",
}

def extract_sequence(pdb_path: str) -> dict[str, str]:
    chains: dict[str, dict[int, str]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain  = line[21]
            resnum = int(line[22:26].strip())
            aa     = THREE_TO_ONE.get(line[17:20].strip(), "X")
            chains.setdefault(chain, {})[resnum] = aa
    return {
        c: "".join(res[r] for r in sorted(res))
        for c, res in chains.items()
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run(pdb: str, chain: str, num_samples: int, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. extract sequence
    print(f"Extracting sequence from {pdb} ...")
    sequences = extract_sequence(pdb)

    if chain not in sequences:
        available = list(sequences.keys())
        raise ValueError(f"Chain '{chain}' not found. Available: {available}")

    seq = sequences[chain]
    print(f"  Chain {chain}: {len(seq)} residues")
    print(f"  {seq[:60]}{'...' if len(seq) > 60 else ''}\n")

    # 2. run BioEmu
    print(f"Running BioEmu ({num_samples} samples) → {output_dir}")
    bioemu_sample(
        sequence=seq,
        num_samples=num_samples,
        output_dir=str(output_dir),
    )
    print("Done.")



def run_gpcrs(psf_path: str, xtc_path: str, num_samples: int, output_dir: str) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    u = mda.Universe(psf_path, xtc_path)
    protein = u.select_atoms("protein")

    seq = ""
    for r in protein.residues:
        seq += THREE_TO_ONE.get(r.resname, "X")
    print(f"sequence : {seq}")


    # 2. run BioEmu
    print(f"Running BioEmu ({num_samples} samples) → {output_dir}")
    bioemu_sample(
        sequence=seq,
        num_samples=num_samples,
        output_dir=str(output_dir),
    )
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run BioEmu on a PDB file.")
    parser.add_argument("--mode",    choices=["d2r", "gpcr"], default="pdb")
    parser.add_argument("--psf",     help="PSF file (GPCR mode)")
    parser.add_argument("--xtc",     help="XTC trajectory (GPCR mode)")
    parser.add_argument("--chain",       default="A",            help="Chain ID to use (default: A)")
    parser.add_argument("--num_samples", default=100, type=int,  help="Number of conformations to sample")
    parser.add_argument("--output_dir",  default="./_ensemble",   help="Output directory for sampled PDBs")
    args = parser.parse_args()


    if args.mode == "gpcr":
        run_gpcrs(args.psf, args.xtc, args.num_samples, args.output_dir)
    else:
        run(args.pdb, args.chain, args.num_samples, args.output_dir)



if __name__ == "__main__":
    main()

