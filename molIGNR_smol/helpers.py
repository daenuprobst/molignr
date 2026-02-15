import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def load_csv(path, column="smiles", max_length=0, label_column=None):
    sep = ","
    if path.endswith("tsv"):
        sep = "\t"

    df = pd.read_csv(path, sep=sep)

    if max_length > 0:
        df = df.loc[df[column].str.len() <= max_length]

    if label_column is not None:
        return df[column].to_list(), df[label_column].to_list()

    return df[column].to_list()


def safe_filename(text):
    text = text.replace("\\", "")
    text = text.replace("/", "")
    return text


def plot_adj_matrices(
    matrix1,
    matrix2,
    save_path,
    title1="Matrix 1",
    title2="Matrix 2",
    cmap="turbo",
    figsize=(12, 5),
    dpi=300,
):
    # Convert torch tensors to numpy if needed
    if hasattr(matrix1, "detach"):
        matrix1 = matrix1.detach().cpu().numpy()
    if hasattr(matrix2, "detach"):
        matrix2 = matrix2.detach().cpu().numpy()

    # Crop matrix2 to match the size of matrix1 (from top-left)
    rows1, cols1 = matrix1.shape[:2]
    matrix2 = matrix2[:rows1, :cols1]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot first matrix
    im1 = axes[0].imshow(matrix1, cmap=cmap, aspect="auto")
    axes[0].set_title(title1)
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    plt.colorbar(im1, ax=axes[0])

    # Plot second matrix
    im2 = axes[1].imshow(matrix2, cmap=cmap, aspect="auto")
    axes[1].set_title(title2)
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


class LossNormalizer:
    def __init__(self, momentum=0.99):
        self.gw_scale = 1.0
        self.mse_scale = 1.0
        self.momentum = momentum

    def update_and_normalize(self, gw_loss, mse_loss):
        # Update running estimates
        self.gw_scale = (
            self.momentum * self.gw_scale + (1 - self.momentum) * gw_loss.item()
        )
        self.mse_scale = (
            self.momentum * self.mse_scale + (1 - self.momentum) * mse_loss.item()
        )

        # Normalize
        gw_norm = gw_loss / (self.gw_scale + 1e-8)
        mse_norm = mse_loss / (self.mse_scale + 1e-8)

        return gw_norm, mse_norm


VALENCE_DICT = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}


# def molecular_graph_to_smiles(
#     adj, atom_types, atom_type_map, bond_type_logits=None, path=None
# ):
#     # try:
#     mol = Chem.RWMol()
#     atom_indices = []
#     atom_symbols = []

#     # Add atoms
#     if atom_types is not None:
#         for atom_type_logits in atom_types:
#             atom_type_idx = np.argmax(atom_type_logits, axis=-1)
#             atom_symbol = atom_type_map[atom_type_idx]
#             atom = Chem.Atom(atom_symbol)
#             idx = mol.AddAtom(atom)
#             atom_indices.append(idx)
#             atom_symbols.append(atom_symbol)
#     else:
#         for i in range(adj.shape[0]):
#             atom = Chem.Atom("C")
#             idx = mol.AddAtom(atom)
#             atom_indices.append(idx)
#             atom_symbols.append("C")

#     bond_type_map = {
#         0: Chem.BondType.SINGLE,
#         1: Chem.BondType.DOUBLE,
#         2: Chem.BondType.TRIPLE,
#         3: Chem.BondType.AROMATIC,
#     }

#     # Track current valence
#     current_valence = [0] * adj.shape[0]

#     # Collect candidate bonds with adjacency above per-atom threshold
#     candidate_bonds = []

#     for i in range(adj.shape[0]):
#         max_val_i = VALENCE_DICT.get(atom_symbols[i], 4)

#         # Compute per-atom threshold (mean of neighbors)
#         neighbor_weights = [adj[i, j] for j in range(adj.shape[0]) if i != j]
#         if len(neighbor_weights) == 0:
#             continue

#         # mu = np.mean(neighbor_weights)
#         # sigma = np.std(neighbor_weights)
#         # atom_thresh = min(mu + 2 * sigma, 0.9)

#         # atom_thresh = np.mean(neighbor_weights)
#         atom_thresh = 0.1

#         # Find neighbors above threshold
#         neighbors = [
#             (j, adj[i, j])
#             for j in range(adj.shape[0])
#             if i != j and adj[i, j] >= atom_thresh
#         ]
#         # Sort neighbors by weight descending
#         neighbors.sort(key=lambda x: x[1], reverse=True)

#         # Keep at most `max_val_i` neighbors
#         neighbors = neighbors[:max_val_i]

#         # Add to candidate bonds (i < j to avoid duplicates)
#         for j, weight in neighbors:
#             if i < j:  # avoid duplicates
#                 candidate_bonds.append((i, j, weight))

#     # Sort all candidate bonds by weight descending (optional, for stronger first)
#     candidate_bonds.sort(key=lambda x: x[2], reverse=True)

#     # Add bonds while respecting valence
#     for i, j, _ in candidate_bonds:
#         # Skip if bond already exists
#         if mol.GetBondBetweenAtoms(i, j) is not None:
#             continue

#         # Determine bond order
#         bond_type = 0
#         bond_order = 1
#         if bond_type_logits is not None and bond_type_logits[i, j].any():
#             bond_type = np.argmax(bond_type_logits[i, j])
#         if bond_type in [0, 1, 2]:
#             bond_order = bond_type + 1

#         # Check valence constraints
#         max_val_i = VALENCE_DICT.get(atom_symbols[i], 4)
#         max_val_j = VALENCE_DICT.get(atom_symbols[j], 4)
#         remaining_val_i = max_val_i - current_valence[i]
#         remaining_val_j = max_val_j - current_valence[j]

#         if bond_order <= remaining_val_i and bond_order <= remaining_val_j:
#             mol.AddBond(atom_indices[i], atom_indices[j], bond_type_map[bond_type])
#             current_valence[i] += bond_order
#             current_valence[j] += bond_order

#     # Sanitize and convert to SMILES
#     mol = mol.GetMol()
#     Chem.SanitizeMol(mol)
#     smiles = Chem.MolToSmiles(mol)

#     if mol and path:
#         Draw.MolToFile(mol, path, size=(300, 300))

#     return smiles

#     # except Exception as e:
#     #     print(f"Failed to convert to SMILES: {e}")
#     #     return None

ALLOWED_VALENCE = {
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}


def atom_explicit_valence(atom):
    return sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())


def can_change_atom(atom, new_symbol):
    max_val = ALLOWED_VALENCE.get(new_symbol, 4)
    return atom_explicit_valence(atom) <= max_val


def can_upgrade_bond(bond, new_bond_type):
    order_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 1.5,
    }

    old_order = bond.GetBondTypeAsDouble()
    new_order = order_map[new_bond_type]
    delta = new_order - old_order

    if delta <= 0:
        return True

    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()

    max1 = ALLOWED_VALENCE.get(a1.GetSymbol(), 4)
    max2 = ALLOWED_VALENCE.get(a2.GetSymbol(), 4)

    return (
        atom_explicit_valence(a1) + delta <= max1
        and atom_explicit_valence(a2) + delta <= max2
    )


def molecular_graph_to_smiles(
    adj, atom_types, atom_type_map, bond_type_logits=None, path=None
):
    try:
        n = adj.shape[0]
        mol = Chem.RWMol()
        for _ in range(n):
            mol.AddAtom(Chem.Atom("C"))

        bond_type_map = {
            0: Chem.BondType.SINGLE,
            1: Chem.BondType.DOUBLE,
            2: Chem.BondType.TRIPLE,
            3: Chem.BondType.AROMATIC,
        }

        current_valence = [0] * n
        candidate_bonds = []

        for i in range(n):
            all_potential_weights = [adj[i, k] for k in range(n) if i != k]
            threshold = np.mean(all_potential_weights) if all_potential_weights else 0

            neighbors = [
                (j, adj[i, j]) for j in range(n) if i != j and adj[i, j] >= threshold
            ]
            neighbors.sort(key=lambda x: x[1], reverse=True)

            for j, w in neighbors[:4]:
                if i < j:
                    candidate_bonds.append((i, j, w))

        # Sort all global candidates by weight to prioritize the strongest bonds
        candidate_bonds.sort(key=lambda x: x[2], reverse=True)

        for i, j, _ in candidate_bonds:
            if mol.GetBondBetweenAtoms(i, j):
                continue

            # Look up the intended type immediately
            if bond_type_logits is not None:
                bond_type_idx = np.argmax(bond_type_logits[i, j])
                requested_type = bond_type_map.get(bond_type_idx, Chem.BondType.SINGLE)
            else:
                requested_type = Chem.BondType.SINGLE

            # Map the type to a numerical value for valence checking
            order_val = {
                Chem.BondType.SINGLE: 1,
                Chem.BondType.DOUBLE: 2,
                Chem.BondType.TRIPLE: 3,
                Chem.BondType.AROMATIC: 1.5,
            }[requested_type]

            # Only add if it doesn't violate the valence of either atom
            if (
                current_valence[i] + order_val <= 4
                and current_valence[j] + order_val <= 4
            ):
                mol.AddBond(i, j, requested_type)
                current_valence[i] += order_val
                current_valence[j] += order_val

        if atom_types is not None:
            for i, logits in enumerate(atom_types):
                atom = mol.GetAtomWithIdx(i)
                new_symbol = atom_type_map[np.argmax(logits)]
                # Update atomic number directly
                atom.SetAtomicNum(Chem.Atom(new_symbol).GetAtomicNum())

        # Finalize and Sanitize
        final_mol = mol.GetMol()
        Chem.SanitizeMol(final_mol)

        if mol and path:
            Draw.MolToFile(mol, path, size=(300, 300))

        return Chem.MolToSmiles(final_mol)

    except Exception as e:
        print(f"Error: {e}")
        return None
