import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import to_dense_adj
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from helpers import plot_adj_matrices


def molecular_collate_fn(batch):
    pyg_batch = Batch.from_data_list(batch)
    num_nodes_per_graph = torch.bincount(pyg_batch.batch)
    max_nodes = num_nodes_per_graph.max().item()
    targets = get_molecular_graph_targets(pyg_batch, max_nodes)
    indices = torch.tensor([item["index"] for item in batch])

    return pyg_batch, targets, indices


def get_molecular_graph_targets(pyg_batch, max_nodes):
    batch_size = pyg_batch.batch.max().item() + 1

    # Convert to dense adjacency (automatically handles batching)
    adj = to_dense_adj(
        pyg_batch.edge_index, pyg_batch.batch, max_num_nodes=max_nodes
    )  # [batch_size, max_nodes, max_nodes]

    # Convert edge attributes to dense bond type matrix
    if (
        hasattr(pyg_batch, "edge_attr_bond_type")
        and pyg_batch.edge_attr_bond_type is not None
    ):
        # Create dense bond type matrix
        bond_types = to_dense_adj(
            pyg_batch.edge_index,
            pyg_batch.batch,
            edge_attr=pyg_batch.edge_attr_bond_type.float(),
            max_num_nodes=max_nodes,
        ).long()

    elif hasattr(pyg_batch, "edge_attr") and pyg_batch.edge_attr is not None:
        # Handle different edge_attr formats
        if pyg_batch.edge_attr.dim() > 1:
            edge_attr_scalar = pyg_batch.edge_attr.argmax(dim=1).float()
        else:
            edge_attr_scalar = pyg_batch.edge_attr.float()

        bond_types = to_dense_adj(
            pyg_batch.edge_index,
            pyg_batch.batch,
            edge_attr=edge_attr_scalar,
            max_num_nodes=max_nodes,
        ).long()
    else:
        bond_types = torch.zeros_like(adj, dtype=torch.long)

    # Prepare atom types
    atom_type_vectors = torch.zeros(batch_size, max_nodes, dtype=torch.long)

    if hasattr(pyg_batch, "atom_types"):
        for i_batch in range(batch_size):
            mask = pyg_batch.batch == i_batch
            n_nodes = mask.sum().item()
            atom_type_vectors[i_batch, :n_nodes] = pyg_batch.atom_types[mask].argmax(
                dim=1
            )

    # Get sizes
    sizes = [(pyg_batch.batch == i).sum().item() for i in range(batch_size)]

    targets = {
        "adj": adj,
        "bond_types": bond_types,
        "atom_types": atom_type_vectors,
        "sizes": sizes,
    }

    return targets


class MolecularDataset(InMemoryDataset):
    ATOM_TYPES = [
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "B",
        "Si",
        "Se",
        "Cl",
        "Br",
        "I",
        "H",
    ]

    VALENCES = [
        4,
        3,
        2,
        1,
        5,
        2,
        3,
        4,
        2,
        1,
        1,
        1,
        1,
    ]

    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
    ]

    # Hybridization types
    HYBRIDIZATION_TYPES = [
        Chem.HybridizationType.S,
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ]

    def __init__(
        self,
        smiles_list: List[str],
        root: Optional[str] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        y=None,
    ):
        self.smiles_list = smiles_list
        self.y = y
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        """Process SMILES strings into PyG Data objects."""
        data_list = []

        for idx, smiles in tqdm(enumerate(self.smiles_list)):
            data = self.smiles_to_graph(smiles)
            if data is not None:
                data["index"] = idx

                if self.y is not None:
                    data["y"] = self.y[idx]

                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Canonicalize a SMILES string using RDKit.

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            return canonical_smiles
        except Exception as e:
            print(f"Error canonicalizing SMILES '{smiles}': {e}")
            return None

    def get_atom_types(self, atom) -> np.ndarray:
        atom_type = atom.GetSymbol()
        features = np.zeros(len(self.ATOM_TYPES) + 1)

        if atom_type in self.ATOM_TYPES:
            idx = self.ATOM_TYPES.index(atom_type)
            features[idx] = 1
        else:
            features[-1] = 1  # "other" category

        return features

    def get_enhanced_atom_features(self, atom, mol) -> np.ndarray:
        features = []

        # 1. One-hot atom type (13 features)
        atom_type = atom.GetSymbol()
        atom_type_one_hot = np.zeros(len(self.ATOM_TYPES) + 1)
        if atom_type in self.ATOM_TYPES:
            idx = self.ATOM_TYPES.index(atom_type)
            atom_type_one_hot[idx] = 1
        else:
            atom_type_one_hot[-1] = 1
        features.append(atom_type_one_hot)

        # 2. Atomic number (normalized by max atomic number in dataset ~118)
        atomic_num = atom.GetAtomicNum() / 118.0
        features.append([atomic_num])

        # 3. Degree (number of neighbors) - normalized by max possible degree (4)
        degree = atom.GetDegree() / 4.0
        features.append([degree])

        # 4. Implicit valence - normalized
        implicit_valence = atom.GetImplicitValence() / 5.0
        features.append([implicit_valence])

        # 5. Formal charge - normalized
        formal_charge = (atom.GetFormalCharge() + 2) / 4.0  # Range: -2 to +2
        features.append([formal_charge])

        # 6. Hybridization type (one-hot, 6 types)
        hybridization = atom.GetHybridization()
        hybridization_one_hot = np.zeros(len(self.HYBRIDIZATION_TYPES))
        if hybridization in self.HYBRIDIZATION_TYPES:
            idx = self.HYBRIDIZATION_TYPES.index(hybridization)
            hybridization_one_hot[idx] = 1
        features.append(hybridization_one_hot)

        # 7. Is aromatic
        is_aromatic = float(atom.GetIsAromatic())
        features.append([is_aromatic])

        # 8. Number of H atoms - normalized
        num_hs = atom.GetTotalNumHs() / 4.0
        features.append([num_hs])

        # 9. Is in ring
        is_in_ring = float(atom.IsInRing())
        features.append([is_in_ring])

        # 10. Accessibility - number of H bond donors/acceptors
        num_h_bond_donors = (
            sum(
                1
                for neighbor in atom.GetNeighbors()
                if neighbor.GetSymbol() in ["N", "O"]
            )
            / 4.0
        )
        features.append([num_h_bond_donors])

        # Concatenate all features
        combined_features = np.concatenate(features)
        return combined_features

    def get_bond_features(self, bond) -> np.ndarray:
        """
        Get one-hot encoded bond features.

        Args:
            bond:  RDKit bond object

        Returns:
            One-hot encoded array of length len(BOND_TYPES)
        """
        bond_type = bond.GetBondType()
        features = np.zeros(len(self.BOND_TYPES))

        if bond_type in self.BOND_TYPES:
            idx = self.BOND_TYPES.index(bond_type)
            features[idx] = 1

        return features

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert a SMILES string to a PyTorch Geometric Data object.

        Args:
            smiles:  SMILES string

        Returns:
            PyG Data object or None if conversion fails
        """
        # Canonicalize SMILES
        canonical_smiles = self.canonicalize_smiles(smiles)
        if canonical_smiles is None:
            print(f"Failed to canonicalize SMILES: {smiles}")
            return None

        # Create molecule object
        mol = Chem.MolFromSmiles(canonical_smiles)

        if mol is None:
            return None

        Chem.Kekulize(mol, clearAromaticFlags=True)

        atom_types = []
        node_features = []

        for atom in mol.GetAtoms():
            atom_types.append(self.get_atom_types(atom))
            node_features.append(self.get_enhanced_atom_features(atom, mol))

        # Set node features
        x = torch.tensor(np.array(node_features), dtype=torch.float)

        # Extract edges and edge features
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Add both directions for undirected graph
            edge_indices.append([i, j])
            edge_indices.append([j, i])

            bond_feature = self.get_bond_features(bond)
            edge_features.append(bond_feature)
            edge_features.append(bond_feature)

        if len(edge_indices) == 0:
            # Molecule with no bonds (single atom)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.BOND_TYPES)), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)

        # Create Data object
        data = Data(
            x=x,
            atom_types=torch.tensor(np.array(atom_types), dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=canonical_smiles,
        )

        return data
