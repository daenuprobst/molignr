from typing import Optional, List
import pandas as pd
import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.utils import to_dense_adj
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch.utils.data import DataLoader
from tqdm import tqdm


# --- Core Invariants Logic (from molsetrep/encoders/common.py) ---
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP2D,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

MOLNET_TASKS = {
    "bace": ["Class"],
    "bbbp": ["p_np"],
    "clintox": ["FDA_APPROVED", "CT_TOX"],
    "esol": ["ESOL predicted log solubility in mols per litre"],
    "freesolv": ["expt"],
    "hiv": ["HIV_active"],
    "lipo": ["exp"],
    "muv": [
        "MUV-692",
        "MUV-689",
        "MUV-846",
        "MUV-859",
        "MUV-644",
        "MUV-548",
        "MUV-852",
        "MUV-600",
        "MUV-810",
        "MUV-712",
        "MUV-737",
        "MUV-858",
        "MUV-713",
        "MUV-733",
        "MUV-652",
        "MUV-466",
        "MUV-832",
    ],
    "qm7": ["u0_atom"],
    "qm8": [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ],
    "qm9": ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv"],
    "sider": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
}


def get_molecular_graph_targets(pyg_batch, max_nodes):
    batch_size = pyg_batch.batch.max().item() + 1

    adj = to_dense_adj(pyg_batch.edge_index, pyg_batch.batch, max_num_nodes=max_nodes)

    if (
        hasattr(pyg_batch, "edge_attr_bond_type")
        and pyg_batch.edge_attr_bond_type is not None
    ):
        bond_types = to_dense_adj(
            pyg_batch.edge_index,
            pyg_batch.batch,
            edge_attr=pyg_batch.edge_attr_bond_type.float(),
            max_num_nodes=max_nodes,
        ).long()

    elif hasattr(pyg_batch, "edge_attr") and pyg_batch.edge_attr is not None:
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


def molecular_collate_fn(batch):
    pyg_batch = Batch.from_data_list(batch)

    num_nodes_per_graph = torch.bincount(pyg_batch.batch)
    max_nodes = num_nodes_per_graph.max().item()
    targets = get_molecular_graph_targets(pyg_batch, max_nodes)

    indices = torch.tensor([item["index"] for item in batch])

    return pyg_batch, targets, indices


def one_hot_encode(prop, vals):
    if not isinstance(vals, (list, range, np.ndarray)):
        vals = range(vals)
    result = [1 if prop == i else 0 for i in vals]
    result.append(1 if sum(result) == 0 else 0)
    return result


def get_atom_features(atom):
    f = []
    f += one_hot_encode(atom.GetTotalDegree(), 6)
    f += one_hot_encode(atom.GetAtomicNum(), 100)
    f += one_hot_encode(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    f += one_hot_encode(atom.GetHybridization(), HYBRIDIZATION_TYPES)
    f += one_hot_encode(atom.GetChiralTag(), 4)
    f.append(float(atom.IsInRing()))
    f += one_hot_encode(atom.GetTotalNumHs(), 6)
    f.append(0.01 * atom.GetMass())
    return f


def get_bond_features(bond):
    f = []
    f += one_hot_encode(bond.GetBondType(), BOND_TYPES)
    f += one_hot_encode(int(bond.GetStereo()), 6)
    f.append(float(bond.GetIsAromatic()))
    f.append(float(bond.GetIsConjugated()))
    return f


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
        labels=None,
        root: Optional[str] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.smiles_list = smiles_list
        self.labels = labels
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
        data_list = []
        for i, (smi, lbl) in enumerate(zip(self.smiles_list, self.labels)):
            mol = Chem.MolFromSmiles(smi)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            if not mol:
                mol = Chem.MolFromSmiles(smi.replace("[NH+2]", "[NH+1]"))
            if not mol:
                continue

            # Make sure to have canonical atom order
            mol = self.canonicalize_mol(mol)

            atom_types = [self.get_atom_types(atom) for atom in mol.GetAtoms()]

            nodes = [get_atom_features(a) for a in mol.GetAtoms()]
            edge_idx, edge_attr = [], []
            for b in mol.GetBonds():
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                attr = get_bond_features(b)
                edge_idx.extend([[u, v], [v, u]])
                edge_attr.extend([attr, attr])

            data = Data(
                x=torch.tensor(nodes, dtype=torch.float),
                edge_index=torch.tensor(edge_idx, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                y=torch.tensor(lbl, dtype=torch.float).view(1, -1),
                atom_types=torch.tensor(np.array(atom_types), dtype=torch.float),
                smiles=smi,
            )

            data["index"] = i

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_atom_types(self, atom) -> np.ndarray:
        atom_type = atom.GetSymbol()
        features = np.zeros(len(self.ATOM_TYPES) + 1)

        if atom_type in self.ATOM_TYPES:
            idx = self.ATOM_TYPES.index(atom_type)
            features[idx] = 1
        else:
            features[-1] = 1  # "other" category

        return features

    def canonicalize_mol(self, mol):
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))


def get_data_loader(
    smiles_list,
    labels,
    root=None,
    batch_size=32,
    shuffle=False,
    num_workers=12,
    transform=None,
    pre_transform=None,
    pre_filter=None,
):
    data_set = MolecularDataset(
        smiles_list,
        labels,
        root,
        transform=transform,
        pre_transform=pre_transform,
        pre_filter=pre_filter,
    )

    return DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: molecular_collate_fn(batch),
        num_workers=num_workers,
    )

    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, row in dataset.iterrows():
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))

        scaffold = _generate_scaffold(row["smiles"])

        # Adapted from original to account for SMILES not readable by MolFromSmiles
        if scaffold is not None:
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size=0.1, test_size=0.1, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def load_molnet_dataset(name, csv_path, root=None, batch_size=32, split="scaffold"):
    df = pd.read_csv(csv_path)
    tasks = MOLNET_TASKS.get(
        name.lower(), [df.columns[1]]
    )  # Fallback to 2nd col if task unknown

    if split == "scaffold":
        train_idx, valid_idx, test_idx = scaffold_split(df)
        train_df, valid_df, test_df = (
            df.iloc[train_idx],
            df.iloc[valid_idx],
            df.iloc[test_idx],
        )
    else:  # Random 80/10/10
        train_df, valid_df, test_df = np.split(
            df.sample(frac=1), [int(0.8 * len(df)), int(0.9 * len(df))]
        )

    return (
        get_data_loader(
            train_df["smiles"],
            train_df[tasks].values,
            root=f"{root}_train",
            batch_size=batch_size,
            shuffle=True,
        ),
        get_data_loader(
            valid_df["smiles"],
            valid_df[tasks].values,
            root=f"{root}_valid",
            batch_size=batch_size,
        ),
        get_data_loader(
            test_df["smiles"],
            test_df[tasks].values,
            root=f"{root}_test",
            batch_size=batch_size,
        ),
    )


def load_adme_dataset(name, folder_path, root=None, batch_size=32):
    train_df = pd.read_csv(f"{folder_path}/ADME_{name}_train.csv")
    test_df = pd.read_csv(f"{folder_path}/ADME_{name}_test.csv")
    valid_df = train_df.sample(frac=0.1)

    return (
        get_data_loader(
            train_df["smiles"],
            train_df[["activity"]].values,
            root=f"{root}_train",
            batch_size=batch_size,
            shuffle=True,
        ),
        get_data_loader(
            valid_df["smiles"],
            valid_df[["activity"]].values,
            root=f"{root}_valid",
            batch_size=batch_size,
        ),
        get_data_loader(
            test_df["smiles"],
            test_df[["activity"]].values,
            root=f"{root}_test",
            batch_size=batch_size,
        ),
    )


if __name__ == "__main__":
    # train, valid, test = load_adme_dataset("MDR1_ER", "data/adme")
    train, valid, test = load_molnet_dataset("bace", "data/moleculenet/bace.csv.xz")
    print(train)
