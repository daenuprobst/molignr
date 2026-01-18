import torch
from data_ import get_dataset
from models.model_cIGNR import *

import MDAnalysis as mda
import matplotlib.pyplot as plt

from torch_geometric.utils import to_dense_adj
from utils import load_model





idx = {
    "V153": 183,   # CYS 25  CA
    "D250": 369,   # LEU 50  CA
    "T337": 676,   # ALA 89  CA
    "N342": 711,   # LEU 94  CA
    "I442": 938,   # THR 122 CA
    "W566": 1467,  # ARG 188 CA
    "A634": 1640,  # GLN 208 CA
    "L658": 1827,  # ILE 232 CA
    "Y735": 1912,  # SER 244 CA
    "L755": 2069   # PHE 264 CA
}



def a100_from_coords(coords:torch.Tensor, idx:dict):

    device = coords.device
    idx_order = torch.tensor([
        idx["V153"], idx["L755"],
        idx["D250"], idx["T337"],
        idx["N342"], idx["I442"],
        idx["W566"], idx["A634"],
        idx["L658"], idx["Y735"]
    ], device=device, dtype=torch.long)

    # get just the 10 atoms
    sel = coords.index_select(dim=1, index=idx_order)
   

    print(f"sel.shape: {sel.shape}")

    # pair them up and compute distances
    pair_ids = torch.tensor([[0,1],[2,3],[4,5],[6,7],[8,9]], device=device, dtype=torch.long)
    diffs = sel[:, pair_ids[:,0], :] - sel[:, pair_ids[:,1], :]
    dists = torch.linalg.norm(diffs, dim =-1)

    print("Distances (Å):", dists[0].detach().cpu().numpy())

    coeffs = torch.tensor([-14.43, -7.62,  9.11, -6.32, -5.22], device=device, dtype=coords.dtype)
    a100 = (dists * coeffs).sum(dim=1) + 278.88

    return a100




import numpy as np



def bw_to_resid_from_gpcrdb_pdb(pdb_path, chain_id = "A", tol = 1e-2):
    """
    Given a PDB file from GPCRdb, return a mapping from Ballesteros-Weinstein
    numbering to residue index in the PDB file.

    Args:
        pdb_path (str): Path to the PDB file.
        chain_id (str): Chain identifier to consider.
        tol (float): Tolerance for matching residue numbers.

    Returns: dict mapping BW numbering (str) to residue index (int).
    """

    bw2resid = {}

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain_id:
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue            
            
            resid = int(line[22:26])
            bfac = float(line[60:66])

            # store with rounding to avoid float issues
            bw_key = round(bfac,2)
            bw2resid[bw_key] = resid


    return bw2resid



def bw_pairs_to_ca_indices(pdf_path, bw2resid, bw_pairs, chain_sel = "segid P", tol =1e-2):
    """
    bw_pairs: list of (bwA, bwB) floats
    Returns: list of (idxA, idxB) atom indices in the MDA
    """

    u = mda.Universe(pdf_path)

    pairs_idx = []

    for bwA, bwB in bw_pairs:
        bwA - round(bwA,2)
        bwB - round(bwB,2)

        if bwA not in bw2resid or bwB not in bw2resid:
            raise ValueError(f"BW numbering {bwA} or {bwB} not found in mapping.")  
        
        residA = bw2resid[bwA]
        residB = bw2resid[bwB]

        a = u.select_atoms(f"protein and {chain_sel} and resid {residA} and name CA")
        b = u.select_atoms(f"protein and {chain_sel} and resid {residB} and name CA")
        if len(a) ==0 or len(b) ==0:
            raise ValueError(f"CA not found for resid {residA} or {residB}. Check segid/chain selection.")
        
        pairs_idx.append((a[0].index, b[0].index))


        return pairs_idx
    



def activation_index_from_coords(coords, pairs_idx):
    """
    coords: (T,N,3) tensor of coordinates
    pairs_idx : 5 pairs of CA atom indices
    Returns : index (T,) and distances (T,5)
    """

    coeffs = torch.tensor([-14.43, -7.62,  9.11, -6.32, -5.22], device=coords.device, dtype=coords.dtype)
    bias   = torch.tensor(278.88, device=coords.device, dtype=coords.dtype)

    if coords.dim() ==2:
        coords = coords.unsqueeze(0)  # (1,N,3)
        squeeze = True
    else:
        squeeze = False

    dists = []
    for (i,j) in pairs_idx:
        d = torch.linalg.norm(coords[:, i, :] - coords[:, j, :], dim =-1)
        dists.append(d)

    dists = torch.stack(dists, dim = -1)
    idx = (dists * coeffs).sum(dim = -1) + bias

    return (idx[0], dists[0]) if squeeze else (idx, dists) 

bw_pairs = [
    (1.53, 7.55),
    (2.50, 3.37),
    (3.42, 4.42),
    (5.66, 6.34),
    (6.58, 7.35),
]






def a100_plots(model, train_loader, prog_args, gpcrdb_pdb = "heavy_chain_GPCRDB.pdb", plot_name = "hist_a100"):

    bw2_resid = bw_to_resid_from_gpcrdb_pdb(gpcrdb_pdb, chain_id = "A", tol = 1e-2)
    pairs_idx = bw_pairs_to_ca_indices(gpcrdb_pdb, bw2_resid, bw_pairs, chain_sel = "chainID A", tol =1e-2)

    ais = []
    dists_list = []
    ais_pred = []
    dists_list_pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            coords = data.x.view(-1, 2191, 3).to(model.device)  # (B,N,3)

            x = data.x.float().to(prog_args.device)

            edge_index = data.edge_index.to(torch.int64).to(prog_args.device)
            batch = data.batch.to(torch.int64).to(prog_args.device)
            coords_pred = model(x, edge_index, batch) 
            coords_pred = coords_pred.view(-1, 2191, 3)  # (B,N,3)
            
            ai_pred, dist_pred = activation_index_from_coords(coords_pred, pairs_idx)
            ais_pred.extend(np.round(ai_pred.cpu().numpy(), 3).tolist())
            dists_list_pred.append(dist_pred.cpu().numpy())

            ai, dists = activation_index_from_coords(coords, pairs_idx)
            dists_list.append(dists.cpu().numpy())
            ais.extend(np.round(ai.cpu().numpy(), 3).tolist())

    dists_list = np.concatenate(dists_list, axis=0)
    dists_list_pred = np.concatenate(dists_list_pred, axis=0)
    print("dists mean (5):", dists_list.mean(axis=0))
    print("dists min/max:", dists_list.min().item(), dists_list.max().item())
    print("ai min/max:", np.array(ais).min().item(), np.array(ais).max().item())

    print("dists mean (5):", dists_list_pred.mean(axis=0))
    print("dists min/max:", dists_list_pred.min().item(), dists_list_pred.max().item())
    print("ai min/max:", np.array(ais_pred).min().item(), np.array(ais_pred).max().item())

    arr1 = np.array(ais)
    arr2 = np.array(ais_pred)

    x = np.linspace(
        min(arr1.min(), arr2.min()),
        max(arr1.max(), arr2.max()),
        500
    )

    plt.figure(figsize=(6,4))
    plt.hist(arr1, bins=50, density=True,
            alpha=0.4, color="tab:blue", label="Dataset")

    plt.hist(arr2, bins=50, density=True,
            alpha=0.4, color="tab:orange", label="Predictions")

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_name, dpi=300)





if __name__ == "__main__":

    path = "/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/FINAL_RESULT/checkpoints_final/lr_0.01/full_data_knn_4_chebnet_dim_8_epoch_140_knn_4_lr_0.01.pt"

    model, prog_args = load_model(path)
    model.eval()

    dataset, train_loader, test_loader = get_dataset(prog_args)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    gpcrdb_pdb = "heavy_chain_GPCRDB.pdb"

    bw2_resid = bw_to_resid_from_gpcrdb_pdb(gpcrdb_pdb, chain_id = "A", tol = 1e-2)
    pairs_idx = bw_pairs_to_ca_indices(gpcrdb_pdb, bw2_resid, bw_pairs, chain_sel = "chainID A", tol =1e-2)

    ais = []
    dists_list = []
    ais_pred = []
    dists_list_pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            coords = data.x.view(-1, 2191, 3).to(model.device)  # (B,N,3)

            x = data.x.float().to(prog_args.device)

            edge_index = data.edge_index.to(torch.int64).to(prog_args.device)
            batch = data.batch.to(torch.int64).to(prog_args.device)
            C_input = to_dense_adj(edge_index, batch=batch).to(prog_args.device)
            loss, coords_pred = model(x, edge_index, batch, C_input, prog_args.M)  
            coords_pred = coords_pred.view(-1, 2191, 3)  # (B,N,3)
            
            ai_pred, dist_pred = activation_index_from_coords(coords_pred, pairs_idx)
            ais_pred.extend(np.round(ai_pred.cpu().numpy(), 3).tolist())
            dists_list_pred.append(dist_pred.cpu().numpy())

            ai, dists = activation_index_from_coords(coords, pairs_idx)
            dists_list.append(dists.cpu().numpy())
            ais.extend(np.round(ai.cpu().numpy(), 3).tolist())

    dists_list = np.concatenate(dists_list, axis=0)
    dists_list_pred = np.concatenate(dists_list_pred, axis=0)
    print("dists mean (5):", dists_list.mean(axis=0))
    print("dists min/max:", dists_list.min().item(), dists_list.max().item())
    print("ai min/max:", np.array(ais).min().item(), np.array(ais).max().item())

    print("dists mean (5):", dists_list_pred.mean(axis=0))
    print("dists min/max:", dists_list_pred.min().item(), dists_list_pred.max().item())
    print("ai min/max:", np.array(ais_pred).min().item(), np.array(ais_pred).max().item())

    # Convert to numpy array (optional but recommended)
    from scipy.stats import gaussian_kde
    arr1 = np.array(ais)
    arr2 = np.array(ais_pred)

    x = np.linspace(
        min(arr1.min(), arr2.min()),
        max(arr1.max(), arr2.max()),
        500)

    plt.figure(figsize=(6,4))
    plt.hist(arr1, bins=50, density=True,
            alpha=0.4, color="tab:blue", label="Dataset")
    plt.hist(arr2, bins=50, density=True,
            alpha=0.4, color="tab:orange", label="Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hist_kde_.png", dpi=300)





    """
    IDX = {
    "V153": 183, "D250": 369, "T337": 676, "N342": 711, "I442": 938,
    "W566": 1467, "A634": 1640, "L658": 1827, "Y735": 1912, "L755": 2069
    }

    def atom_lines(pdb_path):
        lines = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    lines.append(line.rstrip("\n"))
        return lines

    lines = atom_lines("/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/OTHERS/heavy_chain.pdb")

    for k, i in IDX.items():
        print(k, i, "->", lines[i])

    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            x = data.x
            x = x.view(-1, 2191, 3) 
            print(x[0, 183])
            print(x[0].min().item(), x[0].max().item())

            a = a100_from_coords(x, idx)
            break
            print(f"a100 shape: {a.shape}, a100: {a}")
    """