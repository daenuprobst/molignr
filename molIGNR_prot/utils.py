import torch
import argparse
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

from data_ import get_dataset
import os
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from torch_geometric.utils import to_dense_adj
from pathlib import Path

from evaluation.metrics import lddt, tm_score

def arg_parse():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    ### Optimization parameter
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    
    ### Training specific
    parser.add_argument('--n_epoch', dest='n_epoch', type=int,
            help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')

    parser.add_argument('--save_dir', dest='save_dir',
            help='name of the saving directory')

    # General model param
    parser.add_argument('--gnn_num_layer', dest='gnn_num_layer', type=int)

    parser.add_argument('--M',dest='M',type=int) #sampling number for graph reconstruction if needed

    ### SIREN-MLP-model specific
    parser.add_argument('--mlp_dim_hidden', dest='mlp_dim_hidden') #hidden dim (number of neurons) for f_theta
    parser.add_argument('--latent_dim', dest='latent_dim', type=int) #from graph embedding to latent embedding, reducing graph embedding dimension
    parser.add_argument('--mlp_act', dest='mlp_act') # whether to use sine activation for the mlps

    parser.add_argument('--gnn_type', dest='gnn_type') # to specify the type of encoder used
    parser.add_argument('--gnn_layers', nargs='+', type=int, default=[8,8,8])


    parser.add_argument('--save_output', action="store_true")  # python train.py --save_output ---> True 

    parser.set_defaults(dataset='full_atom_structure_knn_4.pt',
                        lr=1e-4,
                        n_epoch=120,
                        batch_size=16,
                        gnn_num_layer=3,
                        latent_dim=16,
                        mlp_dim_hidden='48,36,24',
                        mlp_act = 'sine',
                        M=0)
    return parser.parse_args()



def write_pdb_with_new_coords(
    pdb_in: str,
    coords,               # (N,3) numpy array or torch tensor
    pdb_out: str):
    """
    Replace the coordinates of the atoms in the given pdb with the predicted coordinates. Atom order is fixed. 

    :param pdb_in: path for the original pdb
    :type pdb_in: str
    :param coords: predicted coordinates 
    :param pdb_out: path for the new pdb
    :type pdb_out: str
    """

    # Convert coords to numpy
    if hasattr(coords, "detach"):  
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=float)

    coord_i = 0
    out_lines = []

    with open(pdb_in, "r") as f:
        for line in f:
            record = line[:6].strip()

            is_atom = (record == "ATOM")
            is_het  = (record == "HETATM")

            if is_atom or is_het:
                if coord_i >= len(coords):
                    raise ValueError(
                        f"Not enough coordinates: need >={coord_i+1}, have {len(coords)}"
                    )

                x, y, z = coords[coord_i]
                coord_i += 1

                # Keep everything else the same, only overwrite x/y/z with PDB formatting
                new_line = (
                    line[:30]
                    + f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    + line[54:]
                )
                out_lines.append(new_line)
            else:
                out_lines.append(line)

    if coord_i != len(coords):
        print(f"Warning: {len(coords)} coords provided, but only used {coord_i} (ATOM/HETATM lines).")

    with open(pdb_out, "w") as f:
        f.writelines(out_lines)

    print(f"Wrote {pdb_out} with {coord_i} updated atoms.")


################  VISUALISE the Latent Space  ####################

def plot_latents(latent_codes, save_plot_path):
    """
    Visualise the latent representations
    
    :param latent_codes: normalized latent representations
    :param save_plot_path: path for the plot
    """
    num_frames = latent_codes.shape[0]
    frame_indices = np.arange(num_frames)
    cmap = cm.get_cmap('turbo', num_frames)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        latent_codes[:, 0], latent_codes[:, 1],
        c=frame_indices,
        cmap=cmap,
        s=40,
        edgecolor='k')

    plt.colorbar(label='Frame index (time)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_plot_path, dpi=300) 


def visualize_latents(prog_args, model, train_loader, epoch = None):
    """
    Visualise the latent representations using PCA, t-SNE and UMAP. 
    
    :param prog_args: model hyperpar
    :param model: Description
    :param train_loader: Description
    :param epoch: Description
    """
    model = model.to(torch.device(device))
    model.eval()
    all_zs, loss_mse, loss_gw, lddt_list, tm_score_list = [], [], [], [], []
    N =2191 # number of atoms in the full atom structure
    ## Get the latent encodings...
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for batch_idx, data in enumerate(train_loader):
                x = data.x.float().to(device)
                edge_index = data.edge_index.to(torch.int64).to(device)
                batch = data.batch.to(torch.int64).to(device)
                C_input = to_dense_adj(edge_index, batch=batch).to(device)

                z, node_representation = model.encode(x, edge_index, batch.to(device))
                all_zs.append(z.detach().cpu())

                coords_pred = model.mlp_coords(z.to(device))
                coords_pred = coords_pred.view(-1,3)

                loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
                loss_mse.append(loss_coords.item())

                loss, z, C_recon_list, mods = model.decode(z.to(device), C_input.to(device), N, batch.to(device))
                loss_gw.append(loss.item())

                lddt_ = lddt(coords_pred.view(prog_args.batch_size, N, 3), x.view(prog_args.batch_size, N, 3))
                tm_score_ = tm_score(x, coords_pred)

                lddt_list.append(lddt_)
                tm_score_list.append(tm_score_)

                if batch_idx % 200 ==0:
                    print(f"Batch {batch_idx}, MSE loss: {loss_coords.item():.4f}, GW loss: {loss.item():.4f}")
                    print(f"Training LDDT score {lddt_}")
                    print(f"Training TM-Score {tm_score_}")

    print(f"MSE loss: {np.mean(loss_mse)}")
    print(f"GW loss: {np.mean(loss_gw)}")

    print(f"LDDT in test set : {np.round(np.mean(lddt_list),3)}")
    print(f"TM-Score in test set : {np.round(np.mean(tm_score_list),3)}")

    latent_codes = torch.cat(all_zs, dim=0)
    latent_codes = latent_codes.detach().cpu().numpy().astype(np.float32)

    print("Has NaNs?", np.isnan(latent_codes).any())
    print("Has Infs?", np.isinf(latent_codes).any())
    print("Max / Min values:", latent_codes.max(), latent_codes.min())

    latent_codes  = StandardScaler().fit_transform(latent_codes) 

    ################## PCA #################
    pca = PCA(n_components=2)
    latent_codes_pca = pca.fit_transform(latent_codes)

    explained_var = pca.explained_variance_
    explained_var_ratio = pca.explained_variance_ratio_

    print("Explained variance:", explained_var)
    print("Explained variance ratio:", explained_var_ratio)
    print("Cumulative explained variance ratio:", np.cumsum(explained_var_ratio))

    ################## TSNE #####################
    latent_codes_tsne = TSNE(n_components=2, perplexity=20, init='pca', random_state=0).fit_transform(latent_codes)
    
    ################### UMAP ###################
    reducer = umap.UMAP()
    latent_codes_umap = reducer.fit_transform(latent_codes)

    ppath = os.getcwd() + '/Results/plots/'
    base = Path(ppath)
    base.mkdir(parents=True, exist_ok=True)

    pca_dir = base / "pca" / f"lr_{prog_args.lr}"
    tsne_dir = base / "tsne" / f"lr_{prog_args.lr}"
    umap_dir = base / "umap" / f"lr_{prog_args.lr}"

    # Create directories
    for d in [pca_dir, tsne_dir, umap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save the plots in the created directory
    pca_file  = pca_dir  / f"{prog_args.gnn_type}_{prog_args.repr}_epoch_{epoch}_lr_{prog_args.lr}_pca.png"
    tsne_file = tsne_dir / f"{prog_args.gnn_type}_{prog_args.repr}_epoch_{epoch}_lr_{prog_args.lr}_tsne.png"
    umap_file = umap_dir / f"{prog_args.gnn_type}_{prog_args.repr}_epoch_{epoch}_lr_{prog_args.lr}_umap.png"

    plot_latents(latent_codes_pca,  pca_file)
    plot_latents(latent_codes_tsne, tsne_file)
    plot_latents(latent_codes_umap, umap_file)
