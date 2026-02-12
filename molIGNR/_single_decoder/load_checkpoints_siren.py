from sklearn.discriminant_analysis import StandardScaler
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

from pathlib import Path
import torch
import sys
from models.model_siren import cIGNR
from models.siren_pytorch import SirenNet
from data_ import get_dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from pathlib import Path
import os


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)  # or 'cuda' if using GPU
    prog_args = checkpoint['configs']

    
    snet_adj = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = prog_args.mlp_dim_hidden,
        dim_out = 1, # output graphon (edge) probability 
        num_layers = prog_args.mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = prog_args.mlp_act )
    
    N = 2191    
    
    model = cIGNR(net_adj=snet_adj, input_card=3, emb_dim = prog_args.emb_dim, latent_dim = prog_args.latent_dim, 
                  num_layer=prog_args.gnn_num_layer, 
                  gnn_layers= prog_args.gnn_layers,
                  device=device, flag_emb=prog_args.flag_emb, gnn_type = prog_args.gnn_type, N = N,
                  representation = prog_args.repr)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, prog_args, checkpoint['epoch']




def plot_latents(latent_codes, save_plot_path):
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
    plt.title('Latent codes over time')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_plot_path, dpi=300) 
    
def visualize_latents(prog_args, model, train_loader, epoch = None):
    model = model.to(torch.device(device))
    model.eval()

    all_zs = []
    loss= [] 

    ## Get the latent encodings...
    for batch_idx, data in enumerate(train_loader):
        x = data.x.float().to(device)
        
        edge_index = data.edge_index.to(torch.int64).to(device)
        batch = data.batch.to(torch.int64)

        z, node_representation = model.encode(x, edge_index, batch.to(device))
        all_zs.append(z.detach().cpu())

        # loss.append(loss.item())

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

    ppath = os.getcwd()
    base = Path(f"{ppath}/Results/plots/")
    base.mkdir(parents=True, exist_ok=True)

    pca_dir = base / "pca" / f"lr_{prog_args.lr}"
    tsne_dir = base / "tsne" / f"lr_{prog_args.lr}"
    umap_dir = base / "umap" / f"lr_{prog_args.lr}"

    # Create directories
    for d in [pca_dir, tsne_dir, umap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    gnn_type, latent_dim, dataset = prog_args.gnn_type, prog_args.latent_dim, prog_args.dataset

    # Now build filenames correctly
    pca_file  = pca_dir  / f"siren_{prog_args.gnn_type}_{prog_args.repr}_epoch_{epoch}_dim_{latent_dim}_lr_{prog_args.lr}_pca.png"
    tsne_file = tsne_dir / f"siren_{prog_args.gnn_type}_{prog_args.repr}_epoch_{epoch}_dim_{latent_dim}_lr_{prog_args.lr}_tsne.png"
    umap_file = umap_dir / f"siren_{prog_args.gnn_type}_{prog_args.repr}_epoch_{epoch}_dim_{latent_dim}_lr_{prog_args.lr}_umap.png"

    plot_latents(latent_codes_pca,  pca_file)
    plot_latents(latent_codes_tsne, tsne_file)
    plot_latents(latent_codes_umap, umap_file)




def test(args, test_loader, model, epoch):
    model.eval()
    loss_coords_ = []

    lddt_list = []
    tm_score_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            x = data.x.float().to(args.device)
            edge_index = data.edge_index.to(torch.int64).to(args.device)
            batch = data.batch.to(torch.int64).to(args.device)

            coords_pred = model(x, edge_index, batch)        
            loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))

            print(f"Loss coords : {loss_coords.item():.4f}")
            loss_coords_.append(loss_coords.item())

            N = x.shape[0]//args.batch_size
            lddt_ = lddt(coords_pred.view(args.batch_size, N, 3), x.view(args.batch_size, N, 3))
            tm_score_ = tm_score(x, coords_pred)

            lddt_list.append(lddt_)
            tm_score_list.append(tm_score_)

            print(f"Training LDDT score {lddt_}")
            print(f"Training TM-Score {tm_score_}")

        loss_batch_coords = np.mean(loss_coords_)

        lddt_epoch = np.mean(lddt_list)
        tm_score_epoch = np.mean(tm_score_list)

        print()
        print(f"LDDT in test set : {np.round(lddt_epoch,3)}")
        print(f"TM-Score in test set : {np.round(tm_score_epoch,3)}")
        print()
        print(f"Coords loss in test set coords: {np.round(loss_batch_coords,3)}")
        print()

    return np.round(loss_batch_coords, 3)


    




if __name__=="__main__":
    ppath = os.getcwd()
    path = "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.001/siren_gin_epoch_60_bs_16_dim_8_knn_4_lr_0.001_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.001/siren_gin_epoch_50_bs_16_dim_8_knn_4_lr_0.001_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.001/siren_gin_epoch_30_bs_16_dim_8_knn_4_lr_0.001_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.001/siren_gin_epoch_10_bs_16_dim_8_knn_4_lr_0.001_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.01/siren_gin_epoch_0_bs_16_dim_8_knn_4_lr_0.01_best_model.pt"

    model, args, epoch_ = load_model(path)

    train_loader, test_loader, _ = get_dataset(args, shuffle=False)

    print("running ....")

    visualize_latents(args, model, train_loader, epoch = epoch_)
    print("visualization done....")
    test(args, train_loader, model, epoch_)
 
    #ramachandran_full_data(train_loader, args, plot_name = "full_data_histogram", N =2191)
    #ramachandran_reconstruction(model, train_loader, args, plot_name = f"R_no_siren_rc_{args.repr}")
    #ramachandran_interpolation(model, train_loader, args, plot_name = f"R_no_siren_interpolation_{args.repr}", N=2191)

    # a100_plots(model, train_loader, args, gpcrdb_pdb = "heavy_chain_GPCRDB.pdb", plot_name = "hist_a100")

















"""

def overwrite_pdbs(pdb_path, pred_coords):
    u = mda.Universe(pdb_path)

    print(f"u: {u}")
    atoms = u.atoms

    print(f"atoms : {atoms.positions}")
    print(f"atoms.shape : {len(atoms.positions)} - {len(atoms.positions[0])}")

    atoms.positions = pred_coords
    u.atoms.write("heavy_chain_updates.pdb")

    print(f"Wrote updated pdb file with predicted coordinates.")



def update_coords(pdb_path, pred_coords, replace_records = ("ATOMS", "HETATM")):
    i = 0
    out_lines= [] 

    with open(pdb_path, "r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in replace_records:
                if i>= len(pred_coords):
                    raise ValueError(f"More ATOM/HETATM lines than coordinates (idx={i}).")
                x,y,z = pred_coords[i]  
                print(f"pred_coords : {x},{y},{z}")
                
                new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                line = line[:30] + new_xyz + line[54:]
                i += 1
            out_lines.append(line)

            
            out_pdb = "heavy_chain_updates.pdb"
            Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
            with open(out_pdb, "w") as f:
                f.writelines(out_lines)


def update_pdb_coords(pdb_in, coords_in, pdb_out):
    coords = coords_in

    lines = Path(pdb_in).read_text().splitlines(True)  # keep newlines
    xyz_lines_idx = [i for i, L in enumerate(lines) if L.startswith(("ATOM  ", "HETATM"))]

    if len(xyz_lines_idx) != len(coords):
        raise ValueError(f"#coords ({len(coords)}) != #ATOM/HETATM lines ({len(xyz_lines_idx)})")

    # write back XYZ with 8.3 formatting, padding if line is short
    for k, i in enumerate(xyz_lines_idx):
        L = lines[i]
        if len(L) < 54:
            L = L.rstrip("\n")
            L = L + " " * (54 - len(L)) + "\n"  # ensure we have room for XYZ fields
        x, y, z = coords[k]
        xstr = f"{x:8.3f}"
        ystr = f"{y:8.3f}"
        zstr = f"{z:8.3f}"
        # replace only XYZ slices
        L = L[:30] + xstr + ystr + zstr + L[54:]
        lines[i] = L

    Path(pdb_out).write_text("".join(lines))
    print(f"Updated PDB written to {pdb_out}")

"""