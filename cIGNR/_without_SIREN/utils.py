import torch
import argparse
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

from models.model import cIGNR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from pathlib import Path
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphonAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    ### Optimization parameter
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--step_size', dest='step_size', type=float,
            help='Learning rate scheduler step size')
    parser.add_argument('--gamma', dest='gamma', type=float,
            help='Learning rate scheduler gamma')

    ### Training specific
    parser.add_argument('--n_epoch', dest='n_epoch', type=int,
            help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--cuda', dest='cuda', type=int,
            help='cuda device number')

    parser.add_argument('--feature', dest='feature',
            help='Feature used for encoder.')
    parser.add_argument('--save_dir', dest='save_dir',
            help='name of the saving directory')
    parser.add_argument('--flag_eval',dest='flag_eval',type=int,help='whether to compute graphon recon error') 

    # General model param
    parser.add_argument('--flag_emb',dest='flag_emb',type=int) 
    parser.add_argument('--gnn_num_layer', dest='gnn_num_layer', type=int)

    ### Model selection and sampling reconstruction number
    parser.add_argument('--M',dest='M',type=int) # sampling number for graph reconstruction if needed

    ### SIREN-MLP-model specific
    parser.add_argument('--mlp_dim_hidden', dest='mlp_dim_hidden') # hidden dim (number of neurons) for f_theta
    parser.add_argument('--emb_dim', dest='emb_dim', type=int)
    parser.add_argument('--latent_dim', dest='latent_dim', type=int) # from graph embedding to latent embedding, reducing graph embedding dimension
    parser.add_argument('--mlp_act', dest='mlp_act') # whether to use sine activation for the mlps

    parser.add_argument('--gnn_type', dest='gnn_type') # to specify the type of encoder used
    parser.add_argument('--gnn_layers', nargs='+', type=int, default=[8,8,8])
    parser.add_argument('--gnn_layers_coords', nargs='+', type=int, default=[8,8,8])

    parser.add_argument('--knn', type=int, default=5)

    parser.add_argument('--save_output', action="store_true")  

    ####### representation in the latent space
    parser.add_argument('--repr',type = str, default="graph" )

    parser.set_defaults(dataset='2ratio_rand',
                        feature='row_id',
                        lr=1e-4,
                        n_epoch=12,
                        batch_size=10,
                        cuda=0,
                        save_dir='00',
                        step_size=4,
                        gamma=0.1,
                        gnn_num_layer=3,
                        latent_dim=16,
                        emb_dim=16,
                        mlp_dim_hidden='48,36,24',
                        mlp_act = 'sine',
                        flag_emb=1,
                        flag_eval=0,
                        M=0)
    return parser.parse_args()



def set_hyperparameters(prog_args, gnn_type = "chebnet", dataset = "full_data_knn_4" , latent_dim = 8, epoch = 4000, emb_dim = 2, batch_size = 16, lr = 0.01, gnn_layers = [3, 8, 8, 8]):    
    prog_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(prog_args.device)

    ppath = os.getcwd()

    prog_args.save_path=ppath+'/Results/'+prog_args.save_dir+'/'
    if not os.path.exists(prog_args.save_path):
        os.makedirs(prog_args.save_path)
    print('saving path is:'+prog_args.save_path)


    prog_args.mlp_dim_hidden = [int(x) for x in prog_args.mlp_dim_hidden.split(',')]
    prog_args.mlp_num_layer = len(prog_args.mlp_dim_hidden)

    prog_args.dataset == dataset

    prog_args.gnn_type = gnn_type
    prog_args.latent_dim = latent_dim

    prog_args.n_epoch = epoch
    prog_args.emb_dim = emb_dim
    prog_args.batch_size = batch_size
    prog_args.lr = lr
    prog_args.gnn_layers = gnn_layers + [latent_dim]
    prog_args.gnn_num_layer = len(prog_args.gnn_layers) -1
    
    prog_args.flag_emb = 0
    prog_args.knn = 4


    return prog_args

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)  # or 'cuda' if using GPU
    prog_args = checkpoint['configs']

    prog_args.device = device

    model = cIGNR( input_card= 3, emb_dim = prog_args.emb_dim, latent_dim = prog_args.latent_dim, 
                  num_layer=prog_args.gnn_num_layer, 
                  gnn_layers= prog_args.gnn_layers,
                  device=prog_args.device, flag_emb=prog_args.flag_emb, gnn_type = prog_args.gnn_type, N =2191, representation = prog_args.repr)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model, prog_args




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
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_plot_path, dpi=300) 




    
def visualize_latents(prog_args, model, train_loader, epoch = None):
    model = model.to(torch.device(device))

    all_zs = []
    loss= [] 

    ## Get the latent encodings...
    for batch_idx, data in enumerate(train_loader):
        x = data.x.float().to(device)
        
        edge_index = data.edge_index.to(torch.int64).to(device)
        batch = data.batch.to(torch.int64)

        z, node_representation = model.encode(x, edge_index, batch.to(device))
        all_zs.append(z.detach().cpu())

        if prog_args.repr =='graph':
            coords_pred = model.mlp_coords(z.to(device))
            coords_pred = coords_pred.view(-1,3)
        elif prog_args.repr == "node":
            coords_pred = model.mlp_coords(node_representation.to(device))

        loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
        loss.append(loss_coords.item())

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
    pca_file  = pca_dir  / f"{prog_args.repr}_epoch_{epoch}_dim_{latent_dim}_lr_{prog_args.lr}_pca.png"
    tsne_file = tsne_dir / f"{prog_args.repr}_epoch_{epoch}_dim_{latent_dim}_lr_{prog_args.lr}_tsne.png"
    umap_file = umap_dir / f"{prog_args.repr}_epoch_{epoch}_dim_{latent_dim}_lr_{prog_args.lr}_umap.png"

    plot_latents(latent_codes_pca,  pca_file)
    plot_latents(latent_codes_tsne, tsne_file)
    plot_latents(latent_codes_umap, umap_file)



def write_pdb_with_new_coords(
    pdb_in: str,
    coords,               # (N,3) numpy array or torch tensor
    pdb_out: str,
    include_hetatm: bool = False):
    
    if hasattr(coords, "detach"):  # torch tensor
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=float)

    coord_i = 0
    out_lines = []

    with open(pdb_in, "r") as f:
        for line in f:
            record = line[:6].strip()

            is_atom = (record == "ATOM")
            is_het  = (record == "HETATM")

            if is_atom or (include_hetatm and is_het):
                if coord_i >= len(coords):
                    raise ValueError(
                        f"Not enough coordinates: need >={coord_i+1}, have {len(coords)}")

                x, y, z = coords[coord_i]
                coord_i += 1

                # Keep everything else the same, only overwrite x/y/z with PDB formatting
                new_line = (
                    line[:30]
                    + f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    + line[54:])
                out_lines.append(new_line)
            else:
                out_lines.append(line)

    if coord_i != len(coords):
        print(f"Warning: {len(coords)} coords provided, but only used {coord_i} (ATOM/HETATM lines).")

    with open(pdb_out, "w") as f:
        f.writelines(out_lines)

    print(f"Wrote {pdb_out} with {coord_i} updated atoms.")
