import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

from pathlib import Path
import torch

from models.model_cIGNR import *
from data_ import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

from utils import *
import numpy as np
import MDAnalysis as mda
from metrics import *
from torch_geometric.utils import to_dense_adj
from a100 import a100_plots

def test(args, test_loader, model, epoch):

    model.eval()
    loss_list = []
    loss_coords_ = []

    lddt_list = []
    tm_score_list = []
    total_loss_list = []
        
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            # print(f"Data.shape : {data.x.shape}")
            x = data.x.float().to(args.device)

            edge_index = data.edge_index.to(torch.int64).to(args.device)
            batch = data.batch.to(torch.int64).to(args.device)
            C_input = to_dense_adj(edge_index, batch=batch)

            loss, coords_pred = model(x, edge_index, batch, C_input, args.M)        
            loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))

            print(f"Loss GW : {loss.item():.4f}, Loss coords : {loss_coords.item():.4f}")
            total_loss = loss + loss_coords

            loss_list.append(loss.item())
            loss_coords_.append(loss_coords.item())
            total_loss_list.append(total_loss.item())
            
            N = x.shape[0]//args.batch_size
            lddt_ = lddt(coords_pred.view(args.batch_size, N, 3), x.view(args.batch_size, N, 3))
            tm_score_ = tm_score(x, coords_pred)

            lddt_list.append(lddt_)
            tm_score_list.append(tm_score_)


            print(f"Training LDDT score {lddt_}")
            print(f"Training TM-Score {tm_score_}")

        loss_batch = np.mean(loss_list) # this is not loss on test set, this is average training loss across batches
        loss_batch_coords = np.mean(loss_coords_)
        total_loss_epoch = np.mean(total_loss_list)

        lddt_epoch = np.mean(lddt_list)
        tm_score_epoch = np.mean(tm_score_list)

        print()
        print(f"LDDT in test set : {np.round(lddt_epoch,3)}")
        print(f"TM-Score in test set : {np.round(tm_score_epoch,3)}")
        print()
        print(f"GW loss in test set : {np.round(loss_batch,3)}")
        print(f"Coords loss in test set coords: {np.round(loss_batch_coords,3)}")
        print(f"Total loss in test set coords: {np.round(total_loss_epoch,3)}")
        print()
        # z = get_emb(args,model,test_loader)

    return np.round(loss_batch, 3)



if __name__=="__main__":
    path = "/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/FINAL_RESULT/checkpoints_node_repr/lr_0.01/full_data_knn_4_chebnet_dim_8_epoch_15_knn_4_lr_0.01.pt"

    print(f"path : {path}")
    model, prog_args = load_model(path)
    train_loader, test_loader, _ = get_dataset(prog_args, shuffle = False)
    print("running ....")

    visualize_latents(args, model, train_loader, epoch = None)
    print("visualization done....")
    # test(args, train_loader, model)
 
    #ramachandran_full_data(train_loader, args, plot_name = "full_data_histogram", N =2191)
    # ramachandran_reconstruction(model, train_loader, args, plot_name = f"R_no_siren_rc_{args.repr}")
    ramachandran_interpolation(model, train_loader, args, plot_name = f"R_no_siren_interpolation_{args.repr}", N=2191)

    a100_plots(model, train_loader, prog_args, gpcrdb_pdb = "heavy_chain_GPCRDB.pdb", plot_name = "hist_a100")

    test(prog_args, test_loader, model, None)






