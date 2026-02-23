import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device : {device}")

from pathlib import Path
import torch

import sys

from models.model_siren import *
from data_ import *
from metrics import *

from utils import *
import numpy as np


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
    path = "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.01/gin_epoch_120_bs_10_dim_8_knn_4_lr_0.01_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.01/gin_epoch_410_dim_8_knn_4_lr_0.01_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.01/gin_epoch_40_dim_8_knn_4_lr_0.01_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/node/lr_0.01/epoch_60_dim_8_knn_4_lr_0.01_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/node/lr_0.01/epoch_40_dim_8_knn_4_lr_0.01_best_model.pt"
    # "/home/binal1/Graphons/cIGNR/_without_SIREN/Results/checkpoints/graph/lr_0.01/epoch_40_dim_8_knn_4_lr_0.01_best_model.pt"
    # /home/binal1/Graphons/cIGNR_final/_without_SIREN/Results/checkpoints/graph/lr_0.01/epoch_0_dim_8_knn_4_lr_0.01_best_model.pt"
    model, args, epoch_ = load_model(path)

    train_loader, test_loader, _ = get_dataset(args, shuffle=False)

    print("running ....")

    visualize_latents(args, model, train_loader, epoch = epoch_)
    print("visualization done....")
    # test(args, train_loader, model, epoch_)
 
    #ramachandran_full_data(train_loader, args, plot_name = "full_data_histogram", N =2191)
    #ramachandran_reconstruction(model, train_loader, args, plot_name = f"R_no_siren_rc_{args.repr}")
    #ramachandran_interpolation(model, train_loader, args, plot_name = f"R_no_siren_interpolation_{args.repr}", N=2191)

    # a100_plots(model, train_loader, args, gpcrdb_pdb = "heavy_chain_GPCRDB.pdb", plot_name = "hist_a100")














