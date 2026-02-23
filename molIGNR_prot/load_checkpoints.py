import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

import torch

from utils import write_pdb_with_new_coords
from data_ import get_dataset
from utils import visualize_latents
import numpy as np
from evaluation.metrics import lddt, tm_score
from evaluation.metrics import ramachandran_full_data, ramachandran_reconstruction, ramachandran_interpolation, ramachandran_interpolation2
from torch_geometric.utils import to_dense_adj
from models.siren_pytorch import SirenNet

from models.model_cIGNR import cIGNR

import matplotlib.pyplot as plt
import os

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)  # or 'cuda' if using GPU
    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch']
    prog_args = checkpoint['configs']

    print(f"batch_Size : {prog_args.batch_size}")
    print(f"Dataset : {prog_args.dataset}")

    prog_args.device = device

    snet_adj = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = prog_args.mlp_dim_hidden,
        dim_out = 1, # output graphon (edge) probability 
        num_layers = prog_args.mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = prog_args.mlp_act)
    n_card = 3
    
    model = cIGNR(net_adj=snet_adj, input_card=n_card, emb_dim = prog_args.emb_dim, latent_dim = prog_args.latent_dim, 
                  num_layer=prog_args.gnn_num_layer, 
                  gnn_layers= prog_args.gnn_layers,
                  device=prog_args.device, flag_emb=prog_args.flag_emb, gnn_type = prog_args.gnn_type, N =2191, 
                  representation= prog_args.repr)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)


    return model, prog_args, epoch, checkpoint["scaler"]

                   
 
def test(args, test_loader, model):
    model.eval()
    loss_list = []
    loss_coords_ = []

    lddt_list = []
    tm_score_list = []
    total_loss_list = []

    mse_bb = []
    lddt_list_bb_idx = []
    tm_score_list_bb_idx = []

    current_dir = os.getcwd()
    bb_idx = torch.load(f"{current_dir}/bb_idx.pt")

    N = 2191
    pred_coords = []

    final_loss = []
        
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for batch_idx, data in enumerate(test_loader):

                x = data.x.float().to(args.device)

                edge_index = data.edge_index.to(torch.int64).to(args.device)
                batch = data.batch.to(torch.int64).to(args.device)
                C_input = to_dense_adj(edge_index, batch=batch)

                loss, coords_pred = model(x, edge_index, batch, C_input, args.M)  

                pred_coords.extend(coords_pred.detach().cpu().tolist())

                loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
                coord_bb_pred = coords_pred.view(args.batch_size, N, 3)[:, bb_idx,:].view(-1,3)
                coord_bb_true = x.view(args.batch_size, N, 3)[:, bb_idx,:].view(-1,3)
                loss_coords_bb = torch.nn.functional.mse_loss(coord_bb_pred, coord_bb_true.to(device))
                mse_bb.append(loss_coords_bb.item())

                total_loss = loss + loss_coords

                if batch_idx==0:
                    write_pdb_with_new_coords("/home/binal1/Graphons/cIGNR/evaluation/heavy_chain.pdb",
                                                x[:N][bb_idx], f"bb_{args.gnn_type}_original_data.pdb")
                    
                    write_pdb_with_new_coords("/home/binal1/Graphons/cIGNR/evaluation/heavy_chain.pdb",
                                                coords_pred[:N][bb_idx], f"bb_{args.gnn_type}_predicted.pdb")

                loss_list.append(loss.item())
                loss_coords_.append(loss_coords.item())
                total_loss_list.append(total_loss.item())
                
                # N = x.shape[0]//args.batch_size
                lddt_ = lddt(coords_pred.view(args.batch_size, N, 3), x.view(args.batch_size, N, 3))
                tm_score_ = tm_score(x, coords_pred)

                lddt_bb = lddt(coord_bb_pred.view(args.batch_size, -1, 3), coord_bb_true.view(args.batch_size, -1, 3))
                tm_bb = tm_score(coord_bb_true, coord_bb_pred)

                lddt_list_bb_idx.append(lddt_bb)
                tm_score_list_bb_idx.append(tm_bb)

                lddt_list.append(lddt_)
                tm_score_list.append(tm_score_)


                if batch_idx %100 ==0:
                    print(f"Loss GW : {loss.item():.4f}, Loss coords : {loss_coords.item():.4f}")
                    print(f"Loss coords BB : {loss_coords_bb.item():.4f}")
                    print(f"LDDT score {lddt_}")
                    print(f"TM-Score {tm_score_}")
                    print(f"BB LDDT score {lddt_bb}")
                    print(f"BB TM-Score {tm_bb}")
                    print(f"MSE after alignment : {mse_}")

        loss_batch = np.mean(loss_list) 
        loss_batch_coords = np.mean(loss_coords_)
        total_loss_epoch = np.mean(total_loss_list)

        loss_batch_coords_bb = np.mean(mse_bb)

        lddt_epoch = np.mean(lddt_list)
        tm_score_epoch = np.mean(tm_score_list)

        print()
        print(f"After alignment MSE : {np.mean(final_loss)}")

        print()
        print(f"LDDT: {np.round(lddt_epoch,3)}")
        print(f"TM-Score: {np.round(tm_score_epoch,3)}")
        print()
        print(f"GW loss: {np.round(loss_batch,3)}")
        print(f"MSE loss: {np.round(loss_batch_coords,3)}")
        print(f"Total loss: {np.round(total_loss_epoch,3)}")
        print()
        print(f"BB MSE set coords: {np.round(loss_batch_coords_bb,3)}")
        print(f"BB LDDT: {np.round(np.mean(lddt_list_bb_idx),3)}")
        print(f"BB TM-Score: {np.round(np.mean(tm_score_list_bb_idx),3)}")
        print()
        # z = get_emb(args,model,test_loader)

    return np.round(loss_batch, 3)



if __name__=="__main__":
    
    path = os.getcwd() + "..." # needs to be replaced with the checkpoint directory

    print(f"path : {path}")
    model, prog_args, epoch, _ = load_model(path)
    train_loader, test_loader, _ = get_dataset(prog_args, shuffle = False, num_samples=5000)
    print(f"running {prog_args.gnn_type}")
    print()
    print(f"prog_args: {prog_args}")

    # visualize_latents(prog_args, model, train_loader, epoch = f"{epoch}")
    # print("visualization done....")
    # test(prog_args, train_loader, model)

    # ramachandran_full_data(train_loader, prog_args, plot_name = "full_data_histogram", N =2191)
    # ramachandran_reconstruction(model, train_loader, prog_args, plot_name = f"R_{prog_args.gnn_type}_{epoch}_reconstruction")
    # ramachandran_interpolation(model, train_loader, prog_args, plot_name = f"R_{prog_args.gnn_type}_{epoch}_interpolation", N=2191)
    # sigma = 0.3
    # ramachandran_interpolation2(model, train_loader, prog_args, plot_name = f"R_{prog_args.gnn_type}_{epoch}_noise_interpolation_{sigma}", sigma = sigma, N=2191)
    



