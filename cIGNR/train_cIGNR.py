import numpy as np
import os
import torch
import time
from torch_geometric.utils import to_dense_adj
from pathlib import Path

# --- Model ---
from models.model_cIGNR import cIGNR
from models.siren_pytorch import SirenNet
from data_ import get_dataset
from utils import *
from evaluation.metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE : {device}")

import random
seed = 42

import copy

# Set seed for PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os, psutil, torch
process = psutil.Process(os.getpid())


def report_memory():
    ram_gb = process.memory_info().rss / 1e9
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"[Memory] RAM: {ram_gb:.2f} GB   GPU: {gpu_mem:.2f} GB")



def test(args, test_loader, model, epoch):

    model.eval()
    loss_list = []
    loss_coords_ = []
    loss_gw = []
        
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            with torch.amp.autocast('cuda'):
                x = data.x.float().to(args.device)

                edge_index = data.edge_index.to(torch.int64).to(args.device)
                batch = data.batch.to(torch.int64).to(args.device)
                C_input = to_dense_adj(edge_index, batch=batch).to(device)

                loss, coords_pred = model(x, edge_index, batch, C_input, args.M)        
                loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
                
                total_loss = loss + loss_coords
                if batch_idx%10==0:
                    print(f"Batch: {batch_idx:03d}, Loss GW : {loss.item():.4f}, Loss coords : {loss_coords.item():.4f}, Total loss : {total_loss.item():.4f}")

                loss_list.append(total_loss.item())
                loss_coords_.append(loss_coords.item())
                loss_gw.append(loss.item())
            
        loss_batch = np.round(np.mean(loss_list), 3)
        loss_batch_coords = np.round(np.mean(loss_coords_), 3)
        loss_gw_mean = np.round(np.mean(loss_gw), 3)

    return  loss_batch_coords, loss_gw_mean, loss_batch



def train(args, train_loader, model, test_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_size = args.batch_size

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[201, 250], gamma=0.1)
    loss_list_batch = []
    best_model = None
    lowest_loss_epoch = float('inf')

    since = time.time()
    scaler = torch.amp.GradScaler('cuda')  

    for epoch in range(args.n_epoch):
        start_epoch = time.time()
        model.train()
        
        loss_epoch = []
        mse_epoch = []
        gw_epoch = []

        lddt_epoch = []
        tm_score_epoch = []

        for batch_idx, data in enumerate(train_loader):
            with torch.amp.autocast('cuda'):
                x = data.x.float().to(args.device)
                edge_index = data.edge_index.to(torch.int64).to(args.device)
                edge_index = edge_index.long()

                batch = data.batch.to(torch.int64).to(args.device)
                C_input = to_dense_adj(edge_index, batch=batch)
                
                loss, coords_pred = model(x, edge_index, batch, C_input, args.M)     
                loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
                total_loss = loss + loss_coords

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_epoch.append(total_loss.item())
            mse_epoch.append(loss_coords.item())
            gw_epoch.append(loss.item())

            if args.save_output:
                if batch_idx%100==0 or batch_idx==750:
                    print(f'Epoch: {epoch:03d}, Batch: {batch_idx:03d}, Loss:{total_loss.item():.4f}, Loss GW : {loss.item():.4f}, Loss coords : {loss_coords.item():.4f}')

                    #### LDDT score 
                    if args.dataset == "full_data_knn_4":
                        N = 2191
                    else:
                        N = 273

                    lddt_ = lddt(coords_pred.view(batch_size, N, 3), x.view(batch_size, N, 3))
                    tm_score_ = tm_score(x, coords_pred)
                    lddt_epoch.append(lddt_)
                    tm_score_epoch.append(tm_score_)

                    del lddt_, tm_score_

            del loss, total_loss, loss_coords, x, edge_index, batch, C_input
        
        lr_scheduler.step()
    
        print(f"Epoch time : {time.time() - start_epoch}")
        loss_epoch_mean = np.mean(loss_epoch) # this is not loss on test set, this is average training loss across batches
        mse_epoch_mean = np.mean(mse_epoch)
        gw_epoch_mean = np.mean(gw_epoch)
        loss_list_batch.append(np.round(loss_epoch_mean,3))

        lddt_epoch_mean = np.mean(lddt_epoch)
        tm_score_epoch_mean = np.mean(tm_score_epoch)

        print(f'Epoch: {epoch:03d}, total Loss:{loss_epoch_mean:.4f}, GW Loss:{gw_epoch_mean:.4f}, MSE Loss:{mse_epoch_mean:.4f}')
        print(f"Epoch: {epoch:03d}, lDDT: {lddt_epoch_mean}, tm-score: {tm_score_epoch_mean}")
        print()

        if loss_epoch_mean < lowest_loss_epoch:
            lowest_loss_epoch = loss_epoch_mean
            best_model=copy.deepcopy(model)

        # save trained model and loss here
        if epoch%3==0:
            if args.save_output:
                ppath = os.getcwd() + '/Results'
                saved_path = Path(f"{ppath}/checkpoints/{prog_args.repr}/lr_{prog_args.lr}/"+ f'epoch_{epoch}_dim_{args.latent_dim}_knn_{args.knn}_lr_{prog_args.lr}_best_model.pt')
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                
                saved_path = Path(f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/FINAL_RESULT/checkpoints/lr_{prog_args.lr}/"+ f'{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}_epoch_{epoch}_knn_{args.knn}_lr_{prog_args.lr}.pt')
                
                torch.save({'epoch': epoch, 
                    'batch': batch_idx, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    'configs': args},
                    saved_path)

        ########### ... ###############
        test_coords, test_gw, test_loss = test(args, test_loader, model, epoch_)            

        print(f'Epoch TEST: {epoch:03d}, total Loss:{test_loss:.4f}, GW Loss:{test_gw:.4f}, MSE Loss:{test_coords:.4f}')
        
        if epoch%10==0:
            if best_model is not None and args.save_output:
                saved_path = Path(f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/FINAL_RESULT/checkpoints_node_repr/lr_{prog_args.lr}/"+ f'{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}_epoch_{epoch}_knn_{args.knn}_lr_{prog_args.lr}_best_model.pt')
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'epoch': epoch, 
                    'batch': batch_idx, 
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    'configs': args},
                    saved_path)
   
    finish = time.time()
    print('time used:'+str(finish-since))
    print('loss on per epoch:')
    print(loss_list_batch)

    print('Finished Training')

    return saved_path





def main(prog_args):

    prog_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(prog_args.device)

    ppath = os.getcwd()

    prog_args.save_path = ppath+'/Results/'
    if not os.path.exists(prog_args.save_path):
        os.makedirs(prog_args.save_path)
    print('saving path is:'+prog_args.save_path)

    # Load Specific Dataset
    train_loader, test_loader, n_card = get_dataset(prog_args)

    print(f"Train set length : {len(train_loader.dataset)}")

    prog_args.mlp_dim_hidden = [int(x) for x in prog_args.mlp_dim_hidden.split(',')]
    prog_args.mlp_num_layer = len(prog_args.mlp_dim_hidden)

    snet_adj = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = prog_args.mlp_dim_hidden,
        dim_out = 1, # output graphon (edge) probability 
        num_layers = prog_args.mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = prog_args.mlp_act )
    
    N = 2191    
    
    model = cIGNR(net_adj=snet_adj, input_card=n_card, emb_dim = prog_args.emb_dim, latent_dim = prog_args.latent_dim, 
                  num_layer=prog_args.gnn_num_layer, 
                  gnn_layers= prog_args.gnn_layers,
                  device=prog_args.device, flag_emb=prog_args.flag_emb, gnn_type = prog_args.gnn_type, N = N,
                  representation = prog_args.repr)
    
    model = model.to(torch.device(prog_args.device))
    saved_path = train(prog_args, train_loader, model, test_loader)

    return saved_path, train_loader, test_loader, model


if __name__ == '__main__':

    gnn_types = ['chebnet', 'gin']
    prog_args = arg_parse()
    prog_args.gnn_type = 'chebnet'
    prog_args.latent_dim = 8

    prog_args.dataset = "full_data_knn_4" 
    prog_args.n_epoch = 150
    prog_args.emb_dim = 2
    prog_args.batch_size = 16
    prog_args.gnn_num_layer = 3
    prog_args.gnn_layers = [3, 8, 8, prog_args.latent_dim]
    prog_args.mlp_dim_hidden = '16,12,8'   
    prog_args.flag_emb = 0
    prog_args.knn = 4
    prog_args.lr = 0.01
    print()
    print(f"Dataset : {prog_args.dataset}")
    print(f"Latent dim : {prog_args.latent_dim}")
    print(f"prog_args : {prog_args}")
    print()

    saved_path, train_loader, test_loader, model = main(prog_args)
    print(f"Saved path : {saved_path}")

