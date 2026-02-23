import numpy as np
import os
import torch
import time
import ot

from torch_geometric.utils import to_dense_adj

from pathlib import Path

# --- Model ---
from models.model import cIGNR
# from visualize_latent import *
from data_ import get_dataset
from utils import *
from metrics import *

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
    loss_coords_ = []
        
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            with torch.amp.autocast('cuda'):
                x = data.x.float().to(args.device)
                edge_index = data.edge_index.to(torch.int64).to(args.device)
                batch = data.batch.to(torch.int64).to(args.device)

                coords_pred = model(x, edge_index, batch)        
                loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
                loss_coords_.append(loss_coords.item())
                
                if batch_idx%20==0:
                    print(f"Batch: {batch_idx:03d}, Loss coords : {loss_coords.item():.4f}")

    test_loss = np.round(np.mean(loss_coords_), 3)
    print(f'Epoch TEST: {epoch:03d}, MSE Loss:{test_loss:.4f}')

    return test_loss


def train(args, train_loader, model, test_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_size = args.batch_size

    loss_list_batch = []
    best_model = None
    lowest_loss_epoch = float('inf')

    since = time.time()
    N = 2191
    scaler = torch.amp.GradScaler('cuda') 

    for epoch in range(args.n_epoch):
        start_epoch = time.time()
        model.train()
        
        mse_epoch = []
        lddt_epoch = []
        tm_score_epoch = []

        for batch_idx, data in enumerate(train_loader):
            with torch.amp.autocast('cuda'):
                x = data.x.float().to(args.device)
                edge_index = data.edge_index.to(torch.int64).to(args.device)
                edge_index = edge_index.long()

                batch = data.batch.to(torch.int64).to(args.device)
                coords_pred = model(x, edge_index, batch)     
                loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
    
            scaler.scale(loss_coords).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            mse_epoch.append(loss_coords.item())

            if args.save_output:
                if batch_idx%200==0 or batch_idx==750:
                    print(f'Epoch: {epoch:03d}, Batch: {batch_idx:03d}, Loss coords : {loss_coords.item():.4f}')

                    #### LDDT score 
                    lddt_ = lddt(coords_pred.view(batch_size, N, 3), x.view(batch_size, N, 3))
                    tm_score_ = tm_score(x, coords_pred)
                    lddt_epoch.append(lddt_)
                    tm_score_epoch.append(tm_score_)

                    del lddt_, tm_score_

            del loss_coords, x,  batch
        
        print(f"Epoch time : {time.time() - start_epoch}")
        mse_epoch_mean = np.mean(mse_epoch)

        lddt_epoch_mean = np.round(np.mean(lddt_epoch),3)
        tm_score_epoch_mean = np.mean(tm_score_epoch)

        print(f"Epoch : {epoch:03d}, lDDT : {lddt_epoch_mean}, TM-score : {tm_score_epoch_mean}")
        print(f'Epoch: {epoch:03d}, MSE Loss:{mse_epoch_mean:.4f}')
        print()

        if mse_epoch_mean < lowest_loss_epoch:
            lowest_loss_epoch = mse_epoch_mean
            best_model=copy.deepcopy(model)
            optimizer_best = copy.deepcopy(optimizer)
            scaler_best = copy.deepcopy(scaler)
            

        if epoch%10==0:
            if best_model is not None and args.save_output:
                ppath = os.getcwd() + '/Results'
                saved_path = Path(f"{ppath}/checkpoints/{prog_args.repr}/lr_{prog_args.lr}/"+ f'{args.gnn_type}_epoch_{epoch}_bs_{args.batch_size}_dim_{args.latent_dim}_knn_{args.knn}_lr_{prog_args.lr}_best_model.pt')
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'epoch': epoch, 
                    'batch': batch_idx, 
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer_best.state_dict(),
                    "scaler": scaler_best.state_dict() if scaler_best is not None else None,
                    'configs': args},
                    saved_path)
                
                ### Visualise the latent space...
                # visualize_latents(prog_args, model, train_loader, epoch = epoch)
            elif args.save_output:
                ppath = os.getcwd() + '/Results'
                saved_path = Path(f"{ppath}/checkpoints/{prog_args.repr}/lr_{prog_args.lr}/"+ f'{args.gnn_type}_epoch_{epoch}_bs_{args.batch_size}_dim_{args.latent_dim}_knn_{args.knn}_lr_{prog_args.lr}.pt')
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'epoch': epoch, 
                    'batch': batch_idx, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    'configs': args},
                    saved_path)

        _ = test(args, test_loader, model, epoch)        
        
    print('time used:'+str(time.time()-since))
    print(f'loss on per epoch: {loss_list_batch}')

    print('Training is done!! ')
    return saved_path





if __name__ == '__main__':

    report_memory()

    prog_args = arg_parse()
    # prog_args = set_hyperparameters(prog_args)    
    prog_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppath = os.getcwd()

    prog_args.save_path=ppath+'/Results/'+prog_args.save_dir+'/'
    if not os.path.exists(prog_args.save_path):
        os.makedirs(prog_args.save_path)
    print('saving path is:'+prog_args.save_path)


    # prog_args.mlp_dim_hidden = [int(x) for x in prog_args.mlp_dim_hidden.split(',')]
    # prog_args.mlp_num_layer = len(prog_args.mlp_dim_hidden)

    prog_args.dataset == "full_data_knn_4"

    prog_args.latent_dim = 8
    prog_args.n_epoch = 4000
    prog_args.gnn_type = "gin"

    prog_args.batch_size = 16
    prog_args.gnn_num_layer = 3
    prog_args.gnn_layers = [3, 8, 8, prog_args.latent_dim]
    prog_args.mlp_dim_hidden = '16,12,8'   

    prog_args.lr = 0.01

    prog_args.knn = 4
    print(prog_args.device)

    print(f"prog_args: {prog_args}")
    N = 2191

    model = cIGNR(input_card=3, emb_dim = prog_args.emb_dim, latent_dim = prog_args.latent_dim, 
                  num_layer=prog_args.gnn_num_layer, 
                  gnn_layers= prog_args.gnn_layers,
                  device=prog_args.device, flag_emb=prog_args.flag_emb, gnn_type = prog_args.gnn_type, N = N)
    
    model = model.to(torch.device(prog_args.device))
    # Load Specific Dataset
    train_loader, test_loader = get_dataset(prog_args, shuffle=True)

    print(f"Train set length : {len(train_loader.dataset)}")

    saved_path = train(prog_args, train_loader, model, test_loader)
    print(f"Saved path : {saved_path}")

