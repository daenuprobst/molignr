import numpy as np
import os
import torch
import time
import ot

from torch_geometric.utils import to_dense_adj

from pathlib import Path

# --- Model ---
from models.model_siren import cIGNR
from data_ import get_dataset
from utils import arg_parse, set_hyperparameters, visualize_latents
from metrics import lddt, tm_score
from models.siren_pytorch import SirenNet

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

                x = data.x.float().to(device)

                edge_index = data.edge_index.to(torch.int64).to(device)
                batch = data.batch.to(torch.int64).to(device)
                C_input = to_dense_adj(edge_index, batch=batch).to(device)

                loss= model(x, edge_index, batch, C_input, args.M)    
                loss_coords_.append(loss.item())
                
                if batch_idx%20==0:
                    print(f"Batch: {batch_idx:03d}, Loss coords : {loss.item():.4f}")

    test_loss = np.round(np.mean(loss_coords_), 3)
    print(f'Epoch TEST: {epoch:03d}, MSE Loss:{test_loss:.4f}')

    return test_loss


def train(args, train_loader, model, test_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_list_batch = []
    best_model = None
    lowest_loss_epoch = float('inf')

    since = time.time()
    N = 2191
    scaler = torch.amp.GradScaler('cuda') 


    for epoch in range(args.n_epoch):
        start_epoch = time.time()
        model.train()
        
        for batch_idx, data in enumerate(train_loader):
            with torch.amp.autocast('cuda'):

                x = data.x.float().to(device)

                edge_index = data.edge_index.to(torch.int64).to(device)
                batch = data.batch.to(torch.int64).to(device)
                C_input = to_dense_adj(edge_index, batch=batch).to(device)

                loss= model(x, edge_index, batch, C_input, args.M)        
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_list_batch.append(loss.item())

            if args.save_output:
                if batch_idx%200==0 or batch_idx==750:
                    print(f'Epoch: {epoch:03d}, Batch: {batch_idx:03d}, Loss  : {loss.item():.4f}')

            del x,  batch
        
        print(f"Epoch time : {time.time() - start_epoch}")
        print(f"Epoch loss : {np.round(np.mean(loss_list_batch), 3)}")

        if loss.item() < lowest_loss_epoch:
            lowest_loss_epoch = np.mean(loss_list_batch)
            best_model=copy.deepcopy(model)
            optimizer_best = copy.deepcopy(optimizer)
            scaler_best = copy.deepcopy(scaler)
            

        if epoch%10==0:
            if best_model is not None and args.save_output:
                ppath = os.getcwd() + '/Results'
                saved_path = Path(f"{ppath}/checkpoints/{prog_args.repr}/lr_{prog_args.lr}/"+ f'siren_{args.gnn_type}_epoch_{epoch}_bs_{args.batch_size}_dim_{args.latent_dim}_knn_{args.knn}_lr_{prog_args.lr}_best_model.pt')
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
                saved_path = Path(f"{ppath}/checkpoints/{prog_args.repr}/lr_{prog_args.lr}/"+ f'siren_{args.gnn_type}_epoch_{epoch}_bs_{args.batch_size}_dim_{args.latent_dim}_knn_{args.knn}_lr_{prog_args.lr}.pt')
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
    # prog_args.gnn_type = 'chebnet'
    prog_args.latent_dim = 8

    prog_args.gnn_type= "gin"

    prog_args.dataset = "full_data_knn_4" 
    prog_args.n_epoch = 150
    prog_args.emb_dim = 2
    prog_args.batch_size = 16
    prog_args.gnn_num_layer = 3
    prog_args.gnn_layers = [3, 8, 8, prog_args.latent_dim]
    prog_args.mlp_dim_hidden = '16,12,8'   
    prog_args.flag_emb = 0
    prog_args.knn = 4
    prog_args.lr = 0.001

    prog_args.mlp_dim_hidden = [int(x) for x in prog_args.mlp_dim_hidden.split(',')]
    prog_args.mlp_num_layer = len(prog_args.mlp_dim_hidden)
    print(f"prog_args: {prog_args}")
    N = 2191

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

    
    model = model.to(device)
    # Load Specific Dataset
    train_loader, test_loader, n_card = get_dataset(prog_args, shuffle=True)

    print(f"Train set length : {len(train_loader.dataset)}")

    saved_path = train(prog_args, train_loader, model, test_loader)
    print(f"Saved path : {saved_path}")

