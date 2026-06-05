import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

import torch

from data_ import get_dataset
# from utils import visualize_latents
import numpy as np
from molIGNR_prot.evaluation.metrics import evaluate_backbone, generate_new_samples, generate_alpha_carbons, kabsch_algorithm, lddt, tm_score, compute_tm_score_torch
from molIGNR_prot.evaluation.metrics import ramachandran_interpolation_, ramachandran_linear_interpolation_
from torch_geometric.utils import to_dense_adj
from models.siren_pytorch import SirenNet

from models.model import cIGNR

import matplotlib.pyplot as plt
import os

import time
import MDAnalysis as mda

from utils import visualize_latents
from molIGNR_prot.evaluation.metrics import compute_mse, compute_lddt, compute_tm_score
import torch.nn.functional as F


def load_model(checkpoint_path, dataset = None):
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

    if dataset is not None:
        prog_args.dataset = dataset


    if "bb" in prog_args.dataset:
        N = 1091
    elif "full" in prog_args.dataset:
        N = 2191    
    elif prog_args.dataset == "dyn_95_heavy":
        N = 4250
    elif prog_args.dataset == "dyn_1119_heavy":
        N = 4539
    else:
        print("It should be either backbone or full atom structure... ")

    print(f"N : {N}")
    
    model = cIGNR(net_adj=snet_adj, latent_dim = prog_args.latent_dim, 
                  num_layer=prog_args.gnn_num_layer, 
                  gnn_layers= prog_args.gnn_layers,
                  device=prog_args.device, gnn_type = prog_args.gnn_type, N = N)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    snet_adj.to(device)


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


    if "bb" in prog_args.dataset:
        N = 1091
    elif "full" in prog_args.dataset:
        N = 2191    # total number of atoms in the protein
    elif prog_args.dataset == "dyn_709":
        N = 4782
    elif prog_args.dataset == "dyn_1119_knn4":
        N = 4940
    elif prog_args.dataset == "dyn_1119_knn10":
        N = 4940
    elif prog_args.dataset == "dyn_95_knn4":
        N = 4618
    elif prog_args.dataset == "dyn_95_knn10":
        N = 4618
    elif prog_args.dataset == "dyn_95_backbone_knn4":
        N = 1091
    else:
        print("It should be either backbone or full atom structure... ")
    
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

                # pred_coords.extend(coords_pred.detach().cpu().tolist())

                loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
                coord_bb_pred = coords_pred.view(args.batch_size, N, 3)[:, bb_idx,:].view(-1,3)
                coord_bb_true = x.view(args.batch_size, N, 3)[:, bb_idx,:].view(-1,3)
                loss_coords_bb = torch.nn.functional.mse_loss(coord_bb_pred, coord_bb_true.to(device))
                mse_bb.append(loss_coords_bb.item())

                total_loss = loss + loss_coords
                """
                if batch_idx==0:
                    write_pdb_with_new_coords("/home/binal1/Graphons/cIGNR/evaluation/heavy_chain.pdb",
                                                x[:N][bb_idx], f"bb_{args.gnn_type}_original_data.pdb")
                    
                    write_pdb_with_new_coords("/home/binal1/Graphons/cIGNR/evaluation/heavy_chain.pdb",
                                                coords_pred[:N][bb_idx], f"bb_{args.gnn_type}_predicted.pdb")
                """
                loss_list.append(loss.item())
                loss_coords_.append(loss_coords.item())
                total_loss_list.append(total_loss.item())
                
                # N = x.shape[0]//args.batch_size
                lddt_ = lddt(coords_pred.view(args.batch_size, N, 3), x.view(args.batch_size, N, 3))
                tm_score_ = compute_tm_score_torch(x, coords_pred)

                lddt_bb = lddt(coord_bb_pred.view(args.batch_size, -1, 3), coord_bb_true.view(args.batch_size, -1, 3))
                tm_bb = compute_tm_score_torch(coord_bb_true, coord_bb_pred)

                lddt_list_bb_idx.append(lddt_bb)
                tm_score_list_bb_idx.append(tm_bb)

                lddt_list.append(lddt_)
                tm_score_list.append(tm_score_)

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






def test_(args, dataloader, model, pdb_path = "heavy_chain.pdb"):
    model.eval()
    metrics = {"lddt": [], "tm_score": [], "mse": [],
               "mse_bb": [], "lddt_bb": [], "tm_score_bb": [],
               "mse_sc": [], "tm_score_correct": []}
    
    # bb_idx = torch.load("/home/binal1/LD-FPG/blind/data/processed/atom_indices.pt", weights_only=False)["bb_indices"]                
    # sc_idx = torch.load("/home/binal1/LD-FPG/blind/data/processed/atom_indices.pt", weights_only=False)["sc_indices"]
    # u  = mda.Universe("data/heavy_chain.pdb")
    # ca = u.select_atoms("protein and name CA")
    # ca_indices = ca.indices 

    u  = mda.Universe(pdb_path)
    protein = u.select_atoms("protein")
    backbone = u.select_atoms("protein and backbone")
    ca = u.select_atoms("protein and name CA")
    bb_idx = backbone.indices
    sc_idx = u.select_atoms("protein and not backbone").indices
    ca_indices = ca.indices

    print(f"protein atoms:   {len(protein.indices)}")
    print(f"CA atoms:        {len(ca_indices)}")
    print(f"Backbone atoms:  {len(bb_idx)}")
    print(f"Sidechain atoms: {len(sc_idx)}")
    print(f"Total:           {len(bb_idx) + len(sc_idx)}")
    print(f"Secretin in play.... :)) ")


    
    for batch_idx, data in enumerate(dataloader):

        x = data.x.float().to(args.device)

        edge_index = data.edge_index.to(torch.int64).to(args.device)
        batch = data.batch.to(torch.int64).to(args.device)
        C_input = to_dense_adj(edge_index, batch=batch)

        with torch.no_grad():
            loss, coords_pred = model(x, edge_index, batch, C_input, args.M)  
            _, coords_pred = kabsch_algorithm(x.view(args.batch_size, -1, 3), coords_pred.view(args.batch_size, -1, 3))


        #########################################################
        """
        if batch_idx == 0:
        #    u = mda.Universe("heavy_chain.pdb")
            protein = u.select_atoms("protein and not type H")
            protein.positions = coords_pred[0].cpu()
            protein.write(f"_{prog_args.dataset}_predicted_conformation.pdb")
            print(f"DONE")
        """
        #########################################################




        N = x.shape[0]//args.batch_size
        lddt_ = lddt(coords_pred, x.view(args.batch_size, N, 3))
        # tm_score_ = compute_tm_score_torch(x, coords_pred.view(-1, 3))
        mse_ = torch.nn.functional.mse_loss(coords_pred.view(-1, 3), x.to(device))

        for i, sample in enumerate(x.view(args.batch_size, N, 3)):
            #print(f"sample shape : {sample.shape}")
            #print(f"coords pred.shape : {coords_pred.shape}")

            tm_score_ = compute_tm_score_torch(sample[ca_indices, :], coords_pred[i, ca_indices, :])
            metrics["tm_score_correct"].append(tm_score_)


        #### backbone
        mask = torch.zeros(coords_pred.shape[1], dtype=torch.bool)
        mask[bb_idx] = True
        coord_non_bb = coords_pred[:, ~mask, :].view(-1,3)
        x_non_bb = x.view(args.batch_size, N, 3)[:, ~mask, :].view(-1, 3)

        coord_bb_pred = coords_pred[:, bb_idx,:].view(-1,3)
        coord_bb_true = x.view(args.batch_size, N, 3)[:, bb_idx,:].view(-1,3)

        mse_bb = torch.nn.functional.mse_loss(coord_bb_pred, coord_bb_true.to(device))

        mse_sc = torch.nn.functional.mse_loss(coord_non_bb, x_non_bb.to(device))
        lddt_bb = lddt(coord_bb_pred.view(args.batch_size, -1, 3), coord_bb_true.view(args.batch_size, -1, 3))
        tm_bb = tm_score(coord_bb_true, coord_bb_pred)

        metrics["lddt"].append(lddt_)
        metrics["tm_score"].append(tm_score_)
        metrics["mse"].append(mse_)

        metrics["mse_bb"].append(mse_bb)
        metrics["lddt_bb"].append(lddt_bb)
        metrics["tm_score_bb"].append(tm_bb)    
        metrics["mse_sc"].append(mse_sc)

    # Print average metrics for the epoch
    print(f"----------------------------------------------------------------------")
    print(f"Average LDDT: {torch.tensor(metrics['lddt']).mean():.3f}")
    print(f"Average TM-Score: {torch.tensor(metrics['tm_score']).mean():.3f}")
    print(f"Average MSE: {torch.tensor(metrics['mse']).mean():.3f}")
    print(f"Average BB MSE: {torch.tensor(metrics['mse_bb']).mean():.3f}")
    print(f"Average SC MSE: {torch.tensor(metrics['mse_sc']).mean():.3f}")
    print(f"Average BB LDDT: {torch.tensor(metrics['lddt_bb']).mean():.3f}")
    print(f"Average BB TM-Score: {torch.tensor(metrics['tm_score_bb']).mean():.3f}")
    print(f"Average Corrected TM-Score: {torch.tensor(metrics['tm_score_correct']).mean():.3f}")
    print(f"----------------------------------------------------------------------")
    return metrics  






@torch.no_grad()
def test_gpcrmd(args, dataloader, model, pdb_path = "heavy_chain.pdb"):
    model.eval()
    metrics = {"lddt": [], "tm_score": [], "mse": [],
               "mse_bb": [], "lddt_bb": [], "tm_score_bb": [],
               "mse_sc": [], "tm_score_correct": []}
    
    # bb_idx = torch.load("/home/binal1/LD-FPG/blind/data/processed/atom_indices.pt", weights_only=False)["bb_indices"]                
    # sc_idx = torch.load("/home/binal1/LD-FPG/blind/data/processed/atom_indices.pt", weights_only=False)["sc_indices"]
    # u  = mda.Universe("data/heavy_chain.pdb")
    # ca = u.select_atoms("protein and name CA")
    # ca_indices = ca.indices 

    u  = mda.Universe(pdb_path)
    protein = u.select_atoms("protein")
    backbone = u.select_atoms("protein and backbone")
    ca = u.select_atoms("protein and name CA")
    bb_idx = backbone.indices
    sc_idx = u.select_atoms("protein and not backbone").indices
    ca_indices = ca.indices

    print(f"protein atoms:   {len(protein.indices)}")
    print(f"CA atoms:        {len(ca_indices)}")
    print(f"Backbone atoms:  {len(bb_idx)}")
    print(f"Sidechain atoms: {len(sc_idx)}")
    print(f"Total:           {len(bb_idx) + len(sc_idx)}")
    print(f"Secretin in play.... :)) ")
    ref_seq = [r.resname for r in u.select_atoms("protein and backbone").residues]
   
    for batch_idx, data in enumerate(dataloader):

        x = data.x.float().to(args.device)

        edge_index = data.edge_index.to(torch.int64).to(args.device)
        batch = data.batch.to(torch.int64).to(args.device)
        C_input = to_dense_adj(edge_index, batch=batch)

        with torch.no_grad():
            loss, coords_pred = model(x, edge_index, batch, C_input, args.M)  
            # _, coords_pred = kabsch_algorithm(x.view(args.batch_size, -1, 3), coords_pred.view(args.batch_size, -1, 3))

        #########################################################

        if batch_idx == 0:
        #   u = mda.Universe("heavy_chain.pdb")
            protein = u.select_atoms("protein and not type H")
            protein.positions = coords_pred[0].cpu()
            protein.write(f"_{prog_args.dataset}_predicted_conformation.pdb")
            print(f"DONE")
        #########################################################

        mse_ = compute_mse(x, coords_pred)
        lddt_ = compute_lddt(x.view(-1, N, 3), coords_pred.view(-1, N, 3))

        for i in range(prog_args.batch_size):
            tm_score_ = compute_tm_score(x.view(-1, N, 3)[i, ca_indices, :].cpu().numpy(), ref_seq, coords_pred.view(-1, N, 3)[i, ca_indices, :].cpu().numpy(), ref_seq) 
            #compute_tm_score_torch(ref_coords, pred) # n_residues=len(ref_seq))
            # tm_score_ = compute_tm_score_torch(sample[ca_indices, :], coords_pred[i, ca_indices, :])
            metrics["tm_score_correct"].append(tm_score_)


        #### backbone
        x_non_bb = x.view(args.batch_size, N, 3)[:, sc_idx, :].view(-1, 3)
        coord_non_bb = coords_pred.view(-1, N, 3)[:, sc_idx, :].view(-1,3)

        coord_bb_pred = coords_pred.view(-1, N, 3)[:, bb_idx,:].view(-1,3)
        coord_bb_true = x.view(args.batch_size, N, 3)[:, bb_idx,:].view(-1,3)

        mse_bb = torch.nn.functional.mse_loss(coord_bb_pred, coord_bb_true.to(device))
        mse_sc = torch.nn.functional.mse_loss(coord_non_bb, x_non_bb.to(device))

        lddt_bb = lddt(coord_bb_pred.view(args.batch_size, -1, 3), coord_bb_true.view(args.batch_size, -1, 3))

        metrics["lddt"].append(lddt_)
        metrics["tm_score"].append(tm_score_)
        metrics["mse"].append(mse_)

        metrics["mse_bb"].append(mse_bb)
        metrics["lddt_bb"].append(lddt_bb)
        metrics["mse_sc"].append(mse_sc)

    # Print average metrics for the epoch
    print(f"----------------------------------------------------------------------")
    print(f"Average LDDT: {torch.tensor(metrics['lddt']).mean():.3f}")
    print(f"Average TM-Score: {torch.tensor(metrics['tm_score']).mean():.3f}")
    print(f"Average MSE: {torch.tensor(metrics['mse']).mean():.3f}")
    print(f"Average BB MSE: {torch.tensor(metrics['mse_bb']).mean():.3f}")
    print(f"Average SC MSE: {torch.tensor(metrics['mse_sc']).mean():.3f}")
    print(f"Average BB LDDT: {torch.tensor(metrics['lddt_bb']).mean():.3f}")
    print(f"Average Corrected TM-Score: {torch.tensor(metrics['tm_score_correct']).mean():.3f}")
    print(f"----------------------------------------------------------------------")
    return metrics  



@torch.no_grad()
def test_gpcrmd_(args, dataloader, model, pdb_path="heavy_chain.pdb"):
    model.eval()
    metrics = {
        "lddt": [], "tm_score": [], "mse": [],
        "mse_bb": [], "lddt_bb": [], "mse_sc": [],
        "tm_score_correct": []
    }

    u        = mda.Universe(pdb_path)
    protein  = u.select_atoms("protein")
    backbone = u.select_atoms("protein and backbone")
    ca       = u.select_atoms("protein and name CA")
    bb_idx   = backbone.indices
    sc_idx   = u.select_atoms("protein and not backbone").indices
    ca_indices = ca.indices
    ref_seq  = [r.resname for r in backbone.residues]

    N = len(protein)

    print(f"Protein atoms:   {len(protein)}")
    print(f"CA atoms:        {len(ca_indices)}")
    print(f"Backbone atoms:  {len(bb_idx)}")
    print(f"Sidechain atoms: {len(sc_idx)}")
    print(f"Total:           {len(bb_idx) + len(sc_idx)}")

    for batch_idx, data in enumerate(dataloader):
        x          = data.x.float().to(args.device)
        edge_index = data.edge_index.to(torch.int64).to(args.device)
        batch      = data.batch.to(torch.int64).to(args.device)
        C_input    = to_dense_adj(edge_index, batch=batch)

        _, coords_pred = model(x, edge_index, batch, C_input, args.M)

        # save predicted conformation for first batch
        if batch_idx == 0:
            protein.positions = coords_pred[0].cpu().numpy()
            protein.write(f"_{args.dataset}_predicted_conformation.pdb")

        x_reshaped    = x.view(args.batch_size, N, 3)
        pred_reshaped = coords_pred.view(args.batch_size, N, 3)

        # full atom metrics
        mse_  = torch.nn.functional.mse_loss(coords_pred, x.to(device))
        # compute_mse(x, coords_pred)
        lddt_ = compute_lddt(x_reshaped, pred_reshaped)

        # TM-score per sample
        for i in range(args.batch_size):
            tm_score_ = compute_tm_score(
                x_reshaped[i, ca_indices, :].cpu().numpy(), ref_seq,
                pred_reshaped[i, ca_indices, :].cpu().numpy(), ref_seq
            )
            metrics["tm_score_correct"].append(tm_score_)

        # backbone metrics
        coord_bb_pred = pred_reshaped[:, bb_idx, :].reshape(-1, 3)
        coord_bb_true = x_reshaped[:, bb_idx, :].reshape(-1, 3)
        mse_bb  = F.mse_loss(coord_bb_pred, coord_bb_true.to(args.device))
        lddt_bb = compute_lddt(
            coord_bb_pred.view(args.batch_size, -1, 3),
            coord_bb_true.view(args.batch_size, -1, 3)
        )

        # sidechain metrics
        coord_sc_pred = pred_reshaped[:, sc_idx, :].reshape(-1, 3)
        coord_sc_true = x_reshaped[:, sc_idx, :].reshape(-1, 3)
        mse_sc = F.mse_loss(coord_sc_pred, coord_sc_true.to(args.device))

        print(f"batch : {batch_idx} and mse : {mse_:.3f}")

        metrics["lddt"].append(lddt_)
        metrics["tm_score"].append(tm_score_)
        metrics["mse"].append(mse_)
        metrics["mse_bb"].append(mse_bb)
        metrics["lddt_bb"].append(lddt_bb)
        metrics["mse_sc"].append(mse_sc)

    print(f"{'─'*60}")
    print(f"Average MSE:              {torch.tensor(metrics['mse']).mean():.3f}")
    print(f"Average BB MSE:           {torch.tensor(metrics['mse_bb']).mean():.3f}")
    print(f"Average SC MSE:           {torch.tensor(metrics['mse_sc']).mean():.3f}")
    print(f"Average lDDT:             {torch.tensor(metrics['lddt']).mean():.3f}")
    print(f"Average BB lDDT:          {torch.tensor(metrics['lddt_bb']).mean():.3f}")
    print(f"Average TM-Score:         {torch.tensor(metrics['tm_score_correct']).mean():.3f}")
    print(f"{'─'*60}")

    return metrics



@torch.no_grad()
def test_d2r(args, dataloader, model, pdb_path="heavy_chain.pdb"):
    model.eval()
    metrics = {
        "lddt": [], "tm_score": [], "mse": [],
        "mse_bb": [], "lddt_bb": [], "mse_sc": [],
        "tm_score_correct": []
    }

    u        = mda.Universe(pdb_path)
    protein  = u.select_atoms("protein")
    backbone = u.select_atoms("protein and backbone")
    ca       = u.select_atoms("protein and name CA")
    bb_idx   = backbone.indices
    sc_idx   = u.select_atoms("protein and not backbone").indices
    ca_indices = ca.indices
    ref_seq  = [r.resname for r in backbone.residues]

    N = len(protein)

    print(f"Protein atoms:   {len(protein)}")
    print(f"CA atoms:        {len(ca_indices)}")
    print(f"Backbone atoms:  {len(bb_idx)}")
    print(f"Sidechain atoms: {len(sc_idx)}")
    print(f"Total:           {len(bb_idx) + len(sc_idx)}")


    protein = torch.tensor(protein.positions).to(device)
    print(f"\n protein.shape {protein.shape} \n")

    ca = torch.tensor(ca.positions).to(device)

    for batch_idx, data in enumerate(dataloader):
        x          = data.x.float().to(args.device)
        edge_index = data.edge_index.to(torch.int64).to(args.device)
        batch      = data.batch.to(torch.int64).to(args.device)
        C_input    = to_dense_adj(edge_index, batch=batch)

        _, coords_pred = model(x, edge_index, batch, C_input, args.M)

        # save predicted conformation for first batch
        #if batch_idx == 0:
        #    protein.positions = coords_pred[0].cpu().numpy()
        #    protein.write(f"_{args.dataset}_predicted_conformation.pdb")

        # x_reshaped    =  #x.view(args.batch_size, N, 3)
        pred_reshaped = coords_pred.view(args.batch_size, N, 3)

        for i, pred in enumerate(pred_reshaped):

            _, pred = kabsch_algorithm(protein, pred)

            # full atom metrics
            mse_  = torch.nn.functional.mse_loss(pred, protein.to(device))
            # compute_mse(x, coords_pred)
            lddt_ = compute_lddt(protein, pred)

            tm_score_ = compute_tm_score(
                ca.cpu().numpy(), ref_seq,
                pred[ca_indices,:].cpu().numpy(), ref_seq
            )
            # tm_score(protein, pred)
            
            """
            compute_tm_score(
                ca.cpu().numpy(), ref_seq,
                pred[ca_indices,:].cpu().numpy(), ref_seq
            )
            """
            metrics["tm_score_correct"].append(tm_score_)

            # backbone metrics
            coord_bb_pred = pred[bb_idx, :].reshape(-1, 3)
            coord_bb_true = protein[bb_idx, :].reshape(-1, 3)
            mse_bb  = F.mse_loss(coord_bb_pred, coord_bb_true.to(args.device))
            lddt_bb = compute_lddt(
                coord_bb_pred.view(-1, 3),
                coord_bb_true.view(-1, 3)
            )

            # sidechain metrics
            coord_sc_pred = pred[sc_idx, :].reshape(-1, 3)
            coord_sc_true = protein[sc_idx, :].reshape(-1, 3)
            mse_sc = F.mse_loss(coord_sc_pred, coord_sc_true.to(args.device))

            metrics["lddt"].append(lddt_)
            metrics["tm_score"].append(tm_score_)
            metrics["mse"].append(mse_)
            metrics["mse_bb"].append(mse_bb)
            metrics["lddt_bb"].append(lddt_bb)
            metrics["mse_sc"].append(mse_sc)


    print(f"{'─'*60}")
    print(f"Average MSE:              {torch.tensor(metrics['mse']).mean():.3f}")
    print(f"Average BB MSE:           {torch.tensor(metrics['mse_bb']).mean():.3f}")
    print(f"Average SC MSE:           {torch.tensor(metrics['mse_sc']).mean():.3f}")
    print(f"Average lDDT:             {torch.tensor(metrics['lddt']).mean():.3f}")
    print(f"Average BB lDDT:          {torch.tensor(metrics['lddt_bb']).mean():.3f}")
    print(f"Average TM-Score:         {torch.tensor(metrics['tm_score_correct']).mean():.3f}")
    print(f"{'─'*60}")

    return metrics








if __name__=="__main__":
    
    paths = ["/home/binal1/Graphons/ss_molIGNR_prot_hyper/Results/checkpoints/coordinate_decoder/_full_knn4_gin_epoch_195_dim_16_lr_0.01.pt"] #["Results/checkpoints_final/_dyn_1119_heavy_gin_epoch_195_dim_16_lr_0.01.pt"]

    # "/home/binal1/Graphons/ss_molIGNR_prot_hyper/Results/checkpoints_final/_dyn_1119_heavy_gin_epoch_195_dim_16_lr_0.01.pt"
    # ["Results/checkpoints/coordinate_decoder/_dyn_799_knn4_gin_epoch_145_dim_16_lr_0.01.pt"]

    # ["Results/checkpoints/coordinate_decoder/_full_knn10_gin_epoch_145_dim_16_lr_0.01.pt"]
    # ["Results/checkpoints/coordinate_decoder/_dyn_1119_knn4_gin_epoch_145_dim_16_lr_0.01.pt"]
    # ["Results/checkpoints/coordinate_decoder/_dyn_95_knn4_gin_epoch_145_dim_16_lr_0.01.pt"]
    # ["Results/checkpoints/coordinate_decoder/_dyn_1119_knn4_gin_epoch_145_dim_16_lr_0.01.pt"]
    
    """
    u = mda.Universe("data/dyn_95.pdb")
    all_protein   = u.select_atoms("protein")
    heavy_protein = u.select_atoms("protein and not type H")

    print(f"All protein atoms:   {len(all_protein)}")
    print(f"Heavy protein atoms: {len(heavy_protein)}")
    """

    for path in paths:
        print(f"path : {path}")
        model, prog_args, epoch, _ = load_model(path) # , dataset="dyn_1119_knn4")
        # prog_args.dataset = "dyn_1119_knn4"
        train_loader, test_loader, _ = get_dataset(prog_args, shuffle = True)
        print(f"running {prog_args.gnn_type}")
        print()
        print(f"prog_args: {prog_args}")


        if "bb" in prog_args.dataset:
            N = 1091
        elif "full" in prog_args.dataset:
            N = 2191    # total number of atoms in the protein
        elif prog_args.dataset == "dyn_709":
            N = 4782
        elif prog_args.dataset == "dyn_1119_knn4":
            N = 4940
        elif prog_args.dataset == "dyn_1119_knn10":
            N = 4940
        elif prog_args.dataset == "dyn_95_knn4":
            N = 4618
        elif prog_args.dataset == "dyn_95_knn10":
            N = 4618
        elif prog_args.dataset == "dyn_95_backbone_knn4":
            N = 1091
        elif prog_args.dataset == "dyn_799_knn4":
            N = 4793   
        elif prog_args.dataset == "dyn_95_heavy":
            N = 4250
        elif prog_args.dataset == "dyn_1119_heavy":
            N = 4539
        else:
            print("It should be either backbone or full atom structure... ")

        # visualize_latents(prog_args, model, train_loader, N = N, epoch = f"{epoch}", inference = False)
        print("visualization done....")

        pdb_path = "data/heavy_chain.pdb"
        times = []
        for _ in range(3):
            s = time.time()
            test_d2r(prog_args, train_loader, model, pdb_path)
            #n_sample = 1000
            #sigma = 0
            #evaluate_backbone(model, train_loader, pdb_path, prog_args, N = N, number_samples=n_sample, sigma = sigma)
            #times.append(np.round(time.time()-s,3))
            #print(f"\nTimes : {times}\n")
            #print(f"{time.time()-s} seconds.....")

            break

        
        print(f"Time : {np.mean(times)}")
        
        sigmas = [0.1, 0.2, 0.3]
        num_samples_list = [1000] # [500, 1000, 2000]
        # for sigma in sigmas:
        #    for n_sample in num_samples_list:
        #        print(f" ############### Sigma = {sigma} with {n_sample} samples ##############")
        #        evaluate_backbone(model, train_loader, pdb_path, prog_args, N = N, number_samples=n_sample, sigma = sigma)
        #        print()

        # pdb_path = "data/dyn_1119_heavy.pdb"
        sigma = 0.0
        #ramachandran_interpolation_(
        #    model, train_loader, prog_args,
        #    pdb_path=pdb_path,
        #    plot_name=f"ramachandrans/{prog_args.dataset}_gt",
        #    N=N, sigma=sigma
        #)


        """
        ramachandran_linear_interpolation_(
            model, train_loader, prog_args,
            pdb_path=pdb_path,
            plot_name=f"ramachandrans/{prog_args.dataset}_linear_interpolation_",
            N=N, sigma=sigma, interval = 100)
        print("\n################### ramachandran is done... ###################\n")
        """
        """
        for sigma in sigmas:
            # ramachandran_interpolation2(model, train_loader, prog_args, pdb_path=pdb_path, plot_name = f"R_{prog_args.dataset}_{epoch}_noise_interpolation_{sigma}", sigma = 0.2, N=N)
            for num_sample in num_samples_list:
                print(f"\n----------------------- num_samples : {num_sample}, sigma : {sigma} -----------------------\n")
                new_samples = evaluate_backbone(model, train_loader, pdb_path, prog_args = prog_args, N = N, number_samples = num_sample, sigma = sigma, plot_name = f"R_CA_{sigma}_{num_sample}")
        """

