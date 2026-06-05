import torch

from load_checkpoints import load_model

from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from molIGNR_prot.evaluation.metrics import compute_lddt, compute_mse, tm_score, compute_tm_score_torch, kabsch_algorithm, compute_tm_score
# from utils import visualize_latents
import MDAnalysis as mda


@torch.no_grad()
def run_linear_interpolation(model, prog_args, full_data_path, data_path, interval = 10, N=2191):

    full_data = torch.load(full_data_path, weights_only=False)
    data = torch.load(data_path, weights_only=False)

    full_loader = DataLoader(full_data, batch_size=prog_args.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True) 
    data_loader = DataLoader(data, batch_size=prog_args.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True) 


    z_10 = []
    for idx, data in enumerate(data_loader):
        z, _ = model.encode(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        z_10.append(z.cpu().detach())
    z_10 = torch.cat(z_10, dim=0)

    
    reconstructed = []
    for i in range(len(z_10) - 1):
        ts = torch.linspace(0, 1, steps=interval+1)[1:-1]  # [0.1, 0.2, ..., 0.9]
        z_interp = z_10[i] + ts.unsqueeze(-1) * (z_10[i+1] - z_10[i])  # (9, 16)
        new_coords = model.mlp_coords(z_interp.to(device)).view(-1, N, 3) # (9, 3)
        reconstructed.append(new_coords)

    reconstructed = torch.cat(reconstructed, dim=0) # [ , N, 3]

    u = mda.Universe("data/heavy_chain.pdb")
    bb_atoms = u.select_atoms("protein and backbone")
    bb_idx = bb_atoms.indices
    # Sidechain atom indices (protein, heavy atoms, not backbone)
    sc_atoms = u.select_atoms("protein and not backbone and not name H*")
    sc_idx = sc_atoms.indices
    # CA indices 
    ca_atoms = u.select_atoms("protein and name CA")
    ca_indices = ca_atoms.indices

    ref_seq = [r.resname for r in u.select_atoms("protein").residues]

    cntr = 0
    metrics = {"mse" : [],
               "lddt" : [],
               "tm_score" : [],
               "mse_bb" : [],
               "mse_sc" : [],
               "lddt_bb" : []}
    
    for i, data in enumerate(full_data):
        x = data.x.to(device)

        if i%interval == 0:
            continue
        else:
            inter_coords = reconstructed[cntr]
            _, inter_coords = kabsch_algorithm(x.cpu(), inter_coords.cpu())
            inter_coords = inter_coords.to(device)

            ##### compute the metrics .... 
            mse_ = torch.nn.functional.mse_loss(inter_coords, x) 
            lddt_ = compute_lddt(x, inter_coords)
            tm_score_ = compute_tm_score(x[ca_indices].cpu(), ref_seq, inter_coords[ca_indices].cpu(), ref_seq)

            mse_bb = torch.nn.functional.mse_loss(inter_coords[bb_idx], x[bb_idx]) # compute_mse(inter_coords[bb_idx], x[bb_idx])
            mse_sc = torch.nn.functional.mse_loss(inter_coords[sc_idx], x[sc_idx]) # compute_mse(inter_coords[sc_idx], x[sc_idx])
            lddt_bb = compute_lddt(x[bb_idx], inter_coords[bb_idx])

            metrics["mse"].append(mse_)
            metrics["lddt"].append(lddt_)
            metrics["tm_score"].append(tm_score_)

            metrics["mse_bb"].append(mse_bb)
            metrics["mse_sc"].append(mse_sc)
            metrics["lddt_bb"].append(lddt_bb)
        
            cntr +=1

            if cntr>=reconstructed.shape[0]:
                break

    print(f"Average MSE : {torch.mean(torch.tensor(metrics['mse'])):.3f}"
          f"\nAverage lDDT : {torch.mean(torch.tensor(metrics['lddt'])):.3f}"
          f"\nAverage TM-score : {torch.mean(torch.tensor(metrics['tm_score'])):.3f}",
          f"\nAverage MSE BB : {torch.mean(torch.tensor(metrics['mse_bb'])):.3f}",
            f"\nAverage MSE SC : {torch.mean(torch.tensor(metrics['mse_sc'])):.3f}",
            f"\nAverage lDDT BB : {torch.mean(torch.tensor(metrics['lddt_bb'])):.3f}")
    
    return reconstructed







if __name__ == "__main__":

    # Fill in the corresponding full data and masked data directories and the corresponding checkpoint 
    full_data_path = "...."
    intervals = [10,20,100]
    for interval in intervals: 
        print(f"----------- Interval : {interval} ----------\n")
        data_path = "..."
        checkpoints = "..."
        model, prog_args, epoch, _  = load_model(checkpoints)

        run_linear_interpolation(model.to(device), prog_args, full_data_path, data_path, interval = interval, N =2191)



