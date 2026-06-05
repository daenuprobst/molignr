import MDAnalysis as mda

from MDAnalysis.analysis.dihedrals import Ramachandran
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

import torch

import tmtools

from torch_geometric.loader import DataLoader
try:
    from Bio.PDB import PDBParser
    from Bio.SVDSuperimposer import SVDSuperimposer
except ImportError:
    raise ImportError("Run: pip install biopython")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################ lDDT and TM-score ############################
@torch.no_grad()
def compute_lddt(
    ref_coords: torch.Tensor,
    sample_coords: torch.Tensor,
    inclusion_radius: float = 15.0,
    thresholds: tuple = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """lDDT — superposition-free local distance preservation. Range [0, 1]."""
    ref_coords    = ref_coords.float()
    sample_coords = sample_coords.float()

    ref_dists    = torch.cdist(ref_coords,    ref_coords)     # (N, N)
    sample_dists = torch.cdist(sample_coords, sample_coords)  # (N, N)

    diff = torch.abs(ref_dists - sample_dists)
    mask = (ref_dists < inclusion_radius) & (ref_dists > 0)

    n_pairs = mask.sum()
    if n_pairs == 0:
        return 0.0

    thresholds_t  = torch.tensor(thresholds, dtype=torch.float32, device=ref_coords.device)
    diff_masked   = diff[mask]                                          # (M,)
    per_threshold = (diff_masked.unsqueeze(1) < thresholds_t).float().mean(dim=0)  # (4,)

    return round(per_threshold.mean().item(), 4)

def compute_mse(coords1, coords2_aligned):
    return torch.mean(torch.sum((coords1 - coords2_aligned) ** 2, dim=-1))

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "MSE":"M","HSD":"H","HSE":"H","HSP":"H",
}

@torch.no_grad()
def compute_tm_score(
    ref_coords: torch.Tensor,
    ref_seq: list[str],
    sample_coords: torch.Tensor,
    sample_seq: list[str],
) -> float:
    """TM-score normalised by reference length. Range (0, 1], >0.5 = same fold."""
    ref_str    = "".join(THREE_TO_ONE.get(r, "X") for r in ref_seq)
    sample_str = "".join(THREE_TO_ONE.get(r, "X") for r in sample_seq)
    result = tmtools.tm_align(
        sample_coords.detach().cpu().numpy(),
        ref_coords.detach().cpu().numpy(),
        sample_str,
        ref_str,
    )
    return float(result.tm_norm_chain2)

def compute_centroid(X: torch.Tensor) -> torch.Tensor: 
    """ Source : https://github.com/adityasengar/LD-FPG """ 
    return X.mean(dim=-2)

def kabsch_algorithm(P: torch.Tensor, Q: torch.Tensor):
    """ Aligns Q onto P using Kabsch algorithm. Handles batches [B, N, 3].
    Source: https://github.com/adityasengar/LD-FPG """
    P, Q = P.float(), Q.float(); is_batched = P.ndim == 3
    if not is_batched: P, Q = P.unsqueeze(0), Q.unsqueeze(0)
    B, N, _ = P.shape; centroid_P, centroid_Q = compute_centroid(P), compute_centroid(Q)
    P_c, Q_c = P - centroid_P.unsqueeze(1), Q - centroid_Q.unsqueeze(1)
    C = torch.bmm(Q_c.transpose(1, 2), P_c)
    try: V, S, Wt = torch.linalg.svd(C)
    except Exception as e:
        print(f"Kabsch SVD failed: {e}. Return identity align.")
        U_fallback = torch.eye(3, device=P.device).unsqueeze(0).expand(B, -1, -1)
        Q_aligned_fallback = Q - centroid_Q.unsqueeze(1) + centroid_P.unsqueeze(1)
        return (U_fallback.squeeze(0), Q_aligned_fallback.squeeze(0)) if not is_batched else (U_fallback, Q_aligned_fallback)
    det = torch.det(torch.bmm(V, Wt)); D = torch.eye(3, device=P.device).unsqueeze(0).repeat(B, 1, 1)
    D[:, 2, 2] = torch.sign(det); U = torch.bmm(torch.bmm(V, D), Wt)
    Q_aligned = torch.bmm(Q_c, U) + centroid_P.unsqueeze(1)
    return (U.squeeze(0), Q_aligned.squeeze(0)) if not is_batched else (U, Q_aligned)

################################## RAMACHANDRAN #######################

def save_histogram(all_phi, all_psi, plot_name):
    H, xedges, yedges = np.histogram2d(
        all_phi, all_psi,
        bins=200,
        range=[[-180,180],[-180,180]],
        density=True,
    )
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    im = ax.imshow(H.T, origin='lower',
            extent=[-180, 180, -180, 180],
            cmap="viridis",
            norm=matplotlib.colors.LogNorm())
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability density (log scale)", fontsize=11)
    
    ax.set_xlabel("φ (deg)", fontsize=12)
    ax.set_ylabel("ψ (deg)", fontsize=12)
    ax.set_title("Ramachandran Plot", fontsize=13)
    
    # optional: add reference lines at 0
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='white', linewidth=0.5, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(f"{plot_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

def dihedral_torch(coords):
    """
    coords => (batch, 4, 3)
    returns => (batch, ) in radians
    """
    b1 = coords[:,1] - coords[:,0]
    b2 = coords[:,2] - coords[:,1]
    b3 = coords[:,3] - coords[:,2]
    n1 = torch.cross(b1,b2,dim=1)
    n2 = torch.cross(b2,b3,dim=1)
    b2_len = torch.norm(b2,dim=1,keepdim=True).clamp(min=1e-6)
    b2_unit = b2 / b2_len
    m1 = torch.cross(n1,b2_unit,dim=1)
    x = torch.sum(n1*n2,dim=1)
    y = torch.sum(m1*n2,dim=1)
    angles = torch.atan2(y,x)
    return angles

###################### Ramachandrans #####################

def get_angles(coords, idx_N, idx_CA, idx_C):
    """
    Args:
        coords:         (B, N_atoms, 3) torch tensor
        idx_N/CA/C:     precomputed atom indices
    """
    B = coords.shape[0]
    device = coords.device

    idx_N_t  = torch.as_tensor(idx_N,  dtype=torch.long, device=device)
    idx_CA_t = torch.as_tensor(idx_CA, dtype=torch.long, device=device)
    idx_C_t  = torch.as_tensor(idx_C,  dtype=torch.long, device=device)

    N_  = coords[:, idx_N_t,  :]  # (B, R, 3)
    CA_ = coords[:, idx_CA_t, :]  # (B, R, 3)
    C_  = coords[:, idx_C_t,  :]  # (B, R, 3)

    # Phi: C(i-1), N(i), CA(i), C(i)
    quad_phi = torch.stack([
        C_[:, :-1, :],   # C(i-1)
        N_[:, 1:,  :],   # N(i)
        CA_[:, 1:, :],   # CA(i)
        C_[:, 1:,  :]    # C(i)
    ], dim=2)            # (B, R-1, 4, 3)

    # Psi: N(i), CA(i), C(i), N(i+1)
    quad_psi = torch.stack([
        N_[:, :-1,  :],  # N(i)
        CA_[:, :-1, :],  # CA(i)
        C_[:, :-1,  :],  # C(i)
        N_[:, 1:,   :]   # N(i+1)
    ], dim=2)            # (B, R-1, 4, 3)

    phi = dihedral_torch(quad_phi.reshape(-1, 4, 3)).reshape(B, -1)
    psi = dihedral_torch(quad_psi.reshape(-1, 4, 3)).reshape(B, -1)

    phi_deg = (phi * 180.0 / np.pi).reshape(-1).cpu().numpy()
    psi_deg = (psi * 180.0 / np.pi).reshape(-1).cpu().numpy()

    return phi_deg, psi_deg

def get_indices(pdb_path):
    u        = mda.Universe(pdb_path)
    idx_N    = u.select_atoms("protein and name N").indices
    idx_CA   = u.select_atoms("protein and name CA").indices
    idx_C    = u.select_atoms("protein and name C").indices
    return idx_N, idx_CA, idx_C

def ramachandran_latent_perturbation(
    model, dataloader, prog_args,
    pdb_path="data/heavy_chain.pdb",
    plot_name="perturbation_histogram",
    N=None, sigma=0.1
):
    idx_N, idx_CA, idx_C = get_indices(pdb_path)

    if N is None:
        u = mda.Universe(pdb_path)
        N = len(u.select_atoms("protein and not H"))

    phi_all = []
    psi_all = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(prog_args.device)

            with torch.amp.autocast('cuda'):
                z, _        = model.encode(batch.x, batch.edge_index, batch.batch)
                z_noisy     = z + sigma * torch.randn_like(z)
                coords_pred = model.mlp_coords(z_noisy).view(len(z_noisy), N, 3)

            phi_batch, psi_batch = get_angles(batch.x.view(-1, N, 3), idx_N, idx_CA, idx_C)
            phi_all.extend(phi_batch)
            psi_all.extend(psi_batch)

    phi_all = np.asarray(phi_all, dtype=np.float32)
    psi_all = np.asarray(psi_all, dtype=np.float32)

    print(f"Total angles: {len(phi_all)}")
    print(f"phi range: {phi_all.min():.2f} to {phi_all.max():.2f}")
    print(f"psi range: {psi_all.min():.2f} to {psi_all.max():.2f}")

    save_histogram(all_phi=phi_all, all_psi=psi_all, plot_name=plot_name)
    print("Done.")

@torch.no_grad()
def get_linear_reconstruction(model, data_loader, interval = 10, N=2191):
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

    return reconstructed

def ramachandran_linear_interpolation(
    model, dataloader, prog_args,
    pdb_path="data/heavy_chain.pdb",
    plot_name="interpolation_histogram",
    N=None, sigma=0.1, interval = 10
):
    """
    dataloader should be the one used for linear interpolation, i.e. with every 10th sample from the full dataset.
    """
    idx_N, idx_CA, idx_C = get_indices(pdb_path)

    if N is None:
        u = mda.Universe(pdb_path)
        N = len(u.select_atoms("protein and not H"))

    phi_all = []
    psi_all = []

    reconstructed = get_linear_reconstruction(model, dataloader, interval)
    print(f"reconstructed.shape : {reconstructed.shape}")

    # for sample in reconstructed:
    phi_batch, psi_batch = get_angles(reconstructed, idx_N, idx_CA, idx_C)
    phi_all.extend(phi_batch)
    psi_all.extend(psi_batch)

    phi_all = np.asarray(phi_all, dtype=np.float32)
    psi_all = np.asarray(psi_all, dtype=np.float32)

    print(f"Total angles: {len(phi_all)}")
    print(f"phi range: {phi_all.min():.2f} to {phi_all.max():.2f}")
    print(f"psi range: {psi_all.min():.2f} to {psi_all.max():.2f}")

    save_histogram(all_phi=phi_all, all_psi=psi_all, plot_name=plot_name)
    print("Done.")

############################### Interpolation and sampling from the latent space #################################

def gaussian_latent_perturbation_backbone(model, dataloader, pdb_path, prog_args, N = 2191, number_samples = 1000, sigma = 0.1):
    all_z = []
    all_noisy_z = []

    model.eval()
    generated_samples = []

    u = mda.Universe(pdb_path)
    protein = u.select_atoms("protein")
    backbone = u.select_atoms("protein and backbone")
    ca = u.select_atoms("protein and name CA")
    bb_indices = backbone.indices
    sc_indices = u.select_atoms("protein and not backbone").indices
    ca_indices = ca.indices

    print(f"protein atoms:   {len(protein.indices)}")
    print(f"CA atoms:        {len(ca_indices)}")
    print(f"Backbone atoms:  {len(bb_indices)}")
    print(f"Sidechain atoms: {len(sc_indices)}")
    print(f"Total:           {len(bb_indices) + len(sc_indices)}")
    print(f"Secretin in play.... :)) ")

    ref_coords = torch.tensor(protein.positions).to(device)
    backbone_coords = torch.tensor(backbone.positions).to(device)
    ca_coords = torch.tensor(ca.positions).to(device)
    print(f"ref_coords: {ref_coords.shape}")

    ref_seq = [r.resname for r in u.select_atoms("protein and backbone").residues]

    cntr = 0

    lddts = []
    tm_scores = []
    mse_score = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(prog_args.device)
            z, _ = model.encode(batch.x, batch.edge_index, batch.batch)
            z_noisy = z + sigma * torch.randn_like(z)

            coords_pred = model.mlp_coords(z_noisy).view(prog_args.batch_size, N, 3)
            generated_samples.append(coords_pred.cpu())

            for pred in coords_pred:
                pred = pred.to(device)
                _, pred = kabsch_algorithm(ref_coords, pred)

                mse_ = torch.nn.functional.mse_loss(backbone_coords, pred[bb_indices]) 
                lddt_ = compute_lddt(backbone_coords, pred[bb_indices])
                tm_score_ = compute_tm_score(ca_coords, ref_seq, pred[ca_indices], ref_seq) 

                lddts.append(lddt_)
                tm_scores.append(tm_score_)
                mse_score.append(mse_.cpu().numpy())

                cntr +=1

            all_z.append(z.cpu())
            all_noisy_z.append(z_noisy.cpu())

            if cntr>=number_samples:
                break

    print(f"cntr : {cntr}")
    print()     
    print(f"{'Mean':<35} {np.mean(mse_score):10.3f} {np.mean(lddts):8.4f} {np.mean(tm_scores):10.4f}")
    print(f"{'Std':<35} {np.std(mse_score):10.3f} {np.std(lddts):8.4f} {np.std(tm_scores):10.4f}")
    print(f"{'Min':<35} {np.min(mse_score):10.3f} {np.min(lddts):8.4f} {np.min(tm_scores):10.4f}")
    print(f"{'Max':<35} {np.max(mse_score):10.3f} {np.max(lddts):8.4f} {np.max(tm_scores):10.4f}")

    print("Done with interpolation batches...")


    return torch.cat(generated_samples, dim=0)  # [number_samples, N, 3]


    batch_size = 12

    full_data = torch.load("/home/binal1/LD-FPG/blind/data/processed/full_data_knn_4.pt", weights_only=False)
    data_ = torch.load(f"data/full_data_knn_4_{interval}.pt", weights_only=False)

    full_loader = DataLoader(full_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True) 
    data_loader = DataLoader(data_, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True) 


    
    z_10 = []
    for idx, data in enumerate(data_loader):
        z, _ = model.encode(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        z_10.append(z.cpu().detach())
    z_10 = torch.cat(z_10, dim=0)

    print(f"z_10 shape : {z_10.shape}")

    N=2191
    reconstructed = []
    for i in range(len(z_10) - 1):
        ts = torch.linspace(0, 1, steps=interval+1)[1:-1]  # [0.1, 0.2, ..., 0.9]
        z_interp = z_10[i] + ts.unsqueeze(-1) * (z_10[i+1] - z_10[i])  # (9, 16)
        new_coords = model.mlp_coords(z_interp.to(device)).view(-1, N, 3) # (9, 3)
        reconstructed.append(new_coords)

    reconstructed = torch.cat(reconstructed, dim=0) # [ , N, 3]
    print(f"reconstructed shape : {reconstructed.shape}")

    bb_idx = torch.load("/home/binal1/LD-FPG/blind/data/processed/atom_indices.pt", weights_only=False)["bb_indices"]                
    sc_idx = torch.load("/home/binal1/LD-FPG/blind/data/processed/atom_indices.pt", weights_only=False)["sc_indices"]
    u  = mda.Universe("data/heavy_chain.pdb")
    ca = u.select_atoms("protein and name CA")
    ca_indices = ca.indices 

    print(f"ca_indices shape : {ca_indices.shape}")

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
            # print(f"inter_coords shape : {inter_coords.shape}")
            # print(f"cntrs : {cntr}")
            _, inter_coords = kabsch_algorithm(x.cpu(), inter_coords.cpu())
            inter_coords = inter_coords.to(device)

            mse_ = compute_mse(inter_coords, x)
            lddt_ = compute_lddt(x, inter_coords)
            tm_score_ = compute_tm_score_torch(x[ca_indices].cpu(), inter_coords[ca_indices].cpu())

            mse_bb = compute_mse(inter_coords[bb_idx], x[bb_idx])
            mse_sc = compute_mse(inter_coords[sc_idx], x[sc_idx])
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

    print(f"cntr : {cntr}")

    print(f"Average MSE : {torch.mean(torch.tensor(metrics['mse'])):.3f}"
          f"\nAverage lDDT : {torch.mean(torch.tensor(metrics['lddt'])):.3f}"
          f"\nAverage TM-score : {torch.mean(torch.tensor(metrics['tm_score'])):.3f}",
          f"\nAverage MSE BB : {torch.mean(torch.tensor(metrics['mse_bb'])):.3f}",
            f"\nAverage MSE SC : {torch.mean(torch.tensor(metrics['mse_sc'])):.3f}",
            f"\nAverage lDDT BB : {torch.mean(torch.tensor(metrics['lddt_bb'])):.3f}")
    
    return reconstructed


    all_z = []
    all_noisy_z = []

    model.eval()
    generated_samples = []

    phi_all_interpolated = []
    psi_all_interpolated = []


    ca_idx    = get_ca_indices("heavy_chain.pdb")   # shape (273,)

    df = parse_pdb("heavy_chain.pdb")
 
    coords = df[["x", "y", "z"]].values  # (N, 3)
 

    ca_coords = all_atom_coords[ca_idx]          

    cntr = 0

    lddts = []
    tm_scores = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(prog_args.device)
            z, _ = model.encode(batch.x, batch.edge_index, batch.batch)
            z_noisy = z + sigma * torch.randn_like(z)

            coords_pred = model.mlp_coords(z_noisy).view(prog_args.batch_size, N, 3)
            generated_samples.append(coords_pred.cpu())

            lddt_ = lddt(
                coords_pred.detach(),
                batch.x.reshape(prog_args.batch_size, N, 3))

            tm_score_ = tm_score(
                batch.x,                                  
                coords_pred.detach().view(-1,3),                        
                )
            lddts.append(lddt_)
            tm_scores.append(tm_score_)

            cntr +=1

            all_z.append(z.cpu())
            all_noisy_z.append(z_noisy.cpu())

            phi_all_, psi_all_ = get_angles(coords_pred.cpu().detach(), batch_size=prog_args.batch_size)
            phi_all_interpolated.extend(phi_all_)
            psi_all_interpolated.extend(psi_all_)

            print(f"cntr : {cntr}, lddt : {lddt_:.3f}, tm_score : {tm_score_:.3f}")
            if cntr>=number_samples:
                print(f"cntr : {cntr}")
                break

    
    print()     
    print(f"Average lDDT of generated samples: {sum(lddts)/len(lddts):.3f}")
    print(f"Average TM-score of generated samples: {sum(tm_scores)/len(tm_scores):.3f}")

    phi_all_interpolated = np.asarray(phi_all_interpolated, dtype=np.float32)
    psi_all_interpolated = np.asarray(psi_all_interpolated, dtype=np.float32)
    print(f"len of phi_all_interpolated : {len(phi_all_interpolated)}")
    print(f"len of psi_all_interpolated : {len(psi_all_interpolated)}")

    print("Example phi range:", phi_all_interpolated.min(), phi_all_interpolated.max())
    print("Example psi range:", psi_all_interpolated.min(), psi_all_interpolated.max())

    save_histogram(all_phi=phi_all_interpolated, all_psi = psi_all_interpolated, plot_name=plot_name)
    print("Done with interpolation batches...")


    return torch.cat(generated_samples, dim=0)  # [number_samples, N, 3]