import MDAnalysis as mda
from MDAnalysisTests.datafiles import GRO, XTC

from MDAnalysis.analysis.dihedrals import Ramachandran
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import torch
from data_ import *


########################### lDDT and TM-score ########################

def lddt(predicted_coords, 
    true_coords,
    cutoff = torch.tensor(15.0).to(device),
    thresholds: list = [0.5, 1.0, 2.0, 4.0]
) -> float:
    """ LDDT is a measure of the difference between the true distance matrix
    and the distance matrix of the predicted points. The difference is computed only on 
    points closer than cutoff **in the true structure**.
    

    It is an approximate score. 
    Code is extracted and adapted from 
    AlphaFold 2: https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/lddt.py
    
    Thresholds are hardcoded to be [0.5, 1.0, 2.0, 4.0] as in the original paper.
    
    true_points_mask: (batch, length, 1) binary-valued float array.  This mask
      should be 1 for points that exist in the true points.
    
    I set the true_points_mask to 1 in the code. 
    
    """

    assert len(predicted_coords.shape) ==3
    assert predicted_coords.shape[-1] ==3

    dmat_true = torch.sqrt(
      1e-10
      + torch.sum(
          (true_coords[:, :, None] - true_coords[:, None, :]) ** 2, axis=-1
      ))
    #torch.cdist(true_coords, true_coords).to(device)  # [batch_size, N, N] , where N = 273 
    dmat_predicted = torch.sqrt(
      1e-10
      + torch.sum(
          (predicted_coords[:, :, None] - predicted_coords[:, None, :]) ** 2,
          axis=-1,
      ))

    #torch.cdist(predicted_coords, predicted_coords).to(device)  # [batch_size, N, N] , where N = 273 

    ### distances to score
    true_points_mask = torch.ones(true_coords.shape[0], true_coords.shape[1], 1).to(device)


    dist_to_score = (
        (dmat_true <cutoff) 
        * true_points_mask 
        * true_points_mask.permute(0, 2, 1)
        * (1 - torch.eye(dmat_true.shape[1]).to(device)))

    # shift unscores distances to be far away
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    score = 0.25* (
            (dist_l1 < 0.5).to(torch.float32) 
            + (dist_l1 < 1.0).to(torch.float32)
            + (dist_l1 < 2.0).to(torch.float32)
            + (dist_l1 < 4.0).to(torch.float32)
            )

    # Normalize over the approapriate axes (normalizing over batches)
    reduce_axes = (-2,-1)
    norm = 1.0 / (1e-10 + torch.sum(dist_to_score, axis = reduce_axes))
    score = norm * (1e-10 + torch.sum(dist_to_score * score, axis = reduce_axes))

    return round(torch.mean(torch.round(score, decimals = 3)).item(),3)


def tm_score(true_coords, predicted_coords):

    N = int(true_coords.shape[0])

    d = torch.linalg.vector_norm(true_coords - predicted_coords, dim = -1)

    d0 = 1.24 * (max(N-15, 1)) ** (1/3) - 1.8 
    d0 = max(d0, 0.5)

    score = (1.0/N) * torch.sum(1.0 / (1.0 + (d /d0)) **2)
    score = round(float(score.detach()), 3)

    return score


######################## INTERPOLATION ######################

def interpolate(prog_args, model, batch, z1, z2, steps = 10):
    model.eval()

    zs = []
    with torch.no_grad():
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            zs.append(z.unsqueeze(0))
        zs = torch.cat(zs, dim=0).to(prog_args.device)  # (steps, latent_dim)
        # Decode all interpolated points
        decoded_graphons = model.decode_graphon_from_latent(zs)  # (steps, num_points, num_points)
    return decoded_graphons  # (steps, num_points, num_points)


def interpolate_between_latents(z1, z2, steps=10):
    interpolated_z = []
    for alpha in np.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2
        interpolated_z.append(z.unsqueeze(0))
    interpolated_z = torch.cat(interpolated_z, dim=0)  # (steps, latent_dim)
    return interpolated_z  # (steps, latent_dim)


def slerp(v1, v2, t, DOT_THR=0.9995, to_cpu=False, zdim=-1):
    """SLERP for pytorch tensors interpolating `v1` to `v2` with scale of `t`.

    `DOT_THR` determines when the vectors are too close to parallel.
        If they are too close, then a regular linear interpolation is used.

    `to_cpu` is a flag that optionally computes SLERP on the CPU.
        If the input tensors were on a GPU, it moves them back after the computation.  

    `zdim` is the feature dimension over which to compute norms and find angles.
        For example: if a sequence of 5 vectors is input with shape [5, 768]
        Then `zdim = 1` or `zdim = -1` computes SLERP along the feature dim of 768.

    Theory Reference:
    https://splines.readthedocs.io/en/latest/rotation/slerp.html
    PyTorch reference:
    https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    Numpy reference: 
    https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    """

    # check if we need to move to the cpu
    if to_cpu:
        orig_device = v1.device
        v1, v2 = v1.to('cpu'), v2.to('cpu')

    # take the dot product between normalized vectors
    v1_norm = v1 / torch.norm(v1, dim=zdim, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=zdim, keepdim=True)
    dot = (v1_norm * v2_norm).sum(zdim)

    # if the vectors are too close, return a simple linear interpolation
    if (torch.abs(dot) > DOT_THR).any():
        # print(f'warning: v1 and v2 close to parallel, using linear interpolation instead.')
        res = (1 - t) * v1 + t * v2    

    # else apply SLERP
    else:
        # compute the angle terms we need
        theta   = torch.acos(dot)
        theta_t = theta * t
        sin_theta   = torch.sin(theta)
        sin_theta_t = torch.sin(theta_t)

        # compute the sine scaling terms for the vectors
        s1 = torch.sin(theta - theta_t) / sin_theta
        s2 = sin_theta_t / sin_theta

        # interpolate the vectors
        res = (s1.unsqueeze(zdim) * v1) + (s2.unsqueeze(zdim) * v2)

        # check if we need to move them back to the original device
        if to_cpu:
            res.to(orig_device)

    return res

######################## RAMACHANDRAN plots #######################

def save_histogram(all_phi, all_psi, plot_name):
    H, xedges, yedges = np.histogram2d(
        all_phi, all_psi,
        bins=200,
        range=[[-180,180],[-180,180]],
        density=True,
    )
    plt.imshow(H.T, origin='lower',
            extent=[-180,180,-180,180],
            cmap="viridis",
            norm=matplotlib.colors.LogNorm())
    # plt.colorbar(label="Probability (log scale)")
    plt.xlabel("φ (deg)")
    plt.ylabel("ψ (deg)")

    plt.savefig(f"{plot_name}.png")

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

def get_indices():

    u = mda.Universe("/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/intermediate_files/heavy_chain.pdb")
    idx_N  = []
    idx_CA = []
    idx_C  = []

    for res in u.residues:
        # try to get N, CA, C for this residue
        try:
            N_atom  = res.atoms.select_atoms("name N")[0]
            CA_atom = res.atoms.select_atoms("name CA")[0]
            C_atom  = res.atoms.select_atoms("name C")[0]
        except IndexError:
            # This residue is missing backbone atoms: skip it for φ/ψ
            continue

        idx_N.append(N_atom.index)   # MDAnalysis atom index (0..n_atoms-1)
        idx_CA.append(CA_atom.index)
        idx_C.append(C_atom.index)

    idx_N  = np.array(idx_N)
    idx_CA = np.array(idx_CA)
    idx_C  = np.array(idx_C)

    # print(f"idx_n : {idx_N}")
    return idx_N, idx_CA, idx_C

def get_angles(coords, batch_size=16):
    
    idx_N, idx_CA, idx_C = get_indices()

    idx_N_t  = torch.as_tensor(idx_N,  dtype=torch.long)
    idx_CA_t = torch.as_tensor(idx_CA,  dtype=torch.long)
    idx_C_t  = torch.as_tensor(idx_C,   dtype=torch.long)

    # Gather backbone coords: shape (B, n_residues_with_backbone, 3)
    N  = coords[:, idx_N_t,  :]
    CA = coords[:, idx_CA_t, :]
    C  = coords[:, idx_C_t,  :]

    ##### Phi computation.... 
    # indices : i = 1...n_residues-2
    C_im1 = C[:, :-1, :]
    N_i = N[:, 1:, :]
    CA_i = CA[:, 1:, :]
    C_i = C[:, 1:, :]
    N_ip1 = N[:, 2:, :]

    #### psi computation...
    # indices : i = 1...n_residues-2
    C_im1 = C_im1[:, :-1, :]
    N_i = N_i[:, :-1, :]
    CA_i = CA_i[:, :-1, ]
    C_i = C_i[:, :-1, :]
    # N_ip1 = N_ip1[:, :-1, :]

    quad_phi = torch.stack([C_im1, N_i, CA_i, C_i], dim=2)  # (B, n_residues-2, 4, 3)
    quad_psi = torch.stack([N_i, CA_i, C_i, N_ip1], dim=2)  # (B, n_residues-2, 4, 3)   

    ## compute the angles
    phi = dihedral_torch(quad_phi.view(-1,4,3)).view(batch_size, -1)
    psi = dihedral_torch(quad_psi.view(-1,4,3)).view(batch_size, -1)

    phi_deg = phi *180.0 / np.pi
    psi_deg = psi *180.0 / np.pi  

    phi_all = phi_deg.reshape(-1).cpu().numpy()
    psi_all = psi_deg.reshape(-1).cpu().numpy()

    return phi_all, psi_all

def ramachandran_full_data(dataloader, prog_args, plot_name = "_full_data_histogram_", N =2191):
    batch_size = prog_args.batch_size

    all_phi, all_psi = [], []

    for batch in dataloader:
        x = batch.x
        coords = x.view(batch_size, N, 3)

        phi_all, psi_all = get_angles(coords, batch_size=batch_size)
        all_phi.extend(phi_all)
        all_psi.extend(psi_all)

    all_phi= np.asarray(all_phi, dtype=np.float32)
    all_psi= np.asarray(all_psi, dtype=np.float32)

    print("Example phi range:", all_phi.min(), all_phi.max())
    print("Example psi range:", all_psi.min(), all_psi.max())
    print(f"len of all_phi : {len(all_phi)}")
    print(f"len of all_psi : {len(all_psi)}")

    save_histogram(all_phi=all_phi, all_psi = all_psi, plot_name=plot_name)

    print("Done with ramachandran plot for full data...")

def ramachandran_reconstruction(model, dataloader, prog_args, plot_name = "_reconstruction_histogram_", N=2191):
    batch_size = prog_args.batch_size
    all_phi, all_psi = [], []

    mse = torch.nn.MSELoss()
    loss_list = []

    for batch in dataloader:
        batch = batch.to(prog_args.device)
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                z, _ = model.encode(batch.x, batch.edge_index, batch.batch)
                coords_pred = model.mlp_coords(z)

                loss = mse(coords_pred.view(-1,3), batch.x)
                loss_list.append(loss.item())

                coords_pred = coords_pred.view(batch_size, N, 3)

            phi_all, psi_all = get_angles(coords_pred, batch_size=batch_size)

            all_phi.extend(phi_all)
            all_psi.extend(psi_all)
    
    print(f"Reconstruction MSE loss : {np.mean(loss_list)}")

    all_phi= np.asarray(all_phi, dtype=np.float32)
    all_psi= np.asarray(all_psi, dtype=np.float32)

    print("Example phi range:", all_phi.min(), all_phi.max())
    print("Example psi range:", all_psi.min(), all_psi.max())
    print(f"len of all_phi : {len(all_phi)}")
    print(f"len of all_psi : {len(all_psi)}")

    save_histogram(all_phi=all_phi, all_psi = all_psi, plot_name=plot_name)

    print("Done with ramachandran plot for reconstruction...")



def ramachandran_interpolation(model, dataloader, prog_args, plot_name = "_interpolation_histogram_", N=2191):
    batch_size = prog_args.batch_size

    all_phi, all_psi = [], []
    phi_all_interpolated = []
    psi_all_interpolated = []

    steps = batch_size

    for batch in dataloader:
        batch = batch.to(prog_args.device)
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                ### interpolate here...
                z, _ = model.encode(batch.x, batch.edge_index, batch.batch)
                z1 = z[0]
                z2 = z[-1]

                interpolated_latents = []

                alphas = np.linspace(0, 1, steps + 2)[1:-1]
                for alpha in alphas:
                    # print(f"alpha : {alpha}")
                    interpolated_latent = slerp(z1, z2, t = alpha)
                    interpolated_latents.append(interpolated_latent.unsqueeze(0))
                
                interpolated_latents = torch.cat(interpolated_latents, dim=0)
                coords_pred = model.mlp_coords(interpolated_latents)
                # print(f"coords_pred shape : {coords_pred.shape}")
                coords_pred = coords_pred.view(len(interpolated_latents), N, 3)

            phi_all_, psi_all_ = get_angles(coords_pred, batch_size=len(interpolated_latents))

            phi_all_interpolated.extend(phi_all_)
            psi_all_interpolated.extend(psi_all_)

    phi_all_interpolated = np.asarray(phi_all_interpolated, dtype=np.float32)
    psi_all_interpolated = np.asarray(psi_all_interpolated, dtype=np.float32)
    print(f"len of phi_all_interpolated : {len(phi_all_interpolated)}")
    print(f"len of psi_all_interpolated : {len(psi_all_interpolated)}")

    print("Example phi range:", phi_all_interpolated.min(), phi_all_interpolated.max())
    print("Example psi range:", psi_all_interpolated.min(), psi_all_interpolated.max())

    save_histogram(all_phi=phi_all_interpolated, all_psi = psi_all_interpolated, plot_name=plot_name)
    print("Done with interpolation batches...")


########################## A100s #############################




if __name__ == "__main__":
    
    # with open("/home/binal1/Graphons/implicit_graphon/IGNR/IGNR/Data/Data/full_data_knn_4.pkl",'rb') as f:  
    ppath = os.getcwd()
    model_path = f"{ppath}/Results/without_siren_LD_FPG_/_graph_repr_lr_0.01/full_data_knn_4_chebnet_dim_8_epoch_75_knn_4_lr_0.01.pt"
    model, prog_args  = load_model(model_path)
    model.eval()

    train_loader, test_loader, n_card = get_dataset(prog_args)

    print("FULL DATA RAMACHANDRAN PLOT...")

    ########## Ramachandran for FULL DATA ############
    # ramachandran_full_data(train_loader, prog_args, plot_name = "sample_full_data_")
    print()

    print("RECONSTRUCTION RAMACHANDRAN PLOT...")
    ########## Ramachandran for RECONSTRUCTION ############
    ramachandran_reconstruction(model, train_loader, prog_args, plot_name = "R_no_siren_rc_graph")

    print(f"INTERPOLATION RAMACHANDRAN PLOT...")
    ########## INTERPOLATION ############
    # ramachandran_interpolation(model, train_loader, prog_args, plot_name = "ramachandran_interpolation_training")








    

