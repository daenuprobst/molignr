import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

from pathlib import Path
import torch
from utils import *
import numpy as np

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


"""
if __name__ == "__main__":
    
    model_path = "/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/Result/checkpoints/lr_0.01/full_data_knn_4_chebnet_dim_8_epoch_60_knn_4_lr_0.01_best_model.pt"
    model, prog_args  = load_model(model_path)
    train_loader, test_loader, n_card = get_dataset(prog_args)

    steps = prog_args.batch_size  # number of interpolation steps
    N = 2191
    interpolated_latents = []
    for batch in train_loader:
        batch = batch.to(prog_args.device)
        model.eval()
        with torch.no_grad():
            z, _ = model.encode(batch.x, batch.edge_index, batch.batch)
            z1 = z[0]
            z2 = z[4]


            path = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/Result/pdbs/full_atom_0.pdb"
            save_ca_pdb_(batch.x[:N].cpu().detach().numpy(), path=path)

            path = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/Result/pdbs/full_atom_last.pdb"
            save_ca_pdb_(batch.x[(N*15):].cpu().detach().numpy(), path=path)
        
            for alpha in np.linspace(0, 1, steps):
                print(f"alpha : {alpha}")
                interpolated_latent = slerp(z1, z2, t = alpha)
                interpolated_latents.append(interpolated_latent.unsqueeze(0))
            
            interpolated_latents = torch.cat(interpolated_latents, dim=0)

            # interpolated_latents = interpolate_between_latents(z1, z2, steps=steps)
            new_structures = model.mlp_coords(interpolated_latents)

            # print(f"interpolated_latents.shape : {interpolated_latents.shape}")
            # print(f"batch.x.shape : {batch.x.shape}")

            lddt_ = lddt(new_structures.view(prog_args.batch_size, N, 3), batch.x.view(prog_args.batch_size, N, 3))
            tm_score_ = tm_score(batch.x, new_structures)

            print(f"lDDT : {lddt_}")
            print(f"TM-score : {tm_score_}")


            print(f"new_structures.shape : {new_structures.shape}")
            print()

            N = 2191
            
            
            for i in range(steps):
                structure = new_structures[(N*i):N*(i+1),:]

                coords_pred_ = structure.cpu().detach().numpy()[:2191]
                ## overwrite pdb files with predicted coordinates
                pdb_path = "heavy_chain.pdb"
                overwrite_pdbs(pdb_path, coords_pred_, i)

                print(f"structure.shape : {structure.shape}")
                #path = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/Result/pdbs/_slerp_interpolated_full_atom_{i+1}.pdb"
                # save_ca_pdb_(structure.cpu().detach().numpy(), path=path)

        
        for i in range(steps):
            structure = new_structures[(N*i):N*(i+1),:]
            print(f"structure.shape : {structure.shape}")
            path = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/Result/pdbs/_slerp_interpolated_full_atom_{i+1}.pdb"
            save_ca_pdb_(structure.cpu().detach().numpy(), path=path)
        

        print()

        print(f"interpolated_latents.shape : {interpolated_latents.shape}")

        print(f"z.shape : {z.shape}")

        print(f"OKIEEE")
        break

"""
