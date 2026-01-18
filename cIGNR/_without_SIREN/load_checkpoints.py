import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device : {device}")

from pathlib import Path
import torch

import sys

from models.model import *
from data_ import *
from metrics import *

from utils import *
from a100 import *
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
    path = f"/home/binal1/Graphons/cIGNR_final/_without_SIREN/Results/checkpoints/graph/lr_0.01/epoch_0_dim_8_knn_4_lr_0.01_best_model.pt"
    model, args = load_model(path)

    train_loader, test_loader, _ = get_dataset(args, shuffle=False)

    print("running ....")

    #visualize_latents(args, model, train_loader, epoch = None)
    print("visualization done....")
    # test(args, train_loader, model)
 
    #ramachandran_full_data(train_loader, args, plot_name = "full_data_histogram", N =2191)
    #ramachandran_reconstruction(model, train_loader, args, plot_name = f"R_no_siren_rc_{args.repr}")
    #ramachandran_interpolation(model, train_loader, args, plot_name = f"R_no_siren_interpolation_{args.repr}", N=2191)

    a100_plots(model, train_loader, args, gpcrdb_pdb = "heavy_chain_GPCRDB.pdb", plot_name = "hist_a100")

















"""

def overwrite_pdbs(pdb_path, pred_coords):
    u = mda.Universe(pdb_path)

    print(f"u: {u}")
    atoms = u.atoms

    print(f"atoms : {atoms.positions}")
    print(f"atoms.shape : {len(atoms.positions)} - {len(atoms.positions[0])}")

    atoms.positions = pred_coords
    u.atoms.write("heavy_chain_updates.pdb")

    print(f"Wrote updated pdb file with predicted coordinates.")



def update_coords(pdb_path, pred_coords, replace_records = ("ATOMS", "HETATM")):
    i = 0
    out_lines= [] 

    with open(pdb_path, "r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in replace_records:
                if i>= len(pred_coords):
                    raise ValueError(f"More ATOM/HETATM lines than coordinates (idx={i}).")
                x,y,z = pred_coords[i]  
                print(f"pred_coords : {x},{y},{z}")
                
                new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                line = line[:30] + new_xyz + line[54:]
                i += 1
            out_lines.append(line)

            
            out_pdb = "heavy_chain_updates.pdb"
            Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
            with open(out_pdb, "w") as f:
                f.writelines(out_lines)


def update_pdb_coords(pdb_in, coords_in, pdb_out):
    coords = coords_in

    lines = Path(pdb_in).read_text().splitlines(True)  # keep newlines
    xyz_lines_idx = [i for i, L in enumerate(lines) if L.startswith(("ATOM  ", "HETATM"))]

    if len(xyz_lines_idx) != len(coords):
        raise ValueError(f"#coords ({len(coords)}) != #ATOM/HETATM lines ({len(xyz_lines_idx)})")

    # write back XYZ with 8.3 formatting, padding if line is short
    for k, i in enumerate(xyz_lines_idx):
        L = lines[i]
        if len(L) < 54:
            L = L.rstrip("\n")
            L = L + " " * (54 - len(L)) + "\n"  # ensure we have room for XYZ fields
        x, y, z = coords[k]
        xstr = f"{x:8.3f}"
        ystr = f"{y:8.3f}"
        zstr = f"{z:8.3f}"
        # replace only XYZ slices
        L = L[:30] + xstr + ystr + zstr + L[54:]
        lines[i] = L

    Path(pdb_out).write_text("".join(lines))
    print(f"Updated PDB written to {pdb_out}")

"""