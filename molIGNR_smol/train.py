import time
import random
from collections import defaultdict
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import math
from tqdm import tqdm
import wandb
from data_loader import (
    load_molnet_dataset,
    load_adme_dataset,
    MolecularDataset,
    scaffold_split,
)
import pandas as pd

from torch.utils.data import DataLoader

from torch_geometric.datasets import ZINC, QM7b, MoleculeNet
from tqdm import tqdm

# --- Model ---
from smol_gabor import cIGNR
from molecular_dataset import MolecularDataset, molecular_collate_fn
from helpers import plot_adj_matrices, load_csv, molecular_graph_to_smiles

from rdkit import Chem

import csv
import os


def shuffle_together(*lists, seed=None):
    if not lists:
        return []

    length = len(lists[0])
    if not all(len(lst) == length for lst in lists):
        raise ValueError("All lists must have the same length")

    indices = list(range(length))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)

    return tuple([lst[i] for i in indices] for lst in lists)


def log_test_results(
    epoch,
    train_loss,
    test_loss,
    test_detailed_loss,
    filepath="logs/test_metrics.csv",
):
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ["epoch", "train_loss", "test_loss"] + list(
                test_detailed_loss.keys()
            )
            writer.writerow(header)

        row = [epoch, train_loss, test_loss] + [
            test_detailed_loss[k] for k in test_detailed_loss.keys()
        ]
        writer.writerow(row)


def filter_molecules_by_size(smiles_list, target_size, n_max=100):
    filtered_smiles = []

    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        num_atoms = mol.GetNumAtoms()
        if num_atoms == target_size:
            filtered_smiles.append(smi)
            if len(filtered_smiles) == 100:
                break

    return filtered_smiles


def get_loader():
    # train_loader, valid_loader, test_loader = load_molnet_dataset(
    #     "freesolv", "data/moleculenet/freesolv.csv.xz", "data/tmp"
    # )

    # sample_batch, _, _ = next(iter(train_loader))
    # n_attr = sample_batch.x.shape[1]

    # return train_loader, n_attr

    mn_dataset = MoleculeNet("./data/freesolv", name="freesolv")
    smiles_list = []
    labels = []

    for data in mn_dataset:
        labels.append(data.y.item())
        smiles_list.append(data.smiles)

    labels = None
    smiles_list = [
        # "CCN(CC)C(=O)[C@H]1CN([C@@H]2CC3=CNC4=CC=CC(=C34)C2=C1)C",
        # "CC(C)[C@@H](C(=O)O)N",
        "N1CNCCC1",
        "N1CCCCC1",
        "C1CCCC1",
        "C1CNCC1",
        "C1CNCN1",
        "C1CCCCC1",
        "CCCCC",
        "CCCNC",
        "CNCNC",
        "CCCCCC",
        "CCNCCC",
        "CCNCNC",
        "CCC1CC(=0)CC1C",
        "Cc1cccc1C",
        "CC1CCCCC1",
        "CCC1CCCCC1CC",
        "OC(=O)c1cccnc1",
        "O=C(OC)C=CC(=O)OC",
        "NC(Cc1ccccc1)(C)C",
        "CC(C)CC(CC(=O)O)CN",
        "NCC(O)c1cc(O)c(O)cc1",
        "O=C1C(O)=C(N(C=C1)C)C",
    ]

    # # smiles_list = ["O=C(NCCCN(O)CCCNCCCNCCCCNCCCN)Cc2c1ccccc1[nH]c2"]

    # smiles_list, labels = load_csv(
    #     "~/Code/molsetrep/data/adme/ADME_rPPB_train.csv", label_column="activity"
    # )
    smiles_list = load_csv("~/Downloads/peptides_CPP1708.csv")

    smiles_list = smiles_list[22:25]
    labels = None

    # Create dataset
    dataset = MolecularDataset(smiles_list, root="./data/peptides", y=labels)

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: molecular_collate_fn(batch),
        num_workers=12,
    )

    sample_batch, _, _ = next(iter(train_loader))
    n_attr = sample_batch.x.shape[1]

    return train_loader, n_attr


def get_optimizer_and_scheduler(model, total_epochs, warmup_epochs=10, base_lr=1e-3):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8
    )

    # Learning rate schedule:  warmup + cosine annealing
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        else:
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, device, M, epoch, clip_grad=True):
    model.train()
    total_loss = 0
    num_batches = 0

    all_losses = defaultdict(list)

    for batch_idx, (data, targets, indices) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        loss, z, reconstructions, loss_components = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            targets=targets,
            M=M,
            epoch=epoch,
        )

        if (epoch % 50 == 0 or epoch == 20) and batch_idx == 0 and epoch > 0:
            s = []
            for idx in range(len(reconstructions["adj"])):
                atom_types = None
                if reconstructions["atom_types"]:
                    atom_types = (
                        reconstructions["atom_types"][idx].cpu().detach().numpy()
                    )

                bond_types = None
                if reconstructions["bond_types"]:
                    bond_types = (
                        reconstructions["bond_types"][idx].cpu().detach().numpy()
                    )
                smiles = molecular_graph_to_smiles(
                    reconstructions["adj"][idx].cpu().detach().numpy(),
                    atom_types,
                    MolecularDataset.ATOM_TYPES,
                    bond_types,
                    path=f"data/imgs/struct_{idx}.png",
                )

                s.append(smiles)

                plot_adj_matrices(
                    reconstructions["adj"][idx],
                    targets["adj"][idx],
                    f"./data/imgs/{idx}.png",
                )

            print(s)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        for key, value in loss_components.items():
            all_losses[key].append(value.item())

        total_loss += loss.item()
        num_batches += 1

    average_losses = {
        key: sum(values) / num_batches for key, values in all_losses.items()
    }
    avg_loss = total_loss / num_batches
    return avg_loss, average_losses


@torch.no_grad()
def eval_epoch(model, loader, device, M, epoch):
    model.eval()

    total_loss = 0.0
    num_batches = 0
    all_losses = defaultdict(list)

    for batch_idx, (data, targets, indices) in enumerate(loader):
        data = data.to(device)

        loss, z, reconstructions, loss_components = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            targets=targets,
            M=M,
            epoch=epoch,
        )

        for key, value in loss_components.items():
            all_losses[key].append(value.item())

        total_loss += loss.item()
        num_batches += 1

    average_losses = {
        key: sum(values) / num_batches for key, values in all_losses.items()
    }
    avg_loss = total_loss / num_batches

    return avg_loss, average_losses


def train_model(
    model,
    train_loader,
    device,
    M,
    total_epochs=500,
    save_every=50,
    log_name="log",
    save_path="models/model.pt",
    log_wandb=False,
):
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs=total_epochs)

    best_train_loss = float("inf")

    print("Starting training with Cosine Annealing scheduler...")
    print(f"Total epochs: {total_epochs}")
    print(f"Device: {device}")
    print(f"Saving checkpoint every {save_every} epochs")

    total_time = 0
    n = 0
    for epoch in range(total_epochs):
        start = time.perf_counter()
        train_loss, detailed_loss = train_epoch(
            model, train_loader, optimizer, device, M, epoch
        )
        end = time.perf_counter()
        total_time += end - start

        n += 1
        print(f"Avg time: {(total_time / n) * 1000:.2f}ms, Epoch: {n}")

        scheduler.step()

        current_lrs = [group["lr"] for group in optimizer.param_groups]

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"LRs: {[f'{lr:.2e}' for lr in current_lrs]}"
            )

        if log_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "lr_encoder": current_lrs[0],
                    "lr_decoder": current_lrs[1],
                    "lr_fourier": current_lrs[2],
                }
            )

        # Save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_train_loss": best_train_loss,
                },
                save_path.replace(".pt", "_best.pt"),
            )
            print(f"  → Saved best model (train_loss: {train_loss:.4f})")

        test_loss, test_detailed_loss = eval_epoch(
            model, train_loader, device, M, epoch
        )

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f}"
        )

        log_test_results(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            test_detailed_loss=test_detailed_loss,
            filepath=f"logs/{log_name}.csv",
        )

        if epoch % 10 == 0:
            # print("Train losses:", train_detailed_loss)
            print("--> Test losses  :", test_detailed_loss)
            print("--> Train losses :", detailed_loss)

    print(f"\nTraining complete! Best train loss: {best_train_loss:.4f}")
    return model


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, n_attr = get_loader()

    # Initialize model
    model = cIGNR(
        n_attr=n_attr,
        emb_dim=16,
        latent_dim=16,
        num_layer=5,
        hidden_dims=[256, 256, 256],
        n_atom_types=14,
        n_bond_types=4,
        valences=MolecularDataset.VALENCES,
        network_type="gabor",
        loss_type="sw",
        device=device,
    ).to(device)

    M = 0

    model = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        M=M,
        total_epochs=500,
        save_every=50,  # Save checkpoint every 50 epochs
        save_path="models/model.pt",
        log_name="experiment",
        log_wandb=False,
    )


if __name__ == "__main__":
    main()
