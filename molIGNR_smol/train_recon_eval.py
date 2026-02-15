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
from smol_gabor_recon_only import cIGNR
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


def get_loaders(dataset):
    if dataset == "freesolv":
        mn_dataset = MoleculeNet("./data/freesolv", name="FreeSolv")

        smiles_list = []
        labels = []

        for data in mn_dataset:
            labels.append(data.y.item())
            smiles_list.append(data.smiles)

        smiles_list, labels = shuffle_together(smiles_list, labels, seed=42)

        dataset_train = MolecularDataset(
            smiles_list[50:], root="./data/freesolv_train", y=labels[50:]
        )

        dataset_test = MolecularDataset(
            smiles_list[:50], root="./data/freesolv_test", y=labels[:50]
        )

        # df = pd.DataFrame(smiles_list, columns=["smiles"])

        # train_idx, valid_idx, test_idx = scaffold_split(df)
        # train_df, valid_df, test_df = (
        #     df.iloc[train_idx],
        #     df.iloc[valid_idx],
        #     df.iloc[test_idx],
        # )

        # dataset_train = MolecularDataset(
        #     train_df["smiles"].tolist(), root="./data/freesolv_train"  # , y=labels[50:]
        # )

        # dataset_test = MolecularDataset(
        #     test_df["smiles"].tolist(), root="./data/freesolv_test"  # , y=labels[:50]
        # )
    elif dataset == "esol":
        mn_dataset = MoleculeNet("./data/esol", name="ESOL")

        smiles_list = []
        labels = []

        for data in mn_dataset:
            labels.append(data.y.item())
            smiles_list.append(data.smiles)

        smiles_list, labels = shuffle_together(smiles_list, labels, seed=42)

        dataset_train = MolecularDataset(
            smiles_list[50:], root="./data/esol_train", y=labels[50:]
        )

        dataset_test = MolecularDataset(
            smiles_list[:50], root="./data/esol_test", y=labels[:50]
        )
    elif dataset == "lipo":
        mn_dataset = MoleculeNet("./data/lipo", name="lipo")

        smiles_list = []
        labels = []

        for data in mn_dataset:
            labels.append(data.y.item())
            smiles_list.append(data.smiles)

        smiles_list, labels = shuffle_together(smiles_list, labels, seed=42)

        dataset_train = MolecularDataset(
            smiles_list[50:], root="./data/lipo_train", y=labels[50:]
        )

        dataset_test = MolecularDataset(
            smiles_list[:50], root="./data/lipo_test", y=labels[:50]
        )
    elif dataset == "toxcast":
        mn_dataset = MoleculeNet("./data/toxcast", name="toxcast")

        smiles_list = []
        labels = []

        for data in mn_dataset:
            labels.append(data.y[0][0].item())
            smiles_list.append(data.smiles)

        smiles_list, labels = shuffle_together(smiles_list, labels, seed=42)

        dataset_train = MolecularDataset(
            smiles_list[50:], root="./data/toxcast_train", y=labels[50:]
        )

        dataset_test = MolecularDataset(
            smiles_list[:50], root="./data/toxcast_test", y=labels[:50]
        )
    elif dataset == "freesolv_small":
        mn_dataset = MoleculeNet("./data/freesolv", name="freesolv")

        smiles_list = []
        labels = []

        for data in mn_dataset:
            labels.append(data.y.item())
            smiles_list.append(data.smiles)

        dataset_train = MolecularDataset(
            smiles_list[:10], root="./data/freesolv_small_train", y=labels[:10]
        )

        dataset_test = MolecularDataset(
            smiles_list[10:15], root="./data/freesolv_small_test", y=labels[10:15]
        )
    elif dataset == "peptides":
        smiles_list = load_csv("~/Downloads/peptides_CPP1708.csv")

        dataset_train = MolecularDataset(smiles_list[:5], root="./data/peptides_train")

        dataset_test = MolecularDataset([smiles_list[20]], root="./data/peptides_test")

    train_loader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: molecular_collate_fn(batch),
        num_workers=12,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: molecular_collate_fn(batch),
        num_workers=12,
    )

    sample_batch, _, _ = next(iter(train_loader))
    n_attr = sample_batch.x.shape[1]

    return train_loader, test_loader, n_attr


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

        if (epoch % 200 == 0 or epoch == 20) and batch_idx == 0 and epoch > 0:
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
    test_loader,
    device,
    M,
    total_epochs=500,
    save_every=50,
    log_name="log",
    save_path="models/model.pt",
    log_wandb=False,
):
    """
    Complete training loop (no validation).

    Scheduler.step() is called ONCE PER EPOCH after training.
    Model is saved periodically every save_every epochs.
    """

    # Setup optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs=total_epochs)

    best_train_loss = float("inf")

    print("Starting training with Cosine Annealing scheduler...")
    print(f"Total epochs: {total_epochs}")
    print(f"Device: {device}")
    print(f"Saving checkpoint every {save_every} epochs")

    for epoch in range(total_epochs):
        train_loss, detailed_loss = train_epoch(
            model, train_loader, optimizer, device, M, epoch
        )

        # Call after each epoch
        scheduler.step()

        # Get current learning rates
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

        test_loss, test_detailed_loss = eval_epoch(model, test_loader, device, M, epoch)

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

    train_loader, test_loader, n_attr = get_loaders("toxcast")

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
        loss_type="diff",
        device=device,
    ).to(device)

    M = 0

    model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        M=M,
        total_epochs=1000,
        save_every=50,  # Save checkpoint every 50 epochs
        save_path="models/model.pt",
        log_name="new_lipo_gabor_diff_exp",
        log_wandb=False,
    )

    # Initialize model
    # model = cIGNR(
    #     n_attr=n_attr,
    #     emb_dim=16,
    #     latent_dim=16,
    #     num_layer=5,
    #     hidden_dims=[256, 256, 256],
    #     n_atom_types=14,
    #     n_bond_types=4,
    #     valences=MolecularDataset.VALENCES,
    #     network_type="gabor",
    #     loss_type="gw",
    #     mmd=False,
    #     device=device,
    # ).to(device)

    # M = 0

    # model = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     M=M,
    #     total_epochs=500,
    #     save_every=50,  # Save checkpoint every 50 epochs
    #     save_path="models/model.pt",
    # log_name="new_freesolv_gabor_gw",
    #     log_wandb=False,
    # )

    # # Initialize model
    # model = cIGNR(
    #     n_attr=n_attr,
    #     emb_dim=16,
    #     latent_dim=16,
    #     num_layer=5,
    #     hidden_dims=[256, 256, 256],
    #     n_atom_types=14,
    #     n_bond_types=4,
    #     valences=MolecularDataset.VALENCES,
    #     network_type="siren",
    #     loss_type="gw",
    #     mmd=False,
    #     device=device,
    # ).to(device)

    # M = 0

    # model = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     M=M,
    #     total_epochs=500,
    #     save_every=50,  # Save checkpoint every 50 epochs
    #     save_path="models/model.pt",
    #     log_name="new_freesolv_siren_gw",
    #     log_wandb=False,
    # )

    # train_loader, test_loader, n_attr = get_loaders("esol")

    # # Initialize model
    # model = cIGNR(
    #     n_attr=n_attr,
    #     emb_dim=16,
    #     latent_dim=16,
    #     num_layer=5,
    #     hidden_dims=[256, 256, 256],
    #     n_atom_types=14,
    #     n_bond_types=4,
    #     valences=MolecularDataset.VALENCES,
    #     network_type="gabor",
    #     loss_type="diff",
    #     mmd=False,
    #     device=device,
    # ).to(device)

    # M = 0

    # model = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     M=M,
    #     total_epochs=500,
    #     save_every=50,  # Save checkpoint every 50 epochs
    #     save_path="models/model.pt",
    #     log_name="new_esol_gabor_diff_exp",
    #     log_wandb=False,
    # )

    # model = cIGNR(
    #     n_attr=n_attr,
    #     emb_dim=16,
    #     latent_dim=16,
    #     num_layer=5,
    #     hidden_dims=[256, 256, 256],
    #     n_atom_types=14,
    #     n_bond_types=4,
    #     valences=MolecularDataset.VALENCES,
    #     network_type="siren",
    #     loss_type="diff",
    #     mmd=False,
    #     device=device,
    # ).to(device)

    # M = 0

    # model = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     M=M,
    #     total_epochs=500,
    #     save_every=50,  # Save checkpoint every 50 epochs
    #     save_path="models/model.pt",
    #     log_name="new_esol_siren_diff",
    #     log_wandb=False,
    # )

    # model = cIGNR(
    #     n_attr=n_attr,
    #     emb_dim=16,
    #     latent_dim=16,
    #     num_layer=5,
    #     hidden_dims=[256, 256, 256],
    #     n_atom_types=14,
    #     n_bond_types=4,
    #     valences=MolecularDataset.VALENCES,
    #     network_type="gabor",
    #     loss_type="gw",
    #     mmd=False,
    #     device=device,
    # ).to(device)

    # M = 0

    # model = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     M=M,
    #     total_epochs=500,
    #     save_every=50,  # Save checkpoint every 50 epochs
    #     save_path="models/model.pt",
    #     log_name="new_esol_gabor_gw",
    #     log_wandb=False,
    # )

    # model = cIGNR(
    #     n_attr=n_attr,
    #     emb_dim=16,
    #     latent_dim=16,
    #     num_layer=5,
    #     hidden_dims=[256, 256, 256],
    #     n_atom_types=14,
    #     n_bond_types=4,
    #     valences=MolecularDataset.VALENCES,
    #     network_type="siren",
    #     loss_type="gw",
    #     mmd=False,
    #     device=device,
    # ).to(device)

    # M = 0

    # model = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     M=M,
    #     total_epochs=500,
    #     save_every=50,  # Save checkpoint every 50 epochs
    #     save_path="models/model.pt",
    #     log_name="new_esol_siren_gw",
    #     log_wandb=False,
    # )


if __name__ == "__main__":
    main()
