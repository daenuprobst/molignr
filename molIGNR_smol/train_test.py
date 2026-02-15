import math
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from scipy.stats import pearsonr

from data_loader import load_molnet_dataset, load_adme_dataset
from molecular_dataset import MolecularDataset
from smol_gabor_recon_only import cIGNR
from encoder_only import cIGNR as EncoderOnly


def compute_class_weights(loader, device):
    ys = []
    for data, _, _ in loader:
        ys.append(data.y)
    y = torch.cat(ys).float()

    n_pos = y.sum().item()
    n_neg = y.numel() - n_pos

    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], device=device)

    print(f"\nClass distribution:")
    print(f"  Negative: {n_neg} ({100*n_neg/(n_pos+n_neg):.1f}%)")
    print(f"  Positive: {n_pos} ({100*n_pos/(n_pos+n_neg):.1f}%)")
    print(f"  pos_weight: {pos_weight.item():.3f}")

    return pos_weight


def get_optimizer_and_scheduler(
    model, total_epochs, warmup_epochs=10, base_lr=1e-4, property_lr=1e-3
):
    # Separate parameters into groups
    property_head_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "property_head" in name:
            property_head_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr},
            {"params": property_head_params, "lr": property_lr},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, device, M, epoch, pos_weight=None):
    model.train()
    total_loss = 0.0
    total_prop_loss = 0.0
    total_recon_loss = 0.0
    total_chem_loss = 0.0

    for data, targets, _ in loader:
        data = data.to(device)
        targets["property"] = data.y

        if pos_weight is not None:
            targets["class_weights"] = pos_weight

        optimizer.zero_grad()

        # loss, z, reconstructions, loss_components = model(
        #     x=data.x,
        #     edge_index=data.edge_index,
        #     batch=data.batch,
        #     targets=targets,
        #     M=M,
        #     epoch=epoch,
        # )

        loss, z, reconstructions, loss_components = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            targets=targets,
            M=M,
            epoch=epoch,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += loss_components["reconstruction"]
        total_chem_loss += loss_components["chemistry"]
        total_prop_loss += loss_components["property"]

    return (
        total_loss / len(loader),
        total_recon_loss / len(loader),
        total_chem_loss / len(loader),
        total_prop_loss / len(loader),
    )


@torch.no_grad()
def predict_properties(model, loader, device):
    model.eval()
    preds, labels = [], []

    for data, _, _ in loader:
        data = data.to(device)

        loss, z, reconstructions, y_pred = model.sample(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            M=0,
            return_property=True,
        )

        if isinstance(y_pred, dict):
            pred_values = y_pred["probabilities"].cpu().numpy()
        else:
            pred_values = y_pred.cpu().numpy()

        preds.append(pred_values)
        labels.append(data.y.cpu().numpy())

    return np.concatenate(preds), np.concatenate(labels)


def evaluate(y_true, y_pred, task_type="regression"):
    if task_type == "classification":
        return evaluate_classification(y_true, y_pred)
    else:
        return evaluate_regression(y_true, y_pred)


def evaluate_regression(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearsonr": pearsonr(y_true, y_pred)[0],
    }


def evaluate_classification(y_true, y_pred_probs):
    y_true = y_true.squeeze()
    y_pred_probs = y_pred_probs.squeeze()

    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_probs),
    }

    return metrics


def train_with_evaluation(
    model,
    train_loader,
    valid_loader,
    test_loader,
    device,
    M=0,
    total_epochs=500,
    eval_every=1,
    save_dir="models",
):

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs)

    pos_weight = None
    if model.prediction_task == "classification":
        pos_weight = compute_class_weights(train_loader, device)

    _, y_train = predict_properties(model, train_loader, device)

    if model.prediction_task == "classification":
        best_val = 0.0
        maximize_metric = True
        primary_metric = "roc_auc"
    else:
        best_val = float("inf")
        maximize_metric = False
        primary_metric = "rmse"

    best_test = None

    for epoch in range(total_epochs):
        train_loss, train_recon_loss, train_chem_loss, train_prop_loss = train_epoch(
            model, train_loader, optimizer, device, M, epoch, pos_weight
        )
        scheduler.step()

        print(f"\nEpoch {epoch:04d}")
        print(
            f"  Train loss: {train_loss:.4f}, Reconstruction: {train_recon_loss:4f}, Chemistry: {train_chem_loss:4f}, Property: {train_prop_loss:4f}"
        )

        if epoch % eval_every != 0:
            continue

        y_val_pred, y_val = predict_properties(model, valid_loader, device)
        y_test_pred, y_test = predict_properties(model, test_loader, device)

        val_metrics = evaluate(y_val, y_val_pred, model.prediction_task)
        test_metrics = evaluate(y_test, y_test_pred, model.prediction_task)

        print("  Validation:", val_metrics)
        print("  Test:      ", test_metrics)

        val_score = val_metrics[primary_metric]
        test_score = test_metrics[primary_metric]

        is_best = False
        if epoch > 20:
            if maximize_metric:
                is_best = val_score > best_val
            else:
                is_best = val_score < best_val

        if is_best:
            best_val = val_score
            best_test = test_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_metrics": val_metrics,
                },
                Path(save_dir) / "model_best.pt",
            )
            print("  → Saved new best model")

        print("Best validation score:", best_val)
        print("Best test score:", best_test)

    print("\nTraining complete.")
    print("Best validation score:", best_val)
    print("Best test score:", best_test)

    return best_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_name = "lipo"
    net = "gabor"
    task_type = "regression"
    train_loader, valid_loader, test_loader = load_molnet_dataset(
        data_name,
        f"data/moleculenet/{data_name}.csv.xz",
        f"data/tmp_{data_name}",
    )

    sample_batch, _, _ = next(iter(train_loader))
    n_attr = sample_batch.x.shape[1]

    with open(f"benchmark_results/new_{data_name}_{net}_gw.txt", "w+") as f:
        for i in range(5):
            model = cIGNR(
                n_attr=n_attr,
                emb_dim=16,
                latent_dim=16,
                num_layer=3,
                hidden_dims=[256, 256, 256],
                n_atom_types=14,
                n_bond_types=4,
                valences=MolecularDataset.VALENCES,
                network_type=net,
                loss_type="gw",
                predict_property=True,
                device=device,
            ).to(device)

            best_test = train_with_evaluation(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                device=device,
                M=0,
                total_epochs=1000,
                eval_every=1,
                save_dir="models",
            )

            f.write(f"{best_test}\n")
            f.flush()

    # data_name = "esol"
    # net = "gabor"
    # task_type = "regression"
    # train_loader, valid_loader, test_loader = load_molnet_dataset(
    #     data_name,
    #     f"data/moleculenet/{data_name}.csv.xz",
    #     f"data/tmp_{data_name}",
    # )

    # sample_batch, _, _ = next(iter(train_loader))
    # n_attr = sample_batch.x.shape[1]

    # with open(f"benchmark_results/new_{data_name}_{net}_gw.txt", "w+") as f:
    #     for i in range(3):
    #         model = cIGNR(
    #             n_attr=n_attr,
    #             emb_dim=16,
    #             latent_dim=16,
    #             num_layer=3,
    #             hidden_dims=[256, 256, 256],
    #             n_atom_types=14,
    #             n_bond_types=4,
    #             valences=MolecularDataset.VALENCES,
    #             network_type=net,
    #             loss_type="gw",
    #             predict_property=True,
    #             device=device,
    #         ).to(device)

    #         best_test = train_with_evaluation(
    #             model=model,
    #             train_loader=train_loader,
    #             valid_loader=valid_loader,
    #             test_loader=test_loader,
    #             device=device,
    #             M=0,
    #             total_epochs=500,
    #             eval_every=1,
    #             save_dir="models",
    #         )

    #         f.write(f"{best_test}\n")
    #         f.flush()

    # data_name = "freesolv"
    # net = "siren"
    # task_type = "regression"
    # train_loader, valid_loader, test_loader = load_molnet_dataset(
    #     data_name,
    #     f"data/moleculenet/{data_name}.csv.xz",
    #     f"data/tmp_{data_name}",
    # )

    # sample_batch, _, _ = next(iter(train_loader))
    # n_attr = sample_batch.x.shape[1]

    # with open(f"benchmark_results/new_{data_name}_{net}_diff.txt", "w+") as f:
    #     for i in range(5):
    #         model = cIGNR(
    #             n_attr=n_attr,
    #             emb_dim=16,
    #             latent_dim=16,
    #             num_layer=3,
    #             hidden_dims=[256, 256, 256],
    #             n_atom_types=14,
    #             n_bond_types=4,
    #             valences=MolecularDataset.VALENCES,
    #             network_type=net,
    #             loss_type="diff",
    #             predict_property=True,
    #             device=device,
    #         ).to(device)

    #         best_test = train_with_evaluation(
    #             model=model,
    #             train_loader=train_loader,
    #             valid_loader=valid_loader,
    #             test_loader=test_loader,
    #             device=device,
    #             M=0,
    #             total_epochs=200,
    #             eval_every=1,
    #             save_dir="models",
    #         )

    #         f.write(f"{best_test}\n")
    #         f.flush()


if __name__ == "__main__":
    main()
