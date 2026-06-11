import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import HeteroData
from model import build_model

SEED = 42
LR = 1e-4
BATCH_SIZE = 64
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_CSV = "data/processed_reviews.csv"
OUTPUT_DIR = "outputs/models"
MAX_LENGTH = 128

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#spliting
def get_splits(labels):

    idx = np.arange(len(labels))
    y = labels.numpy()

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED
    )

    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED
    )

    print(
        f"train={len(idx_train)} "
        f"val={len(idx_val)} "
        f"test={len(idx_test)}"
    )

    return (
        torch.tensor(idx_train, dtype=torch.long),
        torch.tensor(idx_val, dtype=torch.long),
        torch.tensor(idx_test, dtype=torch.long),
    )

#weight class
def get_pos_weight(labels):

    y = labels.numpy()

    weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=y
    )

    pos_weight = torch.tensor(
        [weights[1] / weights[0]],
        dtype=torch.float
    )

    print(f"pos weight = {pos_weight.item():.2f}")

    return pos_weight

#train
def train_epoch(
    model,
    data,
    idx_train,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0
    batch_count = 0

    perm = idx_train[
        torch.randperm(len(idx_train))
    ]

    for start in range(
        0,
        len(perm),
        BATCH_SIZE
    ):

        batch_idx = perm[
            start:start+BATCH_SIZE
        ].to(DEVICE)

        optimizer.zero_grad()

        x_dict = {
            "review":
                data["review"].x.to(DEVICE),

            "shop":
                data["shop"].x.to(DEVICE),

            "product":
                data["product"].x.to(DEVICE),
        }

        edge_index_dict = {
            k: v.to(DEVICE)
            for k, v in data.edge_index_dict.items()
        }

        logits = model(
            x_dict,
            edge_index_dict,
            batch_idx
        )

        labels = (
            data["review"]
            .y[batch_idx.cpu()]
            .float()
            .to(DEVICE)
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        model.update_memory(
            batch_idx,
            x_dict,
            edge_index_dict
        )

        total_loss += loss.item()

        batch_count += 1

    return total_loss / batch_count

#evaluate
@torch.no_grad()
def evaluate(model, data, idx):

    model.eval()

    x_dict = {
        "review":
            data["review"].x.to(DEVICE),

        "shop":
            data["shop"].x.to(DEVICE),

        "product":
            data["product"].x.to(DEVICE),
    }

    edge_index_dict = {
        k: v.to(DEVICE)
        for k, v in data.edge_index_dict.items()
    }

    logits = model(
        x_dict,
        edge_index_dict,
        idx.to(DEVICE)
    )

    probs = torch.sigmoid(logits).cpu()

    preds = (probs > 0.5).long()

    labels = data["review"].y[idx]

    return probs, preds, labels


#train
def train_model(
    model_name,
    data,
    use_memory=True
):

    print("\n" + "=" * 50)
    print(f"training {model_name}")
    print("=" * 50)

    labels = data["review"].y

    n_reviews = labels.shape[0]

    idx_train, idx_val, idx_test = get_splits(labels)

    pos_weight = get_pos_weight(labels).to(DEVICE)

    model = build_model(
        n_review_nodes=n_reviews,
        review_feat_dim=data["review"].x.shape[1],
        device=DEVICE
    )

    if not use_memory:
        model.memory.reset()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5
    )

    best_val_loss = float("inf")

    best_state = None

    for epoch in range(1, EPOCHS + 1):

        if not use_memory:
            model.reset_memory()

        train_loss = train_epoch(
            model,
            data,
            idx_train,
            optimizer,
            criterion
        )

        with torch.no_grad():

            x_dict = {
                "review":
                    data["review"].x.to(DEVICE),

                "shop":
                    data["shop"].x.to(DEVICE),

                "product":
                    data["product"].x.to(DEVICE),
            }

            edge_index_dict = {
                k: v.to(DEVICE)
                for k, v in data.edge_index_dict.items()
            }

            val_logits = model(
                x_dict,
                edge_index_dict,
                idx_val.to(DEVICE)
            )

            val_loss = criterion(
                val_logits,
                data["review"]
                .y[idx_val.cpu()]
                .float()
                .to(DEVICE)
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch % 10 == 0:

            print(
                f"epoch {epoch:3d} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f}"
            )

    model.load_state_dict(best_state)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}.pt"
    )

    torch.save({
        "model_state": best_state,
        "idx_test": idx_test,
        "model_name": model_name,
        "n_reviews": n_reviews,
        "feat_dim": data["review"].x.shape[1],
    }, save_path)

    print(f"saved -> {save_path}")

    return model

# main
def run():

    print("training all models")
    print(f"device = {DEVICE}")

    set_seed(SEED)

    configs = [
        (
            "proposed_tgn_xlmr",
            "data/graph_data_xlmr.pt",
            True
        ),
        (
            "baseline1_static_gnn",
            "data/graph_data_xlmr.pt",
            False
        ),
        (
            "baseline2_tgn_mbert",
            "data/graph_data_mbert.pt",
            True
        ),
        (
            "baseline3_tgn_no_xai",
            "data/graph_data_xlmr.pt",
            True
        ),
    ]


    for model_name, graph_path, use_memory in configs:

        save_path = os.path.join(
            OUTPUT_DIR,
            f"{model_name}.pt"
        )

        if os.path.exists(save_path):

            print(
                f"\n[SKIP] {model_name} already exists"
            )

            continue

        print(f"\nUsing graph: {graph_path}")

        data = torch.load(
            graph_path,
            weights_only=False
        )

        train_model(
            model_name,
            data,
            use_memory
        )


if __name__ == "__main__":
    run()
