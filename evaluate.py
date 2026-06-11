import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model
from train import get_splits

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR  = "outputs/models"
RESULTS_DIR = "outputs/results"


@torch.no_grad()
def predict(model, data, idx, device):
    model.eval()
    x_dict = {
        "review" : data["review"].x.to(device),
        "shop"   : data["shop"].x.to(device),
        "product": data["product"].x.to(device),
    }
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}

    logits = model(x_dict, edge_index_dict, idx.to(device))
    probs  = torch.sigmoid(logits).cpu().numpy()
    labels = data["review"].y[idx.cpu()].numpy()
    return probs, labels


def tune_threshold(val_probs, val_labels):
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (val_probs > t).astype(int)
        f1 = f1_score(val_labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2), round(best_f1, 4)


def compute_metrics(labels, preds, probs, model_name, threshold):
    f1        = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    metrics = {
        "model"    : model_name,
        "threshold": threshold,
        "f1"       : round(f1, 4),
        "precision": round(precision, 4),
        "recall"   : round(recall, 4),
        "auc_roc"  : round(auc, 4),
    }

    print(f"\n  Model     : {model_name}")
    print(f"  Threshold : {threshold}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(labels, preds,
          target_names=["Normal", "Coordinated"], zero_division=0))

    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix:\n{cm}")

    return metrics


def run():
    print("=== Evaluation ===")
    print(f"Device: {DEVICE}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_metrics = []

    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".pt"):
            continue

        model_name = fname.replace(".pt", "")
        if "mbert" in model_name.lower():
            graph_path = "data/graph_data_mbert.pt"
        else:
            graph_path = "data/graph_data_xlmr.pt"

        data = torch.load(graph_path, weights_only=False)

        print(f"\nUsing graph: {graph_path}")
        print("review :", data['review'].x.shape)
        ckpt = torch.load(os.path.join(MODELS_DIR, fname),
                          map_location=DEVICE, weights_only=False)

        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")

        model = build_model(
            n_review_nodes  = ckpt["n_reviews"],
            review_feat_dim = ckpt["feat_dim"],
            device          = DEVICE,
        )
        model.load_state_dict(ckpt["model_state"])

        labels_all = data["review"].y
        _, idx_val, idx_test = get_splits(labels_all)

        val_probs, val_labels = predict(model, data, idx_val, DEVICE)
        threshold, val_f1 = tune_threshold(val_probs, val_labels)
        print(f"  best threshold = {threshold} (val F1 = {val_f1})")

        test_probs, test_labels = predict(model, data, idx_test, DEVICE)

        print("\n  Test probability distribution")
        print("   min/mean/max :",
              round(float(test_probs.min()), 4),
              round(float(test_probs.mean()), 4),
              round(float(test_probs.max()), 4))

        test_preds = (test_probs > threshold).astype(int)
        metrics = compute_metrics(test_labels, test_preds, test_probs,
                                  model_name, threshold)
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    csv_path  = os.path.join(RESULTS_DIR, "metrics.csv")
    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(df.to_string(index=False))
    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {json_path}")
    print("Done")


if __name__ == "__main__":
    run()
