import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.explain import Explainer, GNNExplainer

import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model
from train import get_splits
from evaluate import predict, tune_threshold

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH   = "outputs/models/proposed_tgn_xlmr.pt"
CSV_PATH     = "data/processed_reviews.csv"
OUTPUT_DIR   = "outputs/explanations"
MAX_EXPLAIN  = 15
GNN_EPOCHS   = 120
TOP_K_EDGES  = 6

TIME_WINDOW          = timedelta(hours=48)
MIN_GROUP_SIZE       = 3
SIMILARITY_THRESHOLD = 0.70


class HeteroWrapper(torch.nn.Module):
    def __init__(self, model, n_review):
        super().__init__()
        self.model    = model
        self.n_review = n_review

    def forward(self, x_dict, edge_index_dict):
        ids = torch.arange(self.n_review, device=x_dict["review"].device)
        return self.model(x_dict, edge_index_dict, ids)


def coordination_reasoning(df, rev_idx):
    target = df.iloc[rev_idx]
    shop_id   = target["shop_id"]
    t_target  = target["timestamp"]
    window_s  = TIME_WINDOW.total_seconds()

    cluster = df[df["shop_id"] == shop_id].copy()
    cluster = cluster[(cluster["timestamp"] - t_target).abs() <= window_s]

    texts   = cluster["review_text_clean"].fillna("").astype(str).tolist()
    ratings = cluster["rating"].tolist()

    avg_sim = 0.0
    if len(set(texts)) == 1 and len(texts) > 1:
        avg_sim = 1.0
    elif len(texts) >= 2:
        try:
            tfidf = TfidfVectorizer(min_df=1).fit_transform(texts)
            sims  = cosine_similarity(tfidf)
            upper = sims[np.triu_indices(len(texts), k=1)]
            avg_sim = float(np.mean(upper)) if len(upper) else 0.0
        except Exception:
            avg_sim = 0.0

    same_rating = len(set(ratings)) == 1
    group_size  = len(cluster)

    criteria = {
        "min_3_reviews"      : bool(group_size >= MIN_GROUP_SIZE),
        "within_48h"         : True,
        "same_shop"          : True,
        "identical_rating"   : bool(same_rating),
        "cosine_sim>=0.70"   : bool(avg_sim >= SIMILARITY_THRESHOLD),
    }

    reason = {
        "shop_id"        : str(shop_id),
        "group_size"     : int(group_size),
        "avg_cosine_sim" : round(avg_sim, 4),
        "rating"         : int(target["rating"]),
        "all_same_rating": bool(same_rating),
        "criteria_met"   : criteria,
        "is_coordinated_by_rule": all(criteria.values()),
    }
    return reason, cluster


def visualize(rev_idx, shop_idx, product_idx, prob, reason,
              edge_items, save_path):
    G = nx.DiGraph()

    review_node  = f"Review\n{rev_idx}"
    shop_node    = f"Shop\n{shop_idx}"
    product_node = f"Product\n{product_idx}"

    G.add_node(review_node,  ntype="review")
    G.add_node(shop_node,    ntype="shop")
    G.add_node(product_node, ntype="product")

    for src, dst, imp in edge_items:
        G.add_node(src, ntype=G.nodes.get(src, {}).get("ntype", "co_review"))
        G.add_node(dst, ntype=G.nodes.get(dst, {}).get("ntype", "shop"))
        G.add_edge(src, dst, importance=float(imp))

    node_colors, node_sizes = [], []
    for n in G.nodes:
        nt = G.nodes[n].get("ntype", "co_review")
        node_colors.append({
            "review"   : "#e74c3c",
            "shop"     : "#3498db",
            "product"  : "#2ecc71",
            "co_review": "#f39c12",
        }.get(nt, "#f39c12"))
        node_sizes.append(2200 if nt == "review" else 1300)

    pos = nx.spring_layout(G, seed=42, k=0.9)
    pos[review_node] = np.array([0.0, 0.0])

    cmap = plt.get_cmap("YlOrRd")
    imps   = np.array([G.edges[e]["importance"] for e in G.edges]) if G.number_of_edges() else np.array([0.0])
    widths = 1.0 + 6.0 * imps
    ecolors = cmap(0.25 + 0.75 * imps)

    plt.figure(figsize=(9, 6.5))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=7)
    nx.draw_networkx_edges(G, pos, width=list(widths), edge_color=ecolors,
                           arrows=True, arrowsize=16, connectionstyle="arc3,rad=0.05")

    elabels = {e: f"{G.edges[e]['importance']:.2f}" for e in G.edges}
    nx.draw_networkx_edge_labels(G, pos, elabels, font_size=6)

    label_map = {
        "min_3_reviews"   : "at least 3 reviews",
        "within_48h"      : "within 48 hours",
        "same_shop"       : "same shop",
        "identical_rating": "identical rating",
        "cosine_sim>=0.70": "similar text (>=0.70)",
    }
    met = [label_map.get(k, k) for k, v in reason["criteria_met"].items() if v]
    same_rating = "Yes" if reason["all_same_rating"] else "No"

    title = (f"Review {rev_idx}  -  Prediction: COORDINATED "
             f"(confidence {prob*100:.0f}%)")
    line2 = (f"Reviews in burst: {reason['group_size']}    "
             f"Text similarity: {reason['avg_cosine_sim']:.2f}    "
             f"Same rating: {same_rating}")
    line3 = "Criteria met: " + ", ".join(met)
    plt.title(f"{title}\n{line2}\n{line3}", fontsize=10)

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), fraction=0.04, pad=0.02,
                 label="Edge importance (thicker / redder = more influential)")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def run():
    print("=== GNNExplainer ===")
    print(f"Device: {DEVICE}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model_name = ckpt["model_name"]
    graph_path = "data/graph_data_mbert.pt" if "mbert" in model_name.lower() \
                 else "data/graph_data_xlmr.pt"

    data = torch.load(graph_path, weights_only=False)
    print(f"Using graph: {graph_path}")
    print("review :", data["review"].x.shape)

    n_review = ckpt["n_reviews"]
    model = build_model(
        n_review_nodes  = n_review,
        review_feat_dim = ckpt["feat_dim"],
        device          = DEVICE,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    labels = data["review"].y
    _, idx_val, idx_test = get_splits(labels)

    val_probs, val_labels = predict(model, data, idx_val, DEVICE)
    threshold, _ = tune_threshold(val_probs, val_labels)
    print(f"Decision threshold: {threshold}")

    test_probs, _ = predict(model, data, idx_test, DEVICE)
    mask_coord    = test_probs > threshold
    coord_idx     = idx_test.numpy()[mask_coord]
    coord_probs   = test_probs[mask_coord]

    order        = np.argsort(coord_probs)[::-1][:MAX_EXPLAIN]
    explain_ids  = coord_idx[order]
    explain_prob = coord_probs[order]
    print(f"Predicted coordinated in test: {len(coord_idx)} "
          f"-> explaining top {len(explain_ids)}")

    edge_rs = data["review", "posted_in", "shop"].edge_index
    edge_rp = data["review", "reviews", "product"].edge_index
    rev_to_shop    = {int(edge_rs[0, i]): int(edge_rs[1, i]) for i in range(edge_rs.shape[1])}
    rev_to_product = {int(edge_rp[0, i]): int(edge_rp[1, i]) for i in range(edge_rp.shape[1])}

    df = pd.read_csv(CSV_PATH)

    wrapper = HeteroWrapper(model, n_review).to(DEVICE)
    explainer = Explainer(
        model            = wrapper,
        algorithm        = GNNExplainer(epochs=GNN_EPOCHS),
        explanation_type = "model",
        node_mask_type   = "attributes",
        edge_mask_type   = "object",
        model_config     = dict(
            mode       = "binary_classification",
            task_level = "node",
            return_type= "raw",
        ),
    )

    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_explanations = []

    rs_key = ("review", "posted_in", "shop")
    rp_key = ("review", "reviews", "product")

    for i, (rev_idx, prob) in enumerate(zip(explain_ids, explain_prob)):
        rev_idx = int(rev_idx)
        print(f"  [{i+1}/{len(explain_ids)}] Review {rev_idx} | prob={prob:.4f}")

        explanation = explainer(x_dict, edge_index_dict, index=rev_idx)
        edge_mask_dict = explanation.edge_mask_dict

        shop_idx    = rev_to_shop.get(rev_idx, -1)
        product_idx = rev_to_product.get(rev_idx, -1)

        rs_mask = edge_mask_dict[rs_key].detach().cpu().numpy()
        rp_mask = edge_mask_dict[rp_key].detach().cpu().numpy()

        edge_items = []

        own_rs = [j for j in range(edge_rs.shape[1])
                  if int(edge_rs[0, j]) == rev_idx]
        own_rp = [j for j in range(edge_rp.shape[1])
                  if int(edge_rp[0, j]) == rev_idx]
        target_lbl = f"Review\n{rev_idx}"
        for j in own_rs:
            edge_items.append((target_lbl, f"Shop\n{shop_idx}", rs_mask[j]))
        for j in own_rp:
            edge_items.append((target_lbl, f"Product\n{product_idx}", rp_mask[j]))

        co = [(int(edge_rs[0, j]), rs_mask[j]) for j in range(edge_rs.shape[1])
              if int(edge_rs[1, j]) == shop_idx and int(edge_rs[0, j]) != rev_idx]
        co.sort(key=lambda t: t[1], reverse=True)
        for other_rev, imp in co[:TOP_K_EDGES]:
            edge_items.append((f"Review\n{other_rev}", f"Shop\n{shop_idx}", imp))

        if edge_items:
            vals = np.array([e[2] for e in edge_items], dtype=float)
            lo, hi = vals.min(), vals.max()
            norm = (vals - lo) / (hi - lo + 1e-9)
            edge_items = [(s, d, n) for (s, d, _), n in zip(edge_items, norm)]

        reason, _ = coordination_reasoning(df, rev_idx)

        save_path = os.path.join(OUTPUT_DIR, f"explanation_{rev_idx}.png")
        visualize(rev_idx, shop_idx, product_idx, float(prob), reason,
                  edge_items, save_path)

        all_explanations.append({
            "review_idx"      : rev_idx,
            "predicted_prob"  : round(float(prob), 4),
            "true_label"      : int(labels[rev_idx].item()),
            "shop_idx"        : shop_idx,
            "product_idx"     : product_idx,
            "reasoning"       : reason,
            "top_edge_importance": round(float(max(e[2] for e in edge_items)), 4)
                                   if edge_items else 0.0,
            "plot"            : save_path,
        })

    json_path = os.path.join(OUTPUT_DIR, "explanations.json")
    with open(json_path, "w") as f:
        json.dump(all_explanations, f, indent=2)

    print(f"\nSaved {len(all_explanations)} explanations -> {OUTPUT_DIR}/")
    print(f"Saved summary -> {json_path}")
    print("=== Done ===")


if __name__ == "__main__":
    run()
