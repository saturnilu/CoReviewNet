import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData

INPUT_PATH    = "data/processed_reviews.csv"
OUTPUT_GRAPH  = "data/graph_data.pt"
OUTPUT_MAPS   = "data/node_mappings.pt"


# Feature helpers 

def make_review_features(df: pd.DataFrame) -> torch.Tensor:
    rating_norm    = (df["rating"].values - 1) / 4.0               # 0..1
    ts_norm        = (df["timestamp"].values - df["timestamp"].min()) / \
                     (df["timestamp"].max() - df["timestamp"].min() + 1e-9)
    text_len_norm  = df["review_text_clean"].str.len().values / \
                     df["review_text_clean"].str.len().max()
    feats = np.stack([rating_norm, ts_norm, text_len_norm], axis=1)
    return torch.tensor(feats, dtype=torch.float)


def make_shop_features(df: pd.DataFrame, shop_map: dict) -> torch.Tensor:
    shop_stats = df.groupby("shop_id").agg(
        review_count=("review_id", "count"),
        avg_rating=("rating", "mean"),
        coord_ratio=("coordinated", "mean"),
    )  

    n_shops   = len(shop_map)
    feats     = np.zeros((n_shops, 3), dtype=np.float32)
    max_count = shop_stats["review_count"].max()

    for sid, row in shop_stats.iterrows():
        idx = shop_map[str(sid)]
        feats[idx, 0] = row["review_count"] / max_count
        feats[idx, 1] = (row["avg_rating"] - 1) / 4.0
        feats[idx, 2] = row["coord_ratio"]

    return torch.tensor(feats, dtype=torch.float)


def make_product_features(df: pd.DataFrame, product_map: dict) -> torch.Tensor:
    prod_stats = df.groupby("product_id").agg(
        review_count=("review_id", "count"),
        avg_rating=("rating", "mean"),
        avg_price=("product_price", "mean"),
        coord_ratio=("coordinated", "mean"),
    )

    n_products = len(product_map)
    feats      = np.zeros((n_products, 4), dtype=np.float32)
    max_count  = prod_stats["review_count"].max()
    max_price  = prod_stats["avg_price"].max()

    for pid, row in prod_stats.iterrows():
        idx = product_map[str(pid)]
        feats[idx, 0] = row["review_count"] / max_count
        feats[idx, 1] = (row["avg_rating"] - 1) / 4.0
        feats[idx, 2] = row["avg_price"] / (max_price + 1e-9)
        feats[idx, 3] = row["coord_ratio"]

    return torch.tensor(feats, dtype=torch.float)


# Graph builder

def build_graph(input_path: str = INPUT_PATH) -> HeteroData:
    print("=== Graph Construction ===")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows")

    # Node index mappings 
    review_map  = {rid: i for i, rid in enumerate(df["review_id"].values)}
    shop_map    = {str(sid): i for i, sid in enumerate(df["shop_id"].unique())}
    product_map = {str(pid): i for i, pid in enumerate(df["product_id"].unique())}

    print(f"  Review nodes  : {len(review_map):,}")
    print(f"  Shop nodes    : {len(shop_map):,}")
    print(f"  Product nodes : {len(product_map):,}")

    # Node features
    print("Building node features...")
    review_feats  = make_review_features(df)
    shop_feats    = make_shop_features(df, shop_map)
    product_feats = make_product_features(df, product_map)

    # Labels 
    labels = torch.tensor(df["coordinated"].values, dtype=torch.long)

    # Edge indices + timestamps 
    print("Building edges...")
    review_idx  = torch.tensor([review_map[r]        for r in df["review_id"]], dtype=torch.long)
    shop_idx    = torch.tensor([shop_map[str(s)]     for s in df["shop_id"]],   dtype=torch.long)
    product_idx = torch.tensor([product_map[str(p)]  for p in df["product_id"]], dtype=torch.long)
    timestamps  = torch.tensor(df["timestamp"].values, dtype=torch.float)

    # Assemble HeteroData
    data = HeteroData()

    # Node features
    data["review"].x           = review_feats       
    data["review"].y           = labels            
    data["shop"].x             = shop_feats       
    data["product"].x          = product_feats      

    # review -> shop
    data["review", "posted_in", "shop"].edge_index = torch.stack([review_idx, shop_idx])
    data["review", "posted_in", "shop"].edge_attr  = timestamps.unsqueeze(1)

    # review -> product
    data["review", "reviews", "product"].edge_index = torch.stack([review_idx, product_idx])
    data["review", "reviews", "product"].edge_attr  = timestamps.unsqueeze(1)

    # shop -> product (unique pairs)
    shop_prod = df[["shop_id", "product_id"]].drop_duplicates()
    sp_shop_idx = torch.tensor([shop_map[str(s)]    for s in shop_prod["shop_id"]],    dtype=torch.long)
    sp_prod_idx = torch.tensor([product_map[str(p)] for p in shop_prod["product_id"]], dtype=torch.long)
    data["shop", "sells", "product"].edge_index = torch.stack([sp_shop_idx, sp_prod_idx])

    print(f"  Edges review->shop    : {review_idx.shape[0]:,}")
    print(f"  Edges review->product : {review_idx.shape[0]:,}")
    print(f"  Edges shop->product   : {sp_shop_idx.shape[0]:,}")

    return data, review_map, shop_map, product_map


# Main

def run():
    os.makedirs("data", exist_ok=True)

    data, review_map, shop_map, product_map = build_graph()

    torch.save(data, OUTPUT_GRAPH)
    torch.save({
        "review_map":  review_map,
        "shop_map":    shop_map,
        "product_map": product_map,
    }, OUTPUT_MAPS)

    print(f"\nSaved graph  -> {OUTPUT_GRAPH}")
    print(f"Saved maps   -> {OUTPUT_MAPS}")
    print(f"\nGraph summary:")
    print(data)
    print("=== Done ===")


if __name__ == "__main__":
    run()
