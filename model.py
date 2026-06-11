import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class MemoryModule(nn.Module):
    def __init__(self, memory_dim, n_nodes):
        super().__init__()
        self.memory_dim = memory_dim
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        self.register_buffer("memory", torch.zeros(n_nodes, memory_dim))

    def get(self, node_ids):
        return self.memory[node_ids]

    def update(self, node_ids, messages):
        current = self.memory[node_ids]
        updated = self.gru(messages, current)
        self.memory[node_ids] = updated.detach()

    def reset(self):
        self.memory.zero_()


class MessageFunction(nn.Module):
    def __init__(self, input_dim, memory_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )

    def forward(self, x):
        return self.msg_mlp(x)


class EmbeddingModule(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        # review harus jadi dst node agar ter-update
        self.conv1 = HeteroConv({
            ("review", "posted_in", "shop")       : SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("review", "reviews",   "product")    : SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("shop",   "sells",     "product")    : SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("shop",   "rev_posted_in", "review") : SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("product","rev_reviews",   "review") : SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("review", "posted_in", "shop")       : SAGEConv((hidden_dim, hidden_dim), out_dim),
            ("review", "reviews",   "product")    : SAGEConv((hidden_dim, hidden_dim), out_dim),
            ("shop",   "sells",     "product")    : SAGEConv((hidden_dim, hidden_dim), out_dim),
            ("shop",   "rev_posted_in", "review") : SAGEConv((hidden_dim, hidden_dim), out_dim),
            ("product","rev_reviews",   "review") : SAGEConv((hidden_dim, hidden_dim), out_dim),
        }, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TGNFakeReviewDetector(nn.Module):
    def __init__(
        self,
        review_feat_dim=768,
        shop_feat_dim=3,
        product_feat_dim=4,
        memory_dim=128,
        hidden_dim=256,
        out_dim=128,
        n_review_nodes=65348,
        dropout=0.3
    ):
        super().__init__()

        self.review_proj  = nn.Linear(review_feat_dim + memory_dim, hidden_dim)
        self.shop_proj    = nn.Linear(shop_feat_dim, hidden_dim)
        self.product_proj = nn.Linear(product_feat_dim, hidden_dim)

        self.memory           = MemoryModule(memory_dim, n_review_nodes)
        self.message_fn       = MessageFunction(hidden_dim, memory_dim)
        self.embedding_module = EmbeddingModule(hidden_dim, out_dim)
        self.classifier       = MLPClassifier(out_dim, hidden_dim // 2, dropout)

    def _build_edge_index_dict(self, edge_index_dict):
        """Tambah reverse edges agar review jadi dst node."""
        extended = dict(edge_index_dict)
        # shop -> review (reverse of review -> shop)
        if ("review", "posted_in", "shop") in edge_index_dict:
            ei = edge_index_dict[("review", "posted_in", "shop")]
            extended[("shop", "rev_posted_in", "review")] = ei.flip(0)
        # product -> review (reverse of review -> product)
        if ("review", "reviews", "product") in edge_index_dict:
            ei = edge_index_dict[("review", "reviews", "product")]
            extended[("product", "rev_reviews", "review")] = ei.flip(0)
        return extended

    def forward(self, x_dict, edge_index_dict, review_node_ids, timestamps=None):
        # Gabungkan fitur review + memory (full graph)
        full_mem      = self.memory.memory
        full_combined = torch.cat([x_dict["review"], full_mem], dim=-1)

        x_proj = {
            "review" : self.review_proj(full_combined),
            "shop"   : self.shop_proj(x_dict["shop"]),
            "product": self.product_proj(x_dict["product"]),
        }

        # Tambah reverse edges
        extended_edges = self._build_edge_index_dict(edge_index_dict)

        emb_dict   = self.embedding_module(x_proj, extended_edges)
        review_emb = emb_dict.get("review", x_proj["review"])

        # Prediksi hanya untuk batch review_node_ids
        logits = self.classifier(review_emb[review_node_ids])
        return logits

    def update_memory(self, review_node_ids, x_dict, edge_index_dict):
        mem      = self.memory.get(review_node_ids)
        feat     = x_dict["review"][review_node_ids]
        combined = torch.cat([feat, mem], dim=-1)
        proj     = self.review_proj(combined)
        # Slice ke memory_dim (128)
        self.memory.update(review_node_ids, proj.detach()[:, :128])

    def reset_memory(self):
        self.memory.reset()


def build_model(n_review_nodes, review_feat_dim=768, device="cpu"):
    return TGNFakeReviewDetector(
        review_feat_dim=review_feat_dim,
        shop_feat_dim=3,
        product_feat_dim=4,
        memory_dim=128,
        hidden_dim=256,
        out_dim=128,
        n_review_nodes=n_review_nodes,
        dropout=0.3
    ).to(device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    N_review, N_shop, N_product = 100, 10, 20
    model = build_model(n_review_nodes=N_review, device=device)

    x_dict = {
        "review" : torch.randn(N_review, 768).to(device),
        "shop"   : torch.randn(N_shop, 3).to(device),
        "product": torch.randn(N_product, 4).to(device),
    }
    edge_index_dict = {
        ("review", "posted_in", "shop")   : torch.stack([torch.arange(N_review), torch.randint(0, N_shop, (N_review,))]).to(device),
        ("review", "reviews", "product")  : torch.stack([torch.arange(N_review), torch.randint(0, N_product, (N_review,))]).to(device),
        ("shop", "sells", "product")      : torch.stack([torch.arange(N_shop), torch.randint(0, N_product, (N_shop,))]).to(device),
    }

    logits = model(x_dict, edge_index_dict, torch.arange(N_review).to(device))
    print(f"output shape : {logits.shape}")  # [100]
    print(f"params       : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("model OK!")
