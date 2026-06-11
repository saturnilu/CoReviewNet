import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Config 
CONFIGS = [
    (
        "xlm-roberta-base",
        "data/graph_data_xlmr.pt",
        "data/review_embeddings_xlmr.pt"
    ),
    (
        "bert-base-multilingual-cased",
        "data/graph_data_mbert.pt",
        "data/review_embeddings_mbert.pt"
    )
] 
BATCH_SIZE   = 64
MAX_LENGTH   = 128
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_CSV    = "data/processed_reviews.csv"
INPUT_GRAPH  = "data/graph_data.pt"

# Embedding extractor 

def get_embeddings(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    device: str = DEVICE,
) -> torch.Tensor:
    
    model.eval()
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)


        cls_embeddings = output.last_hidden_state[:, 0, :] 
        all_embeddings.append(cls_embeddings.cpu())

        if (start // batch_size + 1) % 10 == 0:
            done = min(start + batch_size, len(texts))
            print(f"  Encoded {done:,}/{len(texts):,} reviews...")

    return torch.cat(all_embeddings, dim=0)  


# Main

def run():

    print("=== Feature Extraction ===")
    print(f"Device: {DEVICE}")

    df = pd.read_csv(INPUT_CSV)
    texts = df["review_text_clean"].fillna("").tolist()

    print(f"Reviews to encode: {len(texts):,}")

    for model_name, output_graph, output_embs in CONFIGS:

        if (
            os.path.exists(output_graph)
            and os.path.exists(output_embs)
        ):
            print(
                f"[SKIP] {model_name} already extracted"
            )
            continue

    for model_name, output_graph, output_embs in CONFIGS:

        print("\n" + "=" * 50)
        print(f"Model : {model_name}")

        data = torch.load(
            INPUT_GRAPH,
            weights_only=False
        )

        if len(texts) != data["review"].num_nodes:
            raise ValueError(
                f"Review count mismatch: "
                f"{len(texts)} vs "
                f"{data['review'].num_nodes}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModel.from_pretrained(
            model_name
        ).to(DEVICE)

        embeddings = get_embeddings(
            texts,
            tokenizer,
            model
        )

        print(
            f"Embeddings shape: {embeddings.shape}"
        )

        data["review"].x = embeddings

        print(
            "review feature shape:",
            data["review"].x.shape
        )

        torch.save(
            data,
            output_graph
        )

        torch.save(
            embeddings,
            output_embs
        )

        print(
            f"Saved graph -> {output_graph}"
        )

        print(
            f"Saved embeddings -> {output_embs}"
        )

    print("\n=== Done ===")


if __name__ == "__main__":
    run()
